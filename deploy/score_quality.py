"""Quality-score the rendered quant variants (bf16/int8/int4) with perceval and
emit a per-level CER/WER/UTMOS summary for PERFORMANCE_PROGRESS.md.

Runs on the HOST (perceval + ASR/MOS models live there, not in the serving
image), reading the WAVs produced by deploy/render_quality.py. int8/int4 are
scored with the bf16 render as the fidelity baseline (MCD/SSL), so the table
shows both absolute intelligibility/MOS and how far each quant drifts from
full precision on identical prompts and seeds.

  .venv/Scripts/python.exe deploy/score_quality.py [quality_dir]
"""
import json
import os
import sys

from perf_eval import truncation

truncation.install()

from perceval import Pair, Regime, Sample, evaluate_batch  # noqa: E402
from perf_eval.prompts import EVAL_PROMPTS  # noqa: E402

QDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "quality")
LEVELS = ["bf16", "int8", "int4"]
BF16_DIR = os.path.join(QDIR, "bf16")


def _val(metrics, key):
    m = metrics.get(key)
    return m.value if (m and m.computed) else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    out = {}
    for lvl in LEVELS:
        d = os.path.join(QDIR, lvl)
        if not os.path.isdir(d):
            print(f"(skip {lvl}: {d} missing)")
            continue
        pairs = []
        for pid, text, _ in EVAL_PROMPTS:
            cw = os.path.join(d, pid + ".wav")
            if not os.path.exists(cw):
                print(f"  (missing {cw})")
                continue
            bw = os.path.join(BF16_DIR, pid + ".wav")
            base = (Sample(audio_path=bw, text=text, sample_id=pid + "_bf16")
                    if (lvl != "bf16" and os.path.exists(bw)) else None)
            pairs.append(Pair(text=text,
                              candidate=Sample(audio_path=cw, text=text, sample_id=pid),
                              baseline=base, regime=Regime.STOCHASTIC, pair_id=pid))
        if not pairs:
            continue
        print(f"\n=== scoring {lvl} ({len(pairs)} clips) ===", flush=True)
        sr = evaluate_batch(pairs, eval_set_name=lvl)
        rows = []
        for r in sr.results:
            rows.append({
                "id": r.pair_id, "verdict": r.verdict.value,
                "cer": _val(r.metrics, "tier1.asr.cer"),
                "wer": _val(r.metrics, "tier1.asr.wer"),
                "mos": _val(r.metrics, "tier2.utmos.mos"),
                "tail": _val(r.metrics, "tier1.tail_truncation"),
                "mcd": _val(r.metrics, "tier3.mcd_dtw"),
                "ssl": _val(r.metrics, "tier3.ssl.cosine_dtw"),
            })
        agg = {"cer": _mean([x["cer"] for x in rows]),
               "wer": _mean([x["wer"] for x in rows]),
               "mos": _mean([x["mos"] for x in rows])}
        out[lvl] = {"set_verdict": sr.set_verdict.value, "fail_rate": sr.failure_rate,
                    "flag_rate": sr.flag_rate, "mean": agg, "rows": rows}

        def f(x, p="{:.3f}"):
            return p.format(x) if x is not None else "  -  "
        print(f"  [{lvl}] verdict={sr.set_verdict.value.upper()} "
              f"fail={sr.failure_rate:.2f} flag={sr.flag_rate:.2f} | "
              f"mean CER={f(agg['cer'])} WER={f(agg['wer'])} UTMOS={f(agg['mos'])}")
        for r in rows:
            print(f"    {r['id']:14} {r['verdict']:5} cer={f(r['cer'])} wer={f(r['wer'])} "
                  f"mos={f(r['mos'])} tail={f(r['tail'])} mcd={f(r['mcd'])} ssl={f(r['ssl'])}")

    with open(os.path.join(QDIR, "quality_scores.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(QDIR, 'quality_scores.json')}")

    # Compact comparison for the diary.
    print("\n=== per-level means (for PERFORMANCE_PROGRESS.md) ===")
    print(f"{'level':6} {'verdict':8} {'CER':>7} {'WER':>7} {'UTMOS':>7} {'fail':>5} {'flag':>5}")
    for lvl in LEVELS:
        if lvl not in out:
            continue
        o = out[lvl]; m = o["mean"]
        def f(x):
            return f"{x:.3f}" if x is not None else "-"
        print(f"{lvl:6} {o['set_verdict']:8} {f(m['cer']):>7} {f(m['wer']):>7} "
              f"{f(m['mos']):>7} {o['fail_rate']:>5.2f} {o['flag_rate']:>5.2f}")


if __name__ == "__main__":
    main()
