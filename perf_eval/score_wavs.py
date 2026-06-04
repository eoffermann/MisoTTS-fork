"""Score a directory of pre-rendered WAVs with perceval vs the stored baseline.

Used to quality-gate output that was produced elsewhere (e.g. the compiled
container), where the WAVs already exist on disk. Builds Pairs from <cand_dir>/
<id>.wav against perf_eval/baseline/<id>.wav, runs perceval (stochastic regime by
default, since compile/quantization legitimately change numerics), and reports
per-clip quality plus the candidate-vs-baseline regression check.

  .venv/Scripts/python.exe -m perf_eval.score_wavs <cand_dir> [deterministic|stochastic]
"""
import os
import sys

from perf_eval import truncation

truncation.install()

from perceval import Pair, Regime, Sample, evaluate_batch  # noqa: E402
from perf_eval.prompts import EVAL_PROMPTS  # noqa: E402
from perf_eval.run_eval import (REPORTS_DIR, _load_baseline_metrics,  # noqa: E402
                                compare_to_baseline)

HERE = os.path.dirname(os.path.abspath(__file__))
# Baseline dir is configurable so this can also compare two ad-hoc render dirs
# (e.g. greedy compiled vs greedy eager for a deterministic parity check).
BASE_DIR = os.environ.get("MISO_BASELINE_DIR", os.path.join(HERE, "baseline"))
_STD_BASELINE = os.path.join(HERE, "baseline")


def main():
    cand_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")
    regime = Regime.parse(sys.argv[2]) if len(sys.argv) > 2 else Regime.STOCHASTIC
    pairs = []
    for pid, text, _ in EVAL_PROMPTS:
        cw = os.path.join(cand_dir, pid + ".wav")
        if not os.path.exists(cw):
            print(f"  (missing candidate {cw})", flush=True)
            continue
        bw = os.path.join(BASE_DIR, pid + ".wav")
        base = Sample(audio_path=bw, text=text, sample_id=pid + "_base") if os.path.exists(bw) else None
        pairs.append(Pair(text=text, candidate=Sample(audio_path=cw, text=text, sample_id=pid),
                          baseline=base, regime=regime, pair_id=pid))
    if not pairs:
        print("no candidate wavs found in", cand_dir)
        return
    print(f"scoring {len(pairs)} clips from {cand_dir} (regime={regime.value})...", flush=True)
    sr = evaluate_batch(pairs, eval_set_name="prerendered")
    print(f"\nset verdict: {sr.set_verdict.value.upper()}  "
          f"fail_rate={sr.failure_rate:.2f} flag_rate={sr.flag_rate:.2f}")

    def g(m):
        return f"{m.value:.3f}" if (m and m.computed) else "-"

    for r in sr.results:
        print(f"  {r.pair_id:14} {r.verdict.value:5} "
              f"tail={g(r.metrics.get('tier1.tail_truncation'))} "
              f"cer={g(r.metrics.get('tier1.asr.cer'))} "
              f"wer={g(r.metrics.get('tier1.asr.wer'))} "
              f"mos={g(r.metrics.get('tier2.utmos.mos'))} "
              f"mcd={g(r.metrics.get('tier3.mcd_dtw'))} "
              f"ssl={g(r.metrics.get('tier3.ssl.cosine_dtw'))}")

    bj = os.path.join(REPORTS_DIR, "baseline.json")
    # Regression-vs-baseline only makes sense against the standard baseline set.
    if os.path.abspath(BASE_DIR) == os.path.abspath(_STD_BASELINE) and os.path.exists(bj):
        regs, imps, _ = compare_to_baseline(sr, _load_baseline_metrics(bj), regime.value)
        print(f"\nREGRESSIONS vs baseline: {len(regs)}")
        for pid, name, b, c, d in regs:
            print(f"  REGRESS {pid:14} {name:22} {b:.3f} -> {c:.3f} ({d:+.3f})")
        print(f"IMPROVEMENTS vs baseline: {len(imps)}")
        for pid, name, b, c, d in imps:
            print(f"  IMPROVE  {pid:14} {name:22} {b:.3f} -> {c:.3f} ({d:+.3f})")


if __name__ == "__main__":
    main()
