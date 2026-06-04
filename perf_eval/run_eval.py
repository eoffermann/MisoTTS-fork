"""MisoTTS perf+quality eval driver.

Two modes:
  --baseline           Render the eval set with the CURRENT code and store it as
                       the reference. Runs perceval no-reference metrics (ASR,
                       artifacts, MOS) + the truncation gate to characterize the
                       baseline (incl. how many clips truncate).
  (default, candidate) Render with the current code, then compare each clip
                       against the stored baseline render under the chosen regime
                       and emit a PASS/FAIL/FLAG verdict + a perf delta.

Quality gate = perceval (+ our truncation metrics). Perf gate = wall/RTF deltas.
Exit code: 0 PASS, 1 FAIL, 2 FLAG, 3 ERROR.

Run with the venv interpreter:
  .\.venv\Scripts\python.exe -m perf_eval.run_eval --baseline --label baseline
  .\.venv\Scripts\python.exe -m perf_eval.run_eval --label f3_eos_sync --regime deterministic
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

_START = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _START:7.1f}s] {msg}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(HERE, "baseline")
RUNS_DIR = os.path.join(HERE, "runs")
REPORTS_DIR = os.path.join(HERE, "reports")

# Metrics that matter most for a fast per-change check (substring match).
FAST_SUBSET = ["tier1", "tier2.utmos", "tier3.mcd_dtw", "tier3.mel.l1", "tier3.ssl"]

KEY_METRICS = ["tier1.tail_truncation", "tier1.tail_word_missing", "tier1.asr.cer",
               "tier1.asr.wer", "tier1.asr.cer_delta", "tier2.utmos.mos",
               "tier3.mcd_dtw", "tier3.ssl.cosine_dtw"]


def _verdict_exit(v) -> int:
    from perceval.types import Verdict
    return {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.FLAG: 2, Verdict.ERROR: 3}[v]


# Candidate-vs-baseline regression gate (D7): the baseline itself FAILs absolute
# thresholds (pre-existing truncation), so we judge a change on whether it makes
# any key metric WORSE than the baseline, not on the absolute set verdict.
# (metric -> (higher_is_better, regression_tolerance))
#
# Tolerances are sized to perceptual significance, not metric noise (calibrated
# on the first real run, 2026-06-04). UTMOS is a noisy single-number MOS
# predictor (~0.1-0.3 jitter per clip) and clips here sit at 4.0+, so only a
# >0.30 drop is meaningful. tail_truncation only matters as it approaches its
# 0.35 FLAG band, so sub-0.15 jitter between two already-clean clips is ignored.
# ASR error tolerances stay tight (those deltas are perceptually real).
REGRESSION_GATES = {
    "tier1.tail_truncation":  (False, 0.15),
    "tier1.tail_word_missing":(False, 0.10),
    "tier1.asr.cer":          (False, 0.03),
    "tier1.asr.wer":          (False, 0.05),
    "tier2.utmos.mos":        (True,  0.30),
}
# Reference-distance parity gate, DETERMINISTIC regime only: an output-preserving
# change should keep candidate≈baseline, i.e. near-zero reference distance.
PARITY_GATES = {"tier3.mcd_dtw": 2.0, "tier3.ssl.cosine_dtw": 0.05}


def _load_baseline_metrics(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data.get("results", []):
        out[r["pair_id"]] = {k: v.get("value") for k, v in r.get("metrics", {}).items()}
    return out


def compare_to_baseline(set_result, base_metrics, regime):
    """Return (regressions, improvements, parity_breaches) as lists of tuples."""
    regressions, improvements = [], []
    for r in set_result.results:
        bm = base_metrics.get(r.pair_id, {})
        for name, (hib, tol) in REGRESSION_GATES.items():
            cm = r.metrics.get(name)
            if not (cm and cm.computed) or bm.get(name) is None:
                continue
            delta = cm.value - bm[name]
            worse = (delta < -tol) if hib else (delta > tol)
            better = (delta > tol) if hib else (delta < -tol)
            if worse:
                regressions.append((r.pair_id, name, bm[name], cm.value, delta))
            elif better:
                improvements.append((r.pair_id, name, bm[name], cm.value, delta))
    parity = []
    if regime == "deterministic":
        for r in set_result.results:
            for name, thr in PARITY_GATES.items():
                cm = r.metrics.get(name)
                if cm and cm.computed and cm.value is not None and cm.value > thr:
                    parity.append((r.pair_id, name, cm.value, thr))
    return regressions, improvements, parity


def write_report(path, set_result, perf_records, baseline_perf, label, regime):
    lines = [f"# Eval report — {label}", "",
             f"- Regime: {regime}",
             f"- Set verdict: **{set_result.set_verdict.value.upper()}**",
             f"- Failure rate: {set_result.failure_rate:.2f}  Flag rate: {set_result.flag_rate:.2f}",
             ""]
    # Quality table
    lines.append("## Per-clip quality")
    lines.append("| id | verdict | tail_trunc | word_miss | cer | wer | utmos | mcd | reasons |")
    lines.append("|----|---------|-----------:|----------:|----:|----:|------:|----:|---------|")
    for r in set_result.results:
        m = r.metrics
        def g(name):
            x = m.get(name)
            return f"{x.value:.3f}" if (x and x.computed) else "–"
        reasons = "; ".join(r.reasons)[:80]
        lines.append(f"| {r.pair_id} | {r.verdict.value} | {g('tier1.tail_truncation')} "
                     f"| {g('tier1.tail_word_missing')} | {g('tier1.asr.cer')} | {g('tier1.asr.wer')} "
                     f"| {g('tier2.utmos.mos')} | {g('tier3.mcd_dtw')} | {reasons} |")
    lines.append("")
    # Perf table
    lines.append("## Per-clip performance")
    if baseline_perf:
        lines.append("| id | wall(s) | base wall(s) | Δwall | rtf | base rtf | audio(s) |")
        lines.append("|----|--------:|-------------:|------:|----:|---------:|---------:|")
        for rec in perf_records:
            b = baseline_perf.get(rec["id"], {})
            bw = b.get("wall_s")
            dw = f"{rec['wall_s'] - bw:+.2f}" if bw else "–"
            lines.append(f"| {rec['id']} | {rec['wall_s']:.2f} | {bw:.2f} | {dw} | {rec['rtf']:.2f} "
                         f"| {b.get('rtf', float('nan')):.2f} | {rec['audio_s']:.2f} |"
                         if bw else
                         f"| {rec['id']} | {rec['wall_s']:.2f} | – | – | {rec['rtf']:.2f} | – | {rec['audio_s']:.2f} |")
    else:
        lines.append("| id | wall(s) | rtf | audio(s) |")
        lines.append("|----|--------:|----:|---------:|")
        for rec in perf_records:
            lines.append(f"| {rec['id']} | {rec['wall_s']:.2f} | {rec['rtf']:.2f} | {rec['audio_s']:.2f} |")
    tot = sum(r["wall_s"] for r in perf_records)
    aud = sum(r["audio_s"] for r in perf_records)
    lines.append("")
    lines.append(f"**Totals:** wall={tot:.1f}s  audio={aud:.1f}s  mean RTF={tot/aud:.2f}"
                 + (f"  (baseline wall={sum(baseline_perf[r['id']]['wall_s'] for r in perf_records if r['id'] in baseline_perf):.1f}s)" if baseline_perf else ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--baseline", action="store_true", help="store this render as the reference")
    ap.add_argument("--regime", default="deterministic", choices=["deterministic", "stochastic"])
    ap.add_argument("--fast", action="store_true", help="run a reduced metric subset")
    ap.add_argument("--quick", action="store_true", help="render only the 3-clip quick subset")
    args = ap.parse_args()

    # Install our truncation metrics + gates before importing the pipeline graph.
    log("Importing perceval + truncation gate...")
    from perf_eval import truncation
    truncation.install()
    from perceval import evaluate_batch, Pair, Sample, Regime
    from perceval.types import Verdict

    log("Importing torch + MisoTTS generator (model load follows)...")
    from generator import load_miso_8b

    log("Loading model...")
    t_load = time.perf_counter()
    gen = load_miso_8b("cuda")
    log(f"Model loaded in {time.perf_counter() - t_load:.1f}s.")

    from perf_eval.render import render_set
    from perf_eval.prompts import EVAL_PROMPTS, QUICK_PROMPTS
    prompts = QUICK_PROMPTS if args.quick else EVAL_PROMPTS
    out_dir = BASELINE_DIR if args.baseline else os.path.join(RUNS_DIR, args.label)
    log(f"Rendering {'quick' if args.quick else 'full'} eval set ({len(prompts)} clips) -> {out_dir}")
    perf_records = render_set(gen, out_dir, prompts=prompts, log=log)

    # Persist perf for this run.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "perf.json"), "w") as f:
        json.dump({r["id"]: r for r in perf_records}, f, indent=2)

    regime = Regime.parse(args.regime)
    baseline_perf = None
    pairs = []
    for rec in perf_records:
        cand = Sample(audio_path=rec["wav"], text=rec["text"], sample_id=rec["id"])
        base = None
        if not args.baseline:
            bwav = os.path.join(BASELINE_DIR, f"{rec['id']}.wav")
            if os.path.exists(bwav):
                base = Sample(audio_path=bwav, text=rec["text"], sample_id=f"{rec['id']}_base")
        pairs.append(Pair(text=rec["text"], candidate=cand, baseline=base,
                          regime=regime, pair_id=rec["id"]))

    if not args.baseline and os.path.exists(os.path.join(BASELINE_DIR, "perf.json")):
        with open(os.path.join(BASELINE_DIR, "perf.json")) as f:
            baseline_perf = json.load(f)

    subset = FAST_SUBSET if args.fast else None
    log(f"Running perceval ({'fast subset' if subset else 'all metrics'})... "
        f"(first run downloads ASR/MOS/SSL models)")
    set_result = evaluate_batch(pairs, metrics_subset=subset, eval_set_name=args.label)

    # Report
    n_trunc = sum(1 for r in set_result.results
                  for nm in ("tier1.tail_truncation", "tier1.tail_word_missing")
                  if (mm := r.metrics.get(nm)) and mm.computed and mm.passed is False)
    log(f"Set verdict: {set_result.set_verdict.value.upper()}  "
        f"failure_rate={set_result.failure_rate:.2f} flag_rate={set_result.flag_rate:.2f}")
    for r in set_result.results:
        tt = r.metrics.get("tier1.tail_truncation")
        wm = r.metrics.get("tier1.tail_word_missing")
        ttv = f"{tt.value:.2f}" if (tt and tt.computed) else "n/a"
        wmv = f"{wm.value:.2f}" if (wm and wm.computed) else "n/a"
        log(f"  {r.pair_id:14} {r.verdict.value:5}  tail_trunc={ttv}  word_miss={wmv}")
    report_path = os.path.join(REPORTS_DIR, f"{args.label}.md")
    write_report(report_path, set_result, perf_records, baseline_perf, args.label, args.regime)
    with open(os.path.join(REPORTS_DIR, f"{args.label}.json"), "w") as f:
        f.write(set_result.to_json())
    log(f"Wrote {report_path}")

    # Truncation summary (the user's concern).
    trunc_ids = [r.pair_id for r in set_result.results
                 if (mm := r.metrics.get("tier1.tail_truncation")) and mm.computed and mm.value >= truncation.TAIL_FAIL]
    log(f"TRUNCATION: {len(trunc_ids)}/{len(set_result.results)} clips flagged by acoustic gate: {trunc_ids}")

    if args.baseline:
        raise SystemExit(_verdict_exit(set_result.set_verdict))

    # ---- Candidate: regression gate vs baseline (D7) ----
    base_json = os.path.join(REPORTS_DIR, "baseline.json")
    if not os.path.exists(base_json):
        log("No baseline.json found; cannot regression-gate. Run with --baseline first.")
        raise SystemExit(3)
    base_metrics = _load_baseline_metrics(base_json)
    regs, imps, parity = compare_to_baseline(set_result, base_metrics, args.regime)
    log(f"REGRESSIONS vs baseline: {len(regs)}")
    for pid, name, b, c, d in regs:
        log(f"  REGRESS {pid:14} {name:24} {b:.3f} -> {c:.3f} ({d:+.3f})")
    log(f"IMPROVEMENTS vs baseline: {len(imps)}")
    for pid, name, b, c, d in imps:
        log(f"  IMPROVE  {pid:14} {name:24} {b:.3f} -> {c:.3f} ({d:+.3f})")
    if args.regime == "deterministic":
        log(f"PARITY breaches (reference distance too high): {len(parity)}")
        for pid, name, v, thr in parity:
            log(f"  PARITY   {pid:14} {name:24} {v:.3f} > {thr}")

    # Perf delta vs baseline.
    if baseline_perf:
        cand_wall = sum(r["wall_s"] for r in perf_records)
        base_wall = sum(baseline_perf[r["id"]]["wall_s"] for r in perf_records
                        if r["id"] in baseline_perf)
        pct = 100 * (cand_wall - base_wall) / base_wall if base_wall else 0.0
        log(f"PERF: candidate wall={cand_wall:.1f}s vs baseline {base_wall:.1f}s "
            f"({pct:+.1f}%); {'FASTER' if pct < 0 else 'slower'}")

    parity_block = (args.regime == "deterministic" and len(parity) > 0)
    gate_ok = (len(regs) == 0) and not parity_block
    log(f"COMMIT GATE: {'PASS — no regressions' if gate_ok else 'BLOCKED'}"
        + (f" ({len(regs)} regressions)" if regs else "")
        + (f" ({len(parity)} parity breaches)" if parity_block else ""))
    raise SystemExit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
