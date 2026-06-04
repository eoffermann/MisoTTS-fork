"""End-truncation detection for MisoTTS, wired into perceval as gating metrics.

User-reported symptom: "many/most clips get cut off at the very end." This adds
two baseline-free metrics so perceval FAILS a render when truncation happens
(works even on the baseline itself, where there is no prior reference):

  tier1.tail_truncation   acoustic: RMS of the final ~120 ms divided by the
                          clip's typical speech-level RMS. A natural utterance
                          trails into near-silence (ratio low); an abruptly cut
                          clip ends at speech energy (ratio near/above 1.0).

  tier1.tail_word_missing semantic: using Whisper, if the transcript is a clean
                          PREFIX of the target text (it covers the start but
                          stops early), the fraction of target words missing
                          from the end. Conservative — only fires on an actual
                          prefix match, so ASR noise rarely false-positives.

Both are registered into perceval's REGISTRY and their thresholds injected into
perceval.aggregation.thresholds.THRESHOLDS at import, so the native verdict
logic (aggregation.decision.classify_pair) gates on them. We keep this in our
own committed module rather than editing the vendored perceval copy.

Thresholds are uniform across regimes: truncation is always a defect, never an
expected consequence of an optimization.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from perceval.metrics_registry import MetricEntry, register
from perceval.types import MetricResult, Pair
from perceval.aggregation import thresholds as _th
from perceval.aggregation.thresholds import Threshold
from perceval.types import Regime

# Tunable cutoffs (calibrated against the smoke diagnostic).
TAIL_MS = 0.120          # window at the end of the clip
SPEECH_PCTILE = 90       # percentile of frame RMS taken as "speech level"
TAIL_FAIL = 0.50         # tail/speech ratio above this == abrupt cutoff -> FAIL
TAIL_FLAG = 0.35
WORDMISS_FAIL = 0.15     # >15% of target words missing off the end -> FAIL
WORDMISS_FLAG = 0.05


def _load_mono(path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def tail_truncation(pair: Pair) -> MetricResult:
    try:
        audio, sr = _load_mono(pair.candidate.audio_path)
        n = audio.shape[0]
        if n < int(0.25 * sr):
            return MetricResult(name="tier1.tail_truncation", value=0.0,
                                higher_is_better=False,
                                extras={"note": "clip too short to assess"})
        fw = max(1, int(0.02 * sr))            # 20 ms frames
        nf = n // fw
        frames = audio[:nf * fw].reshape(nf, fw)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        speech = float(np.percentile(rms, SPEECH_PCTILE))
        tail_n = int(TAIL_MS * sr)
        tail_rms = float(np.sqrt((audio[-tail_n:] ** 2).mean() + 1e-12))
        ratio = min(tail_rms / speech, 2.0) if speech > 0 else 0.0
        return MetricResult(
            name="tier1.tail_truncation", value=float(ratio),
            higher_is_better=False,
            extras={"tail_rms": tail_rms, "speech_rms_p%d" % SPEECH_PCTILE: speech,
                    "duration_s": n / sr},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return MetricResult(name="tier1.tail_truncation",
                            error=f"{type(exc).__name__}: {exc}")


def tail_word_missing(pair: Pair) -> MetricResult:
    try:
        from perceval.tier1.asr import transcribe, normalize_for_wer
        t = transcribe(pair.candidate.audio_path)
        if t.get("skipped"):
            return MetricResult(name="tier1.tail_word_missing",
                                error="ASR unavailable")
        ref = normalize_for_wer(pair.text).split()
        hyp = normalize_for_wer(t["text"]).split()
        if not ref:
            return MetricResult(name="tier1.tail_word_missing", value=0.0,
                                higher_is_better=False, extras={"note": "no ref"})
        # Strict prefix: transcript matches the start of the target but is
        # shorter -> the end was cut. Allow the last hyp word to be partial.
        k = len(hyp)
        is_prefix = 0 < k < len(ref) and hyp[:k - 1] == ref[:k - 1]
        missing = (len(ref) - k) / len(ref) if is_prefix else 0.0
        return MetricResult(
            name="tier1.tail_word_missing", value=float(missing),
            higher_is_better=False,
            extras={"is_prefix": bool(is_prefix), "ref_words": len(ref),
                    "hyp_words": k, "transcript": t["text"],
                    "missing_tail": ref[k:] if is_prefix else []},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return MetricResult(name="tier1.tail_word_missing",
                            error=f"{type(exc).__name__}: {exc}")


def _inject_thresholds() -> None:
    both = lambda fail, flag: {
        Regime.DETERMINISTIC: Threshold(fail=fail, flag=flag, note="end-truncation gate"),
        Regime.STOCHASTIC: Threshold(fail=fail, flag=flag, note="end-truncation gate"),
    }
    _th.THRESHOLDS.setdefault("tier1.tail_truncation", both(TAIL_FAIL, TAIL_FLAG))
    _th.THRESHOLDS.setdefault("tier1.tail_word_missing", both(WORDMISS_FAIL, WORDMISS_FLAG))


def install() -> None:
    """Register the metrics and gates. Idempotent (perceval.register dedupes)."""
    register(MetricEntry(name="tier1.tail_truncation", run=tail_truncation,
                         applies=lambda p: True,
                         description="Acoustic end-of-clip abruptness (tail RMS / speech RMS)."))
    register(MetricEntry(name="tier1.tail_word_missing", run=tail_word_missing,
                         applies=lambda p: bool(p.text),
                         description="Fraction of target words missing off the end (ASR prefix)."))
    _inject_thresholds()
