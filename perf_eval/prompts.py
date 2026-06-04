"""Compact eval prompt set for per-change regression checks.

Six prompts: 3 lengths x 2 emotions (sad, normal), to exercise short/medium/long
generation without the cost of the full 60-prompt profile. Caps are generous:
the model speaks ~1.1 words/s on these, so a 30-word "medium" line reaches its
natural end-of-speech at ~27 s — the old 10/15/22 s caps chopped clips
mid-sentence. Caps now sit well above natural length; clips that finish early
stop at EOS regardless. Each prompt has a stable id so baseline and candidate
renders line up for paired comparison.
"""
from __future__ import annotations

# (id, text, max_audio_length_ms)
EVAL_PROMPTS = [
    # ---- short ----
    ("short_sad",     "I just feel so empty inside today, like nothing matters anymore.", 20_000),
    ("short_normal",  "I went to the grocery store this morning to buy some fresh bread.", 20_000),
    # ---- medium ----
    ("medium_sad",    "The doctor said there's nothing more they can do. We tried every treatment, every option, and now I'm just sitting here waiting to say goodbye.", 40_000),
    ("medium_normal", "I went to the farmers market this morning and picked up some fresh tomatoes, basil, and a loaf of sourdough bread. I'm planning to make pasta for dinner.", 40_000),
    # ---- long ----
    ("long_sad",      "It's been three months since the accident, and I still wake up every morning expecting to hear his footsteps in the kitchen, the coffee maker running, his soft humming. The silence is what hurts the most, the absolute, devastating silence of an empty house.", 50_000),
    ("long_normal",   "The renovation project at the downtown library is finally complete after eighteen months of construction. The new wing includes a children's reading area, several meeting rooms, and a small cafe. The grand reopening is scheduled for next Saturday at noon, and everyone in the community is invited.", 50_000),
]

# Quick subset (one per length) for fast per-change validation.
QUICK_IDS = ["short_sad", "medium_normal", "long_sad"]
QUICK_PROMPTS = [p for p in EVAL_PROMPTS if p[0] in QUICK_IDS]
