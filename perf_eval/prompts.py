"""Compact eval prompt set for per-change regression checks.

Twelve prompts: 3 lengths x 4 emotions (sad, normal, angry, excited), to exercise
short/medium/long generation across the model's full emotional range (it is a
prosody-focused model) without the cost of the full 60-prompt profile. Every
sentence is taken verbatim from the profiler's canonical set (profile_misotts.py)
so a reader knows exactly what the model is asked to say and can verify the
transcript independently. Caps are generous:
the model speaks ~1.1 words/s on these, so a 30-word "medium" line reaches its
natural end-of-speech at ~27 s, where the old 10/15/22 s caps chopped clips
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
    # ---- angry, appended (the third emotion per length, taken verbatim from the
    # profiler's canonical set in profile_misotts.py). Appended rather than
    # interleaved so the original six keep their per-id seeds and stored baselines.
    ("short_angry",   "I can't believe you would lie straight to my face like that!", 20_000),
    ("medium_angry",  "You took the money I was saving for years, lied straight to my face about it, and now you have the audacity to ask me for more. Unbelievable!", 40_000),
    ("long_angry",    "I have had it up to here with your constant excuses, your endless lies, and your complete lack of respect for anyone but yourself. Every single time I give you another chance, you find a new way to disappoint me, and I am absolutely done with it!", 50_000),
    # ---- excited, appended (the fourth emotion per length; this is a prosody-
    # focused model, so the eval should cover its full emotional range). Verbatim
    # from profile_misotts.py.
    ("short_excited", "Oh my god, I just got accepted into my absolute dream school!", 20_000),
    ("medium_excited", "I cannot believe we actually pulled it off! After months of working day and night, our startup just got funded, and we're moving to Silicon Valley next month!", 40_000),
    ("long_excited",  "You will not believe what happened to me today! I was walking down the street, minding my own business, when this casting director stopped me and asked if I'd be interested in auditioning for a major motion picture! Can you imagine? Me, in a Hollywood movie!", 50_000),
]

# Quick subset (one per length) for fast per-change validation.
QUICK_IDS = ["short_sad", "medium_normal", "long_sad"]
QUICK_PROMPTS = [p for p in EVAL_PROMPTS if p[0] in QUICK_IDS]
