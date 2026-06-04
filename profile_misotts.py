import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

import time
from datetime import datetime
from pathlib import Path

# Wall-clock anchor so every log line shows elapsed time since the process
# started — handy for spotting where the long stalls actually are.
_START = time.perf_counter()


def log(msg: str) -> None:
    """Timestamped, flushed progress line.

    flush=True matters: on Windows, piped/redirected stdout is block-buffered,
    so without it these messages wouldn't appear until a buffer fills — exactly
    the "is it locked up?" silence we're trying to kill.
    """
    print(f"[{time.perf_counter() - _START:7.1f}s] {msg}", flush=True)


log("Starting profiler. Importing torch (this pulls in CUDA libs, ~10-30s cold)...")
import torch

log(f"torch {torch.__version__} imported. Importing torchaudio...")
import torchaudio  # type: ignore

log("torchaudio imported. Importing generator (load_miso_8b)...")
from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b

log("Imports complete.")


# Each entry: target audio length in seconds + a generous cap (max_audio_length_ms)
# + 5 sentences per emotion (sad / angry / normal / excited).
#
# Caps are set well above the target lengths: the model speaks far slower than a
# ~2.5 words/sec estimate would suggest (measured ~1.1 words/sec on these
# prompts, e.g. a 30-word "medium" line reaches its natural end-of-speech token
# at ~27 s, not ~12 s). With the old 10/15/22 s caps, medium and long clips were
# cut off mid-sentence before the model emitted EOS. These caps let clips reach
# their natural end; clips that finish early stop at EOS regardless of the cap.
PROMPTS = {
    "short": {
        "target_s": 5,
        "max_ms": 20_000,
        "by_emotion": {
            "sad": [
                "I just feel so empty inside today, like nothing matters anymore.",
                "She left without saying goodbye, and now I'm completely alone again.",
                "The funeral was yesterday, and I still can't believe he's really gone.",
                "Everything reminds me of her, and it just hurts way too much.",
                "I tried my best, but it still wasn't enough this time around.",
            ],
            "angry": [
                "I can't believe you would lie straight to my face like that!",
                "Stop interrupting me every single time I try to speak, right now!",
                "This is absolutely ridiculous, and I am completely done with all of it!",
                "You promised you'd be here, and once again you let me down!",
                "Get out of my house right now, I never want to see you!",
            ],
            "normal": [
                "I went to the grocery store this morning to buy some fresh bread.",
                "The weather today is partly cloudy with a slight chance of light rain.",
                "My favorite color is blue, especially the deep shade of the ocean.",
                "I usually drink coffee with breakfast, and then tea in the afternoon.",
                "The meeting is scheduled for three o'clock in the main conference room.",
            ],
            "excited": [
                "Oh my god, I just got accepted into my absolute dream school!",
                "We're going to Disneyland next week, I really can hardly wait!",
                "I just won the lottery, can you actually believe this is happening?",
                "She said yes! She actually said yes, we are getting married!",
                "This is the best day of my entire life, I'm so happy!",
            ],
        },
    },
    "medium": {
        "target_s": 10,
        "max_ms": 40_000,
        "by_emotion": {
            "sad": [
                "I sat alone in the dark, thinking about everything that went wrong, every choice I made, every single person I hurt along the way. It's overwhelming.",
                "The doctor said there's nothing more they can do. We tried every treatment, every option, and now I'm just sitting here waiting to say goodbye.",
                "After twenty years together, she packed her bags and walked out the door. I never thought love could end so quietly, so completely without any warning.",
                "I lost my job last month, then my apartment, then my dog ran away. Everything I built is gone, and I don't know how to start over.",
                "Mom passed away on Tuesday morning, and the house feels so terribly quiet now. I keep expecting her to call, but the phone never rings anymore.",
            ],
            "angry": [
                "You took the money I was saving for years, lied straight to my face about it, and now you have the audacity to ask me for more. Unbelievable!",
                "I told you a hundred times not to touch my stuff, and what do you do? You break the one thing that meant something to me. I'm furious!",
                "Every single time we have these meetings, you completely ignore what I say and then take credit for my ideas later. This stops right now, today!",
                "He cut me off in traffic, flipped me off, and then had the nerve to honk at me. I swear, people on this road are absolutely insane these days!",
                "You promised me, you swore on everything you cared about, and you still went behind my back. I can never trust you again, do you understand me?",
            ],
            "normal": [
                "I went to the farmers market this morning and picked up some fresh tomatoes, basil, and a loaf of sourdough bread. I'm planning to make pasta for dinner.",
                "The new office building downtown is supposed to open next month. They're holding a ribbon cutting ceremony, and apparently the mayor will be attending the event in person.",
                "I started reading this novel last week about a detective in nineteenth century London. It's pretty interesting so far, though the writing style takes some getting used to.",
                "Our quarterly report is due by Friday afternoon, so please send me your numbers by Wednesday at the latest. We'll review everything together during the next team meeting.",
                "The kids have soccer practice on Tuesdays and Thursdays, piano lessons on Wednesdays, and games every Saturday morning. Our schedule is pretty packed, but they really love it.",
            ],
            "excited": [
                "I cannot believe we actually pulled it off! After months of working day and night, our startup just got funded, and we're moving to Silicon Valley next month!",
                "Guess what just happened! I checked my email this morning, and my book got accepted for publication! After ten years of rejections, someone finally said yes!",
                "The tickets just dropped for the concert, and I managed to grab front row seats! We're going to see them live, and it's going to be absolutely incredible!",
                "She just took her first steps! Right in the living room, she stood up and walked three steps before falling over. Oh my goodness, I'm crying tears of joy!",
                "We won the championship! After years of being the underdogs, we finally did it! The whole team is celebrating, and I have never felt this proud in my life!",
            ],
        },
    },
    "long": {
        "target_s": 15,
        "max_ms": 50_000,
        "by_emotion": {
            "sad": [
                "It's been three months since the accident, and I still wake up every morning expecting to hear his footsteps in the kitchen, the coffee maker running, his soft humming. The silence is what hurts the most, the absolute, devastating silence of an empty house.",
                "I keep going through her things, finding little notes she wrote, photographs from trips we took together, that scarf she always wore in winter. Each item is a tiny earthquake in my chest, and I don't know when, if ever, this pain will start to fade.",
                "The kids don't understand why mommy isn't coming home, and I don't have the heart to explain it properly. I just hold them at night and pretend I'm strong, but the moment they fall asleep, I break down completely in the bathroom alone.",
                "After thirty years at the company, they let me go in a five minute meeting. No warning, no thank you, just a cardboard box and a security guard walking me out the door. I gave them everything, and this is what I got in return.",
                "I went back to our favorite restaurant last week, sat down in our usual booth, and ordered both our meals out of habit. The waiter looked at me with so much pity in his eyes that I had to leave before the food even arrived at the table.",
            ],
            "angry": [
                "I have had it up to here with your constant excuses, your endless lies, and your complete lack of respect for anyone but yourself. Every single time I give you another chance, you find a new way to disappoint me, and I am absolutely done with it!",
                "You sit there at family dinners pretending to be the perfect son, while behind everyone's back you steal from grandma, manipulate our parents, and ruin every relationship I try to build. I'm done covering for you, do you hear me? Completely, absolutely done!",
                "The contractor took our deposit, didn't show up for three straight weeks, then had the nerve to demand more money before he'd even start the work. We are filing a lawsuit, contacting the better business bureau, and making sure nobody ever hires this scam artist again!",
                "I have told the homeowners association seventeen times to fix that broken streetlight, and every single time they brush me off with the same scripted nonsense. Someone is going to get hurt, and when they do, I am going to make sure everyone knows about it!",
                "You knew that meeting was important to me, you knew I had been preparing for weeks, and you still didn't show up. No call, no text, not a single explanation. This is the last time you ever ruin something important to me, I swear on everything I love!",
            ],
            "normal": [
                "The renovation project at the downtown library is finally complete after eighteen months of construction. The new wing includes a children's reading area, several meeting rooms, and a small cafe. The grand reopening is scheduled for next Saturday at noon, and everyone in the community is invited.",
                "Our hiking group meets every Sunday morning at seven sharp in the parking lot near the trailhead. We typically cover about ten miles, take a long break for lunch around noon, and finish up by mid afternoon. Bring water, sunscreen, and sturdy shoes if you want to join us.",
                "The conference next month will cover artificial intelligence, machine learning, and the future of automation across various industries. Registration is open through the company portal, and we have a limited number of seats available. Please let me know by Friday if you are interested in attending the event.",
                "For tonight's recipe, dice three onions, two carrots, and four cloves of garlic. Sauté everything in olive oil for about ten minutes, then add the tomato paste, broth, and herbs. Let it simmer for an hour, and serve over fresh pasta with grated parmesan cheese on top.",
                "The new highway extension will reduce commute times from the suburbs by approximately twenty minutes during peak hours. Construction begins in September and is expected to wrap up by late next summer. The project includes three new exits and two pedestrian bridges across the main thoroughfare downtown.",
            ],
            "excited": [
                "You will not believe what happened to me today! I was walking down the street, minding my own business, when this casting director stopped me and asked if I'd be interested in auditioning for a major motion picture! Can you imagine? Me, in a Hollywood movie!",
                "After fifteen years of trying, after countless rejections and almost giving up entirely, my novel just hit the New York Times bestseller list! Number seven, can you believe it? Number seven on the bestseller list! All those late nights and revisions were finally worth every single second!",
                "We just closed on the house! After months of negotiations, inspections, and mountains of paperwork, we officially own our dream home! It has the big backyard for the kids, the kitchen I always wanted, and we move in next month! I cannot stop smiling, this is incredible!",
                "The pregnancy test came back positive, and we just got back from the doctor's office where they officially confirmed it! We are having a baby! After three years of trying, three years of hoping and waiting, we are finally going to be parents! I cannot believe this is real!",
                "I got the promotion! Not only that, they are giving me a corner office, a company car, and a salary increase that is almost double what I am making now! All those years of late nights and weekends are finally paying off! This is the best news in years!",
            ],
        },
    },
}


STREAM_CHUNK_FRAMES = 25  # ~2 s of 24 kHz audio per chunk; matches README default


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _md_escape(text: str) -> str:
    # Markdown tables: pipe is the column separator, and newlines break the row.
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _stats(rows, fields=("wall_s", "audio_s", "rtf")):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for f in fields:
        vals = [r[f] for r in rows]
        out[f"avg_{f}"] = sum(vals) / n
        out[f"min_{f}"] = min(vals)
        out[f"max_{f}"] = max(vals)
        out[f"total_{f}"] = sum(vals)
    return out


def run_batch_clip(generator, text, length_label, emotion, idx, max_ms,
                   sample_rate, device, out_dir):
    synchronize(device)
    t0 = time.perf_counter()
    audio = generator.generate(
        text=text, speaker=0, context=[], max_audio_length_ms=max_ms,
    )
    synchronize(device)
    t1 = time.perf_counter()

    wall_s = t1 - t0
    audio_s = audio.shape[0] / sample_rate
    rtf = wall_s / audio_s if audio_s > 0 else float("inf")

    filename = f"{length_label}_{emotion}_{idx:02d}.wav"
    torchaudio.save(str(out_dir / filename), audio.unsqueeze(0).cpu(), sample_rate)

    return {
        "length": length_label, "emotion": emotion, "index": idx, "text": text,
        "wall_s": wall_s, "audio_s": audio_s, "rtf": rtf, "filename": filename,
    }


def run_stream_clip(generator, text, length_label, emotion, idx, max_ms,
                    sample_rate, device, out_dir):
    synchronize(device)
    t0 = time.perf_counter()
    ttfb_s = None
    chunks = []
    for chunk in generator.generate_stream(
        text=text, speaker=0, context=[], max_audio_length_ms=max_ms,
        chunk_frames=STREAM_CHUNK_FRAMES,
    ):
        if ttfb_s is None:
            synchronize(device)
            ttfb_s = time.perf_counter() - t0
        chunks.append(chunk.cpu())
    synchronize(device)
    t1 = time.perf_counter()

    wall_s = t1 - t0
    if chunks:
        audio = torch.cat(chunks, dim=0)
    else:
        audio = torch.zeros(0)
    audio_s = audio.shape[0] / sample_rate
    rtf = wall_s / audio_s if audio_s > 0 else float("inf")
    # If no chunk was yielded, ttfb is undefined; report total wall as a stand-in.
    if ttfb_s is None:
        ttfb_s = wall_s

    filename = f"{length_label}_{emotion}_{idx:02d}_stream.wav"
    torchaudio.save(str(out_dir / filename), audio.unsqueeze(0), sample_rate)

    return {
        "length": length_label, "emotion": emotion, "index": idx, "text": text,
        "ttfb_s": ttfb_s, "wall_s": wall_s, "audio_s": audio_s, "rtf": rtf,
        "filename": filename,
    }


def _render_clip_table(lines, rows, include_ttfb):
    if include_ttfb:
        lines.append("| # | Emotion | File | TTFB (s) | Wall (s) | Audio (s) | RTF | Text |")
        lines.append("|---|---------|------|---------:|---------:|----------:|----:|------|")
        for r in rows:
            lines.append(
                f"| {r['index']:02d} "
                f"| {r['emotion']} "
                f"| `{r['filename']}` "
                f"| {r['ttfb_s']:.2f} "
                f"| {r['wall_s']:.2f} "
                f"| {r['audio_s']:.2f} "
                f"| {r['rtf']:.2f} "
                f"| {_md_escape(r['text'])} |"
            )
    else:
        lines.append("| # | Emotion | File | Wall (s) | Audio (s) | RTF | Text |")
        lines.append("|---|---------|------|---------:|----------:|----:|------|")
        for r in rows:
            lines.append(
                f"| {r['index']:02d} "
                f"| {r['emotion']} "
                f"| `{r['filename']}` "
                f"| {r['wall_s']:.2f} "
                f"| {r['audio_s']:.2f} "
                f"| {r['rtf']:.2f} "
                f"| {_md_escape(r['text'])} |"
            )


def _render_section_summary(lines, rows, include_ttfb, title):
    fields = ("wall_s", "audio_s", "rtf") + (("ttfb_s",) if include_ttfb else ())
    s = _stats(rows, fields=fields)
    lines.append(f"**{title} — summary**")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Clips | {s['n']} |")
    if include_ttfb:
        lines.append(f"| Avg TTFB | {s['avg_ttfb_s']:.2f} s |")
        lines.append(f"| Min TTFB (fastest first chunk) | {s['min_ttfb_s']:.2f} s |")
        lines.append(f"| Max TTFB | {s['max_ttfb_s']:.2f} s |")
    lines.append(f"| Avg wall time | {s['avg_wall_s']:.2f} s |")
    lines.append(f"| Avg audio length | {s['avg_audio_s']:.2f} s |")
    lines.append(f"| Avg RTF | {s['avg_rtf']:.2f} |")
    lines.append(f"| Min RTF (fastest) | {s['min_rtf']:.2f} |")
    lines.append(f"| Max RTF (slowest) | {s['max_rtf']:.2f} |")
    lines.append(f"| Total wall time | {s['total_wall_s']:.2f} s |")
    lines.append(f"| Total audio generated | {s['total_audio_s']:.2f} s |")
    lines.append("")


def _render_totals_table(lines, results, include_ttfb):
    if include_ttfb:
        lines.append(
            "| Length | N | Avg TTFB (s) | Avg Wall (s) | Avg Audio (s) | "
            "Avg RTF | Min RTF | Max RTF | Total Wall (s) | Total Audio (s) |"
        )
        lines.append(
            "|--------|--:|-------------:|-------------:|--------------:|"
            "--------:|--------:|--------:|---------------:|----------------:|"
        )
    else:
        lines.append(
            "| Length | N | Avg Wall (s) | Avg Audio (s) | Avg RTF | "
            "Min RTF | Max RTF | Total Wall (s) | Total Audio (s) |"
        )
        lines.append(
            "|--------|--:|-------------:|--------------:|--------:|"
            "--------:|--------:|---------------:|----------------:|"
        )
    fields = ("wall_s", "audio_s", "rtf") + (("ttfb_s",) if include_ttfb else ())
    for length_label in ("short", "medium", "long"):
        rows = [r for r in results if r["length"] == length_label]
        if not rows:
            continue
        s = _stats(rows, fields=fields)
        if include_ttfb:
            lines.append(
                f"| {length_label} | {s['n']} | {s['avg_ttfb_s']:.2f} "
                f"| {s['avg_wall_s']:.2f} | {s['avg_audio_s']:.2f} | {s['avg_rtf']:.2f} "
                f"| {s['min_rtf']:.2f} | {s['max_rtf']:.2f} "
                f"| {s['total_wall_s']:.2f} | {s['total_audio_s']:.2f} |"
            )
        else:
            lines.append(
                f"| {length_label} | {s['n']} "
                f"| {s['avg_wall_s']:.2f} | {s['avg_audio_s']:.2f} | {s['avg_rtf']:.2f} "
                f"| {s['min_rtf']:.2f} | {s['max_rtf']:.2f} "
                f"| {s['total_wall_s']:.2f} | {s['total_audio_s']:.2f} |"
            )
    overall = _stats(results, fields=fields)
    if include_ttfb:
        lines.append(
            f"| **overall** | **{overall['n']}** | **{overall['avg_ttfb_s']:.2f}** "
            f"| **{overall['avg_wall_s']:.2f}** | **{overall['avg_audio_s']:.2f}** "
            f"| **{overall['avg_rtf']:.2f}** "
            f"| **{overall['min_rtf']:.2f}** | **{overall['max_rtf']:.2f}** "
            f"| **{overall['total_wall_s']:.2f}** | **{overall['total_audio_s']:.2f}** |"
        )
    else:
        lines.append(
            f"| **overall** | **{overall['n']}** "
            f"| **{overall['avg_wall_s']:.2f}** | **{overall['avg_audio_s']:.2f}** "
            f"| **{overall['avg_rtf']:.2f}** "
            f"| **{overall['min_rtf']:.2f}** | **{overall['max_rtf']:.2f}** "
            f"| **{overall['total_wall_s']:.2f}** | **{overall['total_audio_s']:.2f}** |"
        )


def write_markdown_report(
    md_path: Path,
    batch_results: list,
    stream_results: list,
    env: dict,
) -> None:
    lines: list[str] = []
    lines.append("# Miso TTS 8B — Measured Performance")
    lines.append("")
    lines.append(f"- **Generated:** {env['timestamp']}")
    lines.append(f"- **Device:** {env['device']}" + (f" ({env['device_name']})" if env.get("device_name") else ""))
    lines.append(f"- **Torch:** {env['torch_version']} · **dtype:** {env['dtype']}")
    lines.append(f"- **Sample rate:** {env['sample_rate']} Hz")
    lines.append(f"- **Model:** {env['model_source']}")
    lines.append(f"- **Stream chunk_frames:** {STREAM_CHUNK_FRAMES} (~{STREAM_CHUNK_FRAMES * 0.08:.1f} s/chunk)")
    lines.append("")
    lines.append(
        "RTF = wall-clock seconds ÷ generated audio seconds. "
        "RTF < 1.0 means faster than realtime; lower is better. "
        "TTFB = wall-clock seconds from `generate_stream()` call until the first audio chunk is ready."
    )
    lines.append("")

    section_titles = {
        "short": "Short clips (target ~5 s)",
        "medium": "Medium clips (target ~10 s)",
        "long": "Long clips (target ~15 s)",
    }

    # ----- Batch section -----
    lines.append("# Batch (non-streaming) — `generator.generate()`")
    lines.append("")
    for length_label in ("short", "medium", "long"):
        rows = [r for r in batch_results if r["length"] == length_label]
        if not rows:
            continue
        lines.append(f"## {section_titles[length_label]}")
        lines.append("")
        _render_clip_table(lines, rows, include_ttfb=False)
        lines.append("")
        _render_section_summary(lines, rows, include_ttfb=False,
                                title=section_titles[length_label])

    lines.append("## Batch totals")
    lines.append("")
    _render_totals_table(lines, batch_results, include_ttfb=False)
    lines.append("")

    # ----- Streaming section -----
    if stream_results:
        lines.append(f"# Streaming — `generator.generate_stream(chunk_frames={STREAM_CHUNK_FRAMES})`")
        lines.append("")
        for length_label in ("short", "medium", "long"):
            rows = [r for r in stream_results if r["length"] == length_label]
            if not rows:
                continue
            lines.append(f"## {section_titles[length_label]} (streaming)")
            lines.append("")
            _render_clip_table(lines, rows, include_ttfb=True)
            lines.append("")
            _render_section_summary(lines, rows, include_ttfb=True,
                                    title=section_titles[length_label] + " (streaming)")

        lines.append("## Streaming totals")
        lines.append("")
        _render_totals_table(lines, stream_results, include_ttfb=True)
        lines.append("")

        # ----- Batch vs streaming comparison -----
        lines.append("# Batch vs streaming comparison")
        lines.append("")
        lines.append(
            "| Length | N | Batch Avg Wall (s) | Stream Avg Wall (s) | "
            "Stream Avg TTFB (s) | Batch Avg RTF | Stream Avg RTF |"
        )
        lines.append(
            "|--------|--:|-------------------:|--------------------:|"
            "--------------------:|--------------:|---------------:|"
        )

        def _fmt(stats, key, fmt=".2f"):
            return f"{stats[key]:{fmt}}" if stats else "–"

        for length_label in ("short", "medium", "long"):
            b = [r for r in batch_results if r["length"] == length_label]
            s = [r for r in stream_results if r["length"] == length_label]
            n_show = max(len(b), len(s))
            if n_show == 0:
                continue
            bs = _stats(b) if b else None
            ss = _stats(s, fields=("wall_s", "audio_s", "rtf", "ttfb_s")) if s else None
            lines.append(
                f"| {length_label} | {n_show} "
                f"| {_fmt(bs, 'avg_wall_s')} "
                f"| {_fmt(ss, 'avg_wall_s')} "
                f"| {_fmt(ss, 'avg_ttfb_s')} "
                f"| {_fmt(bs, 'avg_rtf')} "
                f"| {_fmt(ss, 'avg_rtf')} |"
            )

        bs_all = _stats(batch_results) if batch_results else None
        ss_all = _stats(stream_results, fields=("wall_s", "audio_s", "rtf", "ttfb_s")) if stream_results else None
        n_all = max(len(batch_results), len(stream_results))
        lines.append(
            f"| **overall** | **{n_all}** "
            f"| **{_fmt(bs_all, 'avg_wall_s')}** "
            f"| **{_fmt(ss_all, 'avg_wall_s')}** "
            f"| **{_fmt(ss_all, 'avg_ttfb_s')}** "
            f"| **{_fmt(bs_all, 'avg_rtf')}** "
            f"| **{_fmt(ss_all, 'avg_rtf')}** |"
        )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log(f"Using device: {device}")

    device_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    if device_name:
        log(f"GPU: {device_name}")
    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    log(f"Loading model from: {model_source}")
    log("  (this maps the ~32 GB checkpoint into memory — expect a long pause here)")
    t_load = time.perf_counter()
    generator = load_miso_8b(device, model_path_or_repo_id=model_source)
    log(f"Model loaded in {time.perf_counter() - t_load:.1f}s.")
    sample_rate = generator.sample_rate
    model_dtype = next(generator._model.parameters()).dtype

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Warm-up: batch path (kernel compile / cudnn tuning), then streaming path
    # (Mimi streaming captures a CUDA graph at the first full chunk's shape).
    log("Warming up batch path...")
    _ = generator.generate(
        text="Warming up the model.", speaker=0, context=[],
        max_audio_length_ms=4_000,
    )
    synchronize(device)

    log(f"Warming up streaming path (chunk_frames={STREAM_CHUNK_FRAMES})...")
    for _chunk in generator.generate_stream(
        text="Warming up the streaming path with a slightly longer prompt.",
        speaker=0, context=[], max_audio_length_ms=6_000,
        chunk_frames=STREAM_CHUNK_FRAMES,
    ):
        pass
    synchronize(device)
    log("Warm-up complete. Starting timed runs (60 batch + 60 streaming clips)...")

    batch_results = []
    stream_results = []

    for length_label, cfg in PROMPTS.items():
        max_ms = cfg["max_ms"]
        target_s = cfg["target_s"]
        print(
            f"\n=== {length_label.upper()} "
            f"(target ~{target_s}s, max_audio_length_ms={max_ms}) ==="
        )
        for emotion, sentences in cfg["by_emotion"].items():
            for idx, text in enumerate(sentences, start=1):
                br = run_batch_clip(
                    generator, text, length_label, emotion, idx, max_ms,
                    sample_rate, device, out_dir,
                )
                batch_results.append(br)
                print(
                    f"  [{length_label}/{emotion}/{idx:02d}] batch  "
                    f"wall={br['wall_s']:6.2f}s  audio={br['audio_s']:5.2f}s  "
                    f"RTF={br['rtf']:5.2f}  -> {br['filename']}"
                )

                sr = run_stream_clip(
                    generator, text, length_label, emotion, idx, max_ms,
                    sample_rate, device, out_dir,
                )
                stream_results.append(sr)
                print(
                    f"  [{length_label}/{emotion}/{idx:02d}] stream "
                    f"ttfb={sr['ttfb_s']:5.2f}s  wall={sr['wall_s']:6.2f}s  "
                    f"audio={sr['audio_s']:5.2f}s  RTF={sr['rtf']:5.2f}  "
                    f"-> {sr['filename']}"
                )

    # Console summary tables (batch then streaming).
    def _print_avg_table(rows, label, include_ttfb):
        print("\n" + "=" * 78)
        print(f"Average performance per length category — {label}")
        print("=" * 78)
        header = (
            f"{'Length':10}  {'N':>3}  "
            + (f"{'Avg TTFB':>9}  " if include_ttfb else "")
            + f"{'Avg Wall':>9}  {'Avg Audio':>10}  {'Avg RTF':>9}"
        )
        print(header)
        print("-" * 78)
        for length_label in PROMPTS.keys():
            ls = [r for r in rows if r["length"] == length_label]
            if not ls:
                continue
            n = len(ls)
            avg_wall = sum(r["wall_s"] for r in ls) / n
            avg_audio = sum(r["audio_s"] for r in ls) / n
            avg_rtf = sum(r["rtf"] for r in ls) / n
            line = f"{length_label:10}  {n:3d}  "
            if include_ttfb:
                avg_ttfb = sum(r["ttfb_s"] for r in ls) / n
                line += f"{avg_ttfb:9.2f}  "
            line += f"{avg_wall:9.2f}  {avg_audio:10.2f}  {avg_rtf:9.2f}"
            print(line)

    _print_avg_table(batch_results, "batch", include_ttfb=False)
    _print_avg_table(stream_results, "streaming", include_ttfb=True)

    # CSV dumps (one per mode).
    batch_csv = out_dir / "profile_results_batch.csv"
    with open(batch_csv, "w", encoding="utf-8") as f:
        f.write("length,emotion,index,wall_s,audio_s,rtf,filename,text\n")
        for r in batch_results:
            safe_text = r["text"].replace('"', '""')
            f.write(
                f"{r['length']},{r['emotion']},{r['index']},"
                f"{r['wall_s']:.4f},{r['audio_s']:.4f},{r['rtf']:.4f},"
                f"{r['filename']},\"{safe_text}\"\n"
            )
    print(f"\nWrote batch CSV to {batch_csv}")

    stream_csv = out_dir / "profile_results_stream.csv"
    with open(stream_csv, "w", encoding="utf-8") as f:
        f.write("length,emotion,index,ttfb_s,wall_s,audio_s,rtf,filename,text\n")
        for r in stream_results:
            safe_text = r["text"].replace('"', '""')
            f.write(
                f"{r['length']},{r['emotion']},{r['index']},"
                f"{r['ttfb_s']:.4f},{r['wall_s']:.4f},{r['audio_s']:.4f},"
                f"{r['rtf']:.4f},{r['filename']},\"{safe_text}\"\n"
            )
    print(f"Wrote streaming CSV to {stream_csv}")

    # Markdown report alongside the script for easy reading.
    md_path = Path("TTS_MEASURED_PERFORMANCE.md")
    write_markdown_report(
        md_path,
        batch_results,
        stream_results,
        env={
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "dtype": str(model_dtype).replace("torch.", ""),
            "sample_rate": sample_rate,
            "model_source": model_source,
        },
    )
    print(f"Wrote performance report to {md_path}")


if __name__ == "__main__":
    main()
