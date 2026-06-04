import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

import time
from datetime import datetime
from pathlib import Path

import torch
import torchaudio  # type: ignore

from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b


# Each entry: target audio length in seconds + a generous cap (max_audio_length_ms)
# + 5 sentences per emotion (sad / angry / normal / excited) sized to roughly hit
# the target when spoken at a natural pace (~2.5 words/sec).
PROMPTS = {
    "short": {
        "target_s": 5,
        "max_ms": 10_000,
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
        "max_ms": 15_000,
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
        "max_ms": 22_000,
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


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _md_escape(text: str) -> str:
    # Markdown tables: pipe is the column separator, and newlines break the row.
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _stats(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    walls = [r["wall_s"] for r in rows]
    audios = [r["audio_s"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    return {
        "n": n,
        "avg_wall": sum(walls) / n,
        "avg_audio": sum(audios) / n,
        "avg_rtf": sum(rtfs) / n,
        "min_rtf": min(rtfs),
        "max_rtf": max(rtfs),
        "total_wall": sum(walls),
        "total_audio": sum(audios),
    }


def write_markdown_report(
    md_path: Path,
    results: list,
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
    lines.append("")
    lines.append(
        "RTF = wall-clock seconds ÷ generated audio seconds. "
        "RTF < 1.0 means faster than realtime; lower is better."
    )
    lines.append("")

    section_titles = {
        "short": "Short clips (target ~5 s)",
        "medium": "Medium clips (target ~10 s)",
        "long": "Long clips (target ~15 s)",
    }

    for length_label in ("short", "medium", "long"):
        rows = [r for r in results if r["length"] == length_label]
        if not rows:
            continue
        lines.append(f"## {section_titles[length_label]}")
        lines.append("")
        lines.append(
            "| # | Emotion | File | Wall (s) | Audio (s) | RTF | Text |"
        )
        lines.append(
            "|---|---------|------|---------:|----------:|----:|------|"
        )
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
        lines.append("")

        s = _stats(rows)
        lines.append(f"**{section_titles[length_label]} — summary**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Clips | {s['n']} |")
        lines.append(f"| Avg wall time | {s['avg_wall']:.2f} s |")
        lines.append(f"| Avg audio length | {s['avg_audio']:.2f} s |")
        lines.append(f"| Avg RTF | {s['avg_rtf']:.2f} |")
        lines.append(f"| Min RTF (fastest) | {s['min_rtf']:.2f} |")
        lines.append(f"| Max RTF (slowest) | {s['max_rtf']:.2f} |")
        lines.append(f"| Total wall time | {s['total_wall']:.2f} s |")
        lines.append(f"| Total audio generated | {s['total_audio']:.2f} s |")
        lines.append("")

    # Totals: side-by-side comparison + overall.
    lines.append("## Totals")
    lines.append("")
    lines.append(
        "| Length | N | Avg Wall (s) | Avg Audio (s) | Avg RTF | "
        "Min RTF | Max RTF | Total Wall (s) | Total Audio (s) |"
    )
    lines.append(
        "|--------|--:|-------------:|--------------:|--------:|"
        "--------:|--------:|---------------:|----------------:|"
    )
    for length_label in ("short", "medium", "long"):
        rows = [r for r in results if r["length"] == length_label]
        if not rows:
            continue
        s = _stats(rows)
        lines.append(
            f"| {length_label} | {s['n']} "
            f"| {s['avg_wall']:.2f} | {s['avg_audio']:.2f} | {s['avg_rtf']:.2f} "
            f"| {s['min_rtf']:.2f} | {s['max_rtf']:.2f} "
            f"| {s['total_wall']:.2f} | {s['total_audio']:.2f} |"
        )
    overall = _stats(results)
    lines.append(
        f"| **overall** | **{overall['n']}** "
        f"| **{overall['avg_wall']:.2f}** | **{overall['avg_audio']:.2f}** "
        f"| **{overall['avg_rtf']:.2f}** "
        f"| **{overall['min_rtf']:.2f}** | **{overall['max_rtf']:.2f}** "
        f"| **{overall['total_wall']:.2f}** | **{overall['total_audio']:.2f}** |"
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    device_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    print(f"Loading model from: {model_source}")
    generator = load_miso_8b(device, model_path_or_repo_id=model_source)
    sample_rate = generator.sample_rate
    model_dtype = next(generator._model.parameters()).dtype

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Warm-up run so first-iteration kernel compilation / cudnn tuning does not
    # skew the first measured clip.
    print("Warming up...")
    _ = generator.generate(
        text="Warming up the model.",
        speaker=0,
        context=[],
        max_audio_length_ms=4_000,
    )
    synchronize(device)

    results = []  # one dict per clip

    for length_label, cfg in PROMPTS.items():
        max_ms = cfg["max_ms"]
        target_s = cfg["target_s"]
        print(
            f"\n=== {length_label.upper()} "
            f"(target ~{target_s}s, max_audio_length_ms={max_ms}) ==="
        )
        for emotion, sentences in cfg["by_emotion"].items():
            for idx, text in enumerate(sentences, start=1):
                synchronize(device)
                t0 = time.perf_counter()
                audio = generator.generate(
                    text=text,
                    speaker=0,
                    context=[],
                    max_audio_length_ms=max_ms,
                )
                synchronize(device)
                t1 = time.perf_counter()

                wall_s = t1 - t0
                audio_s = audio.shape[0] / sample_rate
                rtf = wall_s / audio_s if audio_s > 0 else float("inf")

                filename = f"{length_label}_{emotion}_{idx:02d}.wav"
                fpath = out_dir / filename
                torchaudio.save(
                    str(fpath),
                    audio.unsqueeze(0).cpu(),
                    sample_rate,
                )

                print(
                    f"  [{length_label}/{emotion}/{idx:02d}] "
                    f"wall={wall_s:6.2f}s  audio={audio_s:5.2f}s  "
                    f"RTF={rtf:5.2f}  -> {filename}"
                )
                results.append(
                    {
                        "length": length_label,
                        "emotion": emotion,
                        "index": idx,
                        "text": text,
                        "wall_s": wall_s,
                        "audio_s": audio_s,
                        "rtf": rtf,
                        "filename": filename,
                    }
                )

    # Per-clip detail table.
    print("\n" + "=" * 78)
    print("Per-clip results")
    print("=" * 78)
    print(f"{'File':34}  {'Wall (s)':>9}  {'Audio (s)':>10}  {'RTF':>6}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['filename']:34}  {r['wall_s']:9.2f}  "
            f"{r['audio_s']:10.2f}  {r['rtf']:6.2f}"
        )

    # Aggregate table by length category.
    print("\n" + "=" * 78)
    print("Average performance per length category")
    print("=" * 78)
    print(
        f"{'Length':10}  {'N':>3}  {'Avg Wall (s)':>14}  "
        f"{'Avg Audio (s)':>15}  {'Avg RTF':>9}"
    )
    print("-" * 78)
    for length_label in PROMPTS.keys():
        rows = [r for r in results if r["length"] == length_label]
        n = len(rows)
        avg_wall = sum(r["wall_s"] for r in rows) / n
        avg_audio = sum(r["audio_s"] for r in rows) / n
        avg_rtf = sum(r["rtf"] for r in rows) / n
        print(
            f"{length_label:10}  {n:3d}  {avg_wall:14.2f}  "
            f"{avg_audio:15.2f}  {avg_rtf:9.2f}"
        )

    # CSV dump for later analysis.
    csv_path = out_dir / "profile_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("length,emotion,index,wall_s,audio_s,rtf,filename,text\n")
        for r in results:
            safe_text = r["text"].replace('"', '""')
            f.write(
                f"{r['length']},{r['emotion']},{r['index']},"
                f"{r['wall_s']:.4f},{r['audio_s']:.4f},{r['rtf']:.4f},"
                f"{r['filename']},\"{safe_text}\"\n"
            )
    print(f"\nWrote per-clip CSV to {csv_path}")

    # Markdown report alongside the script for easy reading.
    md_path = Path("TTS_MEASURED_PERFORMANCE.md")
    write_markdown_report(
        md_path,
        results,
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
