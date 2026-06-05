import contextlib
from dataclasses import dataclass
import os
import queue
import threading
from typing import Iterator, List, Optional, Tuple

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import MISO_TTS_8B_CONFIG, Model, ModelArgs
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
patch_bitsandbytes_import_for_unquantized_layers()


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def _stack_audio_frames(samples: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(samples).permute(1, 2, 0)


def _match_num_samples(audio: torch.Tensor, num_samples: int) -> torch.Tensor:
    if audio.size(0) > num_samples:
        return audio[:num_samples]
    if audio.size(0) < num_samples:
        padding = torch.zeros(num_samples - audio.size(0), dtype=audio.dtype, device=audio.device)
        return torch.cat([audio, padding], dim=0)
    return audio


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        self._frame_size = self._model.config.audio_num_codebooks + 1

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(self._model.config.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}] {text.lstrip()}")
        text_frame = torch.zeros(len(text_tokens), self._frame_size).long()
        text_frame_mask = torch.zeros(len(text_tokens), self._frame_size).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._frame_size).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._frame_size).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, audio_num_codebooks + 1), (seq_len, audio_num_codebooks + 1)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def _prepare_prompt(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if prompt_tokens.size(0) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        return prompt_tokens, prompt_tokens_mask, max_generation_len

    def _generate_frames(
        self,
        prompt_tokens: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        max_generation_len: int,
        temperature: float,
        topk: int,
    ) -> Iterator[torch.Tensor]:
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        # The EOS frame is all zeros, and once the model emits it EVERY subsequent
        # frame is also all zeros (verified: clean, stable padding, not drift). The
        # original code tested `torch.all(sample == 0)` in a Python `if` on EVERY
        # frame, which forces a device->host sync per frame (thousands per utterance)
        # that serializes the GPU pipeline - the same stall class the decoder-cache
        # rewind (models.py) and the no-sync sampling were designed to avoid. Instead
        # accumulate the EOS condition on-device and sync it only once every
        # `eos_check_every` frames. The cost is overshooting EOS by up to
        # eos_check_every-1 all-zero (silence) frames; generate() trims all trailing
        # zero frames, and the streamed silence is bounded and inaudible.
        n = max(1, int(os.environ.get("MISO_EOS_CHECK_EVERY", "5")))
        zero_acc = torch.zeros((), device=self.device, dtype=torch.int32)
        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            yield sample
            zero_acc = zero_acc + (sample == 0).all().to(torch.int32)
            if (i + 1) % n == 0 and int(zero_acc) > 0:
                break  # EOS occurred within the last n frames (overshoot <= n-1 silent frames)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    def _decode_frames(self, samples: List[torch.Tensor]) -> torch.Tensor:
        return self._audio_tokenizer.decode(_stack_audio_frames(samples)).squeeze(0).squeeze(0)

    def _tail_is_chopped(self, audio: torch.Tensor, thresh: float = 0.35) -> bool:
        """True if the clip ends near speech energy (abrupt cutoff) rather than
        trailing to silence. Compares the RMS of the final 120 ms against a
        high percentile of frame RMS (the speech level)."""
        sr = self.sample_rate
        n = audio.shape[0]
        if n < int(0.25 * sr):
            return False
        x = audio.detach().float()
        fw = max(1, int(0.02 * sr))
        nf = n // fw
        rms = (x[: nf * fw].reshape(nf, fw).pow(2).mean(dim=1) + 1e-12).sqrt()
        speech = torch.quantile(rms, 0.9)
        tail = (x[-int(0.12 * sr):].pow(2).mean() + 1e-12).sqrt()
        return bool(speech > 0 and (tail / speech) > thresh)

    def _watermark_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # This applies an imperceptible watermark to identify audio as AI-generated.
        # If using Miso TTS in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, MISO_TTS_WATERMARK)
        # watermark() returns audio at min(44100, self.sample_rate) Hz, which
        # equals self.sample_rate (24k) -- so this was a no-op polyphase resample
        # over the whole clip on every call. Guard it; only resample if needed.
        if wm_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        return audio

    def _watermark_stream_chunk(self, audio: torch.Tensor, *, is_final: bool) -> torch.Tensor:
        target_num_samples = audio.size(0)
        try:
            audio = self._watermark_audio(audio)
        except Exception:
            # SilentCipher rejects very short chunks, and with the ramped emit
            # schedule the first chunks are as small as one 80 ms frame. Emit the
            # chunk unwatermarked rather than failing the stream: the bulk of the
            # clip is still watermarked per chunk, and batch generate() always
            # watermarks the whole clip. Task 6 turns this into an explicit,
            # configurable leading-defer policy.
            self._stream_wm_skipped = getattr(self, "_stream_wm_skipped", 0) + 1
            return audio
        return _match_num_samples(audio, target_num_samples)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        prompt_tokens, prompt_tokens_mask, max_generation_len = self._prepare_prompt(
            text, speaker, context, max_audio_length_ms
        )
        samples = list(
            self._generate_frames(prompt_tokens, prompt_tokens_mask, max_generation_len, temperature, topk)
        )

        # _generate_frames yields the all-zero EOS frame plus up to eos_check_every-1
        # all-zero overshoot frames (periodic EOS check). Trim ALL trailing all-zero
        # frames - they are silence. On clips that already trail to silence, decoding
        # them adds a faint blip, so decode WITHOUT them first and only re-include ONE
        # trailing frame when the clip would otherwise end chopped (so the codec
        # renders the final decay).
        k = len(samples)
        while k > 0 and bool((samples[k - 1] == 0).all()):
            k -= 1
        has_eos = k < len(samples)
        core = samples[:k] if has_eos else samples
        audio = self._decode_frames(core)
        if has_eos and self._tail_is_chopped(audio):
            audio = self._decode_frames(samples[:k + 1])
        audio = self._watermark_audio(audio)

        return audio

    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        chunk_frames: int = 25,
        start_frames: int = 1,
        ramp: float = 2.0,
        max_frames: Optional[int] = None,
        rtf_adapt: bool = False,
        wm_defer_ms: float = 0.0,
    ) -> Iterator[torch.Tensor]:
        """Stream audio with a low-latency ramp.

        The Mimi streaming decoder is wrapped in a CUDA graph captured at the FIRST
        decoded chunk's shape, so it cannot accept variable chunk sizes. We decouple
        DECODE granularity from EMIT cadence: decode exactly one frame at a time
        (the graph is captured once at shape 1 and replayed for every frame, cheap
        and numerically equivalent to batch decode), accumulate the decoded audio in
        a host buffer, and emit to the caller on a growing ramp. The first emit is
        one 80 ms frame (minimal TTFB); the emit size then grows (x`ramp`) up to a
        cap, amortizing per-chunk encode/watermark overhead once playback has a head
        start. `chunk_frames` is kept for back-compat as the cap (override with
        `max_frames`).
        """
        import time

        cap = int(max_frames if max_frames is not None else chunk_frames)
        if cap <= 0:
            raise ValueError("max chunk frames must be greater than 0")
        emit_target = max(1, min(int(start_frames), cap))
        growth = max(1.0, float(ramp))
        wm_defer_samples = int(self.sample_rate * wm_defer_ms / 1000.0)

        # Tokenize the prompt (including Mimi encode of any context audio) before
        # entering Mimi streaming mode, so streamed generation conditions on the
        # same context tokens as generate().
        with torch.inference_mode():
            prompt_tokens, prompt_tokens_mask, max_generation_len = self._prepare_prompt(
                text, speaker, context, max_audio_length_ms
            )

        frames = self._generate_frames(prompt_tokens, prompt_tokens_mask, max_generation_len, temperature, topk)

        all_samples: List[torch.Tensor] = []
        buffer: List[torch.Tensor] = []   # decoded per-frame audio, not yet emitted
        pending = 0                        # frames currently in buffer
        streamed_num_samples = 0
        emitted_samples = 0
        frame_wall = 0.0
        frame_count = 0

        def _emit(chunk_audio, *, is_final):
            nonlocal streamed_num_samples, emitted_samples
            streamed_num_samples += chunk_audio.size(0)
            # Leading-defer: skip watermarking the first wm_defer_ms of audio (Task 6).
            if emitted_samples < wm_defer_samples:
                out = chunk_audio
            else:
                out = self._watermark_stream_chunk(chunk_audio, is_final=is_final)
            emitted_samples += chunk_audio.size(0)
            return out

        # Pipeline path: a worker thread owns a side CUDA stream and does the
        # per-frame Mimi decode + ramp emit + watermark + cpu-copy, so the main
        # thread keeps the GPU GENERATING continuously instead of idling through
        # decode/watermark/yield between bursts (recovers toward the batch RTF).
        # Off by default until validated; MISO_STREAM_PIPELINE=1 enables it.
        if os.environ.get("MISO_STREAM_PIPELINE", "0") == "1" and str(self.device).startswith("cuda"):
            yield from self._generate_stream_pipelined(
                frames, all_samples, emit_target, cap, growth, rtf_adapt, wm_defer_samples)
            return

        # Each chunk is computed inside torch.inference_mode() and yielded outside it.
        # A `with torch.inference_mode():` spanning a `yield` would stay active while
        # the generator is suspended, silently putting the caller's loop in inference
        # mode.
        with self._audio_tokenizer.streaming(1):
            finished = False
            while not finished:
                out: Optional[torch.Tensor] = None
                with torch.inference_mode():
                    t = time.perf_counter()
                    sample = next(frames, None)
                    if sample is None:
                        finished = True
                    else:
                        all_samples.append(sample)
                        audio = self._decode_frames([sample])  # one frame, shape-1 graph
                        frame_wall += time.perf_counter() - t
                        frame_count += 1
                        if audio.numel() > 0:
                            buffer.append(audio)
                            pending += 1
                        if pending >= emit_target:
                            out = _emit(torch.cat(buffer, dim=0), is_final=False)
                            buffer, pending = [], 0
                            nxt = max(emit_target + 1, int(emit_target * growth))
                            # RTF-aware: if generation cannot sustain realtime, tiny
                            # chunks only add overhead -> jump to the cap.
                            if rtf_adapt and frame_count and \
                               (frame_wall / frame_count) / 0.08 > 1.0:
                                nxt = cap
                            emit_target = min(cap, nxt)
                if out is not None and out.numel() > 0:
                    yield out

            # Compute the EOS-trim boundary ONCE (one end-of-stream sync): keep the
            # speech frames plus a single EOS-decay frame, and drop the periodic-
            # check overshoot (post-EOS all-zero frames = silence) so the streamed
            # length matches generate() instead of trailing up to N-1 silent frames.
            keep_frames = len(all_samples)
            if all_samples:
                with torch.inference_mode():
                    last_nonzero = len(all_samples) - 1
                    while last_nonzero >= 0 and bool((all_samples[last_nonzero] == 0).all()):
                        last_nonzero -= 1
                    keep_frames = min(len(all_samples), last_nonzero + 2)

            # Flush the final partial ramp group, trimming any trailing overshoot
            # frames still sitting in the buffer.
            if buffer:
                emitted_frames = len(all_samples) - len(buffer)
                buffer = buffer[:max(0, keep_frames - emitted_frames)]
                if buffer:
                    with torch.inference_mode():
                        out = _emit(torch.cat(buffer, dim=0), is_final=True)
                    if out is not None and out.numel() > 0:
                        yield out

        # Tail reconciliation against the SAME EOS-trimmed frame set, so it neither
        # re-adds the trimmed overshoot nor chops the kept decay. Normally a no-op
        # (per-frame streaming decode matches the batch decode length).
        tail_audio: Optional[torch.Tensor] = None
        with torch.inference_mode():
            if keep_frames > 0:
                full_audio = self._decode_frames(all_samples[:keep_frames])
                if streamed_num_samples < full_audio.size(0):
                    tail = full_audio[streamed_num_samples:]
                    if tail.numel() > 0:
                        tail_audio = self._watermark_stream_chunk(tail, is_final=True)

        if tail_audio is not None:
            yield tail_audio

    def _generate_stream_pipelined(self, frames, all_samples, emit_target, cap, growth,
                                   rtf_adapt, wm_defer_samples):
        """Overlap the per-frame decode + watermark + cpu-copy with generation.

        A single worker thread owns a dedicated CUDA stream and the Mimi streaming
        context: it pulls generated frames over in_q, decodes each on its stream,
        ramp-buffers, watermarks + copies finished chunks to the CPU, and pushes
        them over out_q. The MAIN thread runs only the autoregressive generation
        loop (keeping the GPU busy back to back) plus the one end-of-stream EOS
        sync, and yields the worker's finished CPU chunks in order. Ordering is FIFO
        via the single worker. inference_mode is thread-local, so the worker enters
        its own and the driver holds none at any yield. The decode stream is
        event-synced to the generation stream so a frame is never decoded before
        generation finishes writing it.
        """
        import time as _time

        backlog = max(2, int(os.environ.get("MISO_STREAM_PIPELINE_BACKLOG", "8")))
        in_q: "queue.Queue" = queue.Queue(maxsize=backlog)
        out_q: "queue.Queue" = queue.Queue(maxsize=backlog)
        decode_stream = torch.cuda.Stream(device=self.device)
        state = {"streamed": 0, "error": None}
        _ERR, _END = object(), object()

        def worker():
            emit_t = emit_target
            buf, pending, emitted_frames, emitted_samples = [], 0, 0, 0
            frame_wall, frame_count = 0.0, 0

            def finish(chunk, *, is_final):
                nonlocal emitted_samples
                state["streamed"] += chunk.size(0)
                out = (chunk if emitted_samples < wm_defer_samples
                       else self._watermark_stream_chunk(chunk, is_final=is_final))
                emitted_samples += chunk.size(0)
                decode_stream.synchronize()  # block the WORKER (not generation) until ready
                return out.detach().to("cpu")

            try:
                # All worker GPU work runs on decode_stream, so the Mimi streaming
                # CUDA graph is captured AND replayed on this one stream (never the
                # default generation stream), and streaming(1) state is touched only
                # here.
                with torch.inference_mode(), torch.cuda.stream(decode_stream), \
                        self._audio_tokenizer.streaming(1):
                    while True:
                        item = in_q.get()
                        if item[0] is _END:
                            buf = buf[:max(0, item[1] - emitted_frames)]  # trim overshoot
                            if buf:
                                out_q.put(finish(torch.cat(buf, dim=0), is_final=True))
                            out_q.put(_END)
                            return
                        sample, ev = item
                        decode_stream.wait_event(ev)
                        t = _time.perf_counter()
                        audio = self._decode_frames([sample])  # one frame, shape-1 graph
                        frame_wall += _time.perf_counter() - t
                        frame_count += 1
                        if audio.numel() > 0:
                            buf.append(audio)
                            pending += 1
                        if pending >= emit_t:
                            out_q.put(finish(torch.cat(buf, dim=0), is_final=False))
                            emitted_frames += pending
                            buf, pending = [], 0
                            nxt = max(emit_t + 1, int(emit_t * growth))
                            if rtf_adapt and frame_count and (frame_wall / frame_count) / 0.08 > 1.0:
                                nxt = cap
                            emit_t = min(cap, nxt)
            except BaseException as exc:  # surface worker faults to the driver
                state["error"] = exc
                out_q.put(_ERR)

        w = threading.Thread(target=worker, name="miso-stream-decode", daemon=True)
        w.start()

        keep_frames = 0
        try:
            while True:
                with torch.inference_mode():
                    sample = next(frames, None)
                if sample is None:
                    break
                all_samples.append(sample)
                ev = torch.cuda.Event()
                ev.record()  # on the current (default/generation) stream
                # Drain finished chunks before possibly blocking on a full in_q, so a
                # full out_q + full in_q cannot deadlock.
                while True:
                    try:
                        o = out_q.get_nowait()
                    except queue.Empty:
                        break
                    if o is _ERR:
                        raise state["error"] or RuntimeError("stream worker failed")
                    if o is not _END and o.numel() > 0:
                        yield o
                in_q.put((sample, ev))

            # EOS-trim boundary, one main-thread sync, identical to the sync path.
            keep_frames = len(all_samples)
            if all_samples:
                with torch.inference_mode():
                    last_nonzero = len(all_samples) - 1
                    while last_nonzero >= 0 and bool((all_samples[last_nonzero] == 0).all()):
                        last_nonzero -= 1
                    keep_frames = min(len(all_samples), last_nonzero + 2)
            in_q.put((_END, keep_frames))

            while True:
                o = out_q.get()
                if o is _END:
                    break
                if o is _ERR:
                    raise state["error"] or RuntimeError("stream worker failed")
                if o.numel() > 0:
                    yield o
        finally:
            # Unblock a worker stuck on a full out_q (e.g. consumer GeneratorExit)
            # so it can see _END and exit, then join.
            try:
                in_q.put_nowait((_END, 0))
            except queue.Full:
                pass
            deadline = _time.perf_counter() + 10.0
            while w.is_alive() and _time.perf_counter() < deadline:
                try:
                    out_q.get(timeout=0.1)
                except queue.Empty:
                    pass
            w.join(timeout=1.0)

        # Tail reconciliation on the main thread, after the worker (and the Mimi
        # streaming context) is done, against the same EOS-trimmed frame set.
        tail_audio = None
        with torch.inference_mode():
            if keep_frames > 0:
                full_audio = self._decode_frames(all_samples[:keep_frames])
                if state["streamed"] < full_audio.size(0):
                    tail = full_audio[state["streamed"]:]
                    if tail.numel() > 0:
                        tail_audio = self._watermark_stream_chunk(tail, is_final=True)
        if tail_audio is not None:
            yield tail_audio.detach().to("cpu")


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state_dict[key.removeprefix("module.")] = value
    if not state_dict:
        raise ValueError("Checkpoint did not contain any tensor state_dict entries")
    return state_dict


@contextlib.contextmanager
def _skip_random_init():
    """Skip torch.nn.init random fills during model construction.

    Model(config) builds the full ~8B-param stack and random-initializes every
    weight (kaiming/normal over billions of elements on the CPU), but those
    values are overwritten a moment later by load_state_dict from the checkpoint
    -- pure wasted startup time. Patch the nn.init fills to no-ops during
    construction. The tensors are still allocated; only the random fill is
    skipped. RoPE caches and other computed buffers do NOT use nn.init, so they
    are built normally; load_state_dict (strict) still guarantees every param is
    populated from the checkpoint.
    """
    import torch.nn.init as I
    names = ["kaiming_uniform_", "kaiming_normal_", "normal_", "uniform_",
             "xavier_uniform_", "xavier_normal_", "trunc_normal_"]
    saved = {}

    def _noop(tensor, *a, **k):
        return tensor

    for n in names:
        if hasattr(I, n):
            saved[n] = getattr(I, n)
            setattr(I, n, _noop)
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(I, n, fn)


QUANT_SCHEMES = ("int8", "int4")


def _quantize_and_move(model: Model, scheme: str, device: str, dtype: torch.dtype) -> None:
    """Weight-only quantize the backbone/decoder Linear layers WHILE moving the
    model to `device`, without ever materializing the full bf16 model on `device`.

    The point is fitting a card that cannot hold the ~17 GB bf16 model: we quantize
    layer by layer, so the device only ever holds the already-quantized weights plus
    one bf16 layer in flight. Each target Linear is moved to `device` and quantized
    in place (its bf16 weight is freed as the int8/int4 weight replaces it), keeping
    the device peak near the quantized model size (~10 GB int8, ~6-7 GB int4) rather
    than the bf16 peak. int4 packing (tinygemm `_convert_weight_to_int4pack`) is
    CUDA-only, so it MUST happen on the device; the same layer-wise path serves int8.

    This is weight-only quantization: weights are stored int8/int4 and dequantized
    to `dtype` for the matmul. For MisoTTS it is purely a MEMORY lever, not a speed
    one - the frame-by-frame decode's tiny per-step matmuls (M=1 backbone, M=10
    decoder) cannot feed the hardware low-precision GEMMs (int8 `_int_mm`, fp8/fp4
    `_scaled_mm` all require M>=16), so dynamic/activation quant and hardware
    fp8/nvfp4 give no compute win here. Integer weight-only delivers the memory
    saving on ANY GPU, which is what lowers the hardware floor. The embeddings,
    output heads, and projection stay in `dtype` (small and precision-sensitive).
    """
    import torch.nn as nn
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

    cfg = {"int8": int8_weight_only, "int4": int4_weight_only}.get(scheme)
    if cfg is None:
        raise ValueError(f"unknown quantize scheme {scheme!r}; expected one of {QUANT_SCHEMES}")

    model.to(dtype=dtype)  # cast to the compute dtype on CPU before quantizing
    n = 0
    for fqn, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "head" not in fqn and "projection" not in fqn:
            mod.to(device)
            quantize_(mod, cfg())
            n += 1
    model.to(device)  # move the remaining (unquantized) params: embeddings, heads, norms
    print(f"[load] {scheme} weight-only: quantized {n} Linear layers; model on {device}",
          flush=True)


def _load_model(
    model_path_or_repo_id: str,
    config: ModelArgs,
    device: str,
    dtype: torch.dtype,
    quantize: Optional[str] = None,
    prequantized: bool = False,
) -> Model:
    if os.path.isfile(model_path_or_repo_id):
        model_file = model_path_or_repo_id
    elif os.path.isdir(model_path_or_repo_id):
        name = "model.pt" if prequantized else "model.safetensors"
        model_file = os.path.join(model_path_or_repo_id, name)
    else:
        name = "model.pt" if prequantized else "model.safetensors"
        model_file = hf_hub_download(repo_id=model_path_or_repo_id, filename=name)

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Could not resolve model checkpoint: {model_path_or_repo_id}")

    with _skip_random_init():
        model = Model(config)

    if prequantized:
        # A torch.save'd quantized state_dict: torchao AffineQuantizedTensor weights
        # for the backbone/decoder Linears, bf16 elsewhere (see
        # deploy/build_quant_checkpoint.py). Load with assign=True so each param
        # OBJECT is replaced by the quantized tensor (you cannot copy_ an int8/int4
        # tensor into the fp32 Parameter the model was built with). weights_only is
        # False because unpickling the tensor subclass needs it -- only load
        # checkpoints you trust (our own BigBlueCeiling repos).
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, assign=True)
        model.to(device)
    elif model_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load .safetensors checkpoint files") from exc
        model.load_state_dict(load_file(model_file, device="cpu"))
        if quantize:
            # Quantize layer-wise while moving to `device` so a small card never has
            # to hold the full bf16 model (see _quantize_and_move). This is the
            # fallback when a pre-quantized repo is unavailable.
            _quantize_and_move(model, quantize, device, dtype)
        else:
            model.to(device=device, dtype=dtype)
    else:
        checkpoint = torch.load(model_file, map_location="cpu")
        model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
        if quantize:
            _quantize_and_move(model, quantize, device, dtype)
        else:
            model.to(device=device, dtype=dtype)

    model.eval()
    return model


def load_miso_8b(
    device: str = "cuda",
    model_path_or_repo_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    quantize: Optional[str] = None,
    prequantized: bool = False,
) -> Generator:
    source = model_path_or_repo_id or os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    model = _load_model(source, MISO_TTS_8B_CONFIG, device=device, dtype=dtype,
                        quantize=quantize, prequantized=prequantized)
    return Generator(model)
