# Voice prompts (pre-assigned voices)

Drop a reference clip and its transcript here to register a named voice. The
serving core discovers them at startup and exposes them by name.

For a voice named `alice`:

```
deploy/prompts/alice.wav   # a clean reference clip of the target voice
deploy/prompts/alice.txt   # the exact transcript of alice.wav
```

The reference is wrapped as a conditioning context segment; its Mimi codes are
computed once and reused across requests (so a registered voice encodes only
once). Request the voice by id (`"voice": "alice"`) in either API surface.

The built-in `default` voice always exists (the model's own voice, no reference).

This directory is mounted into the container at `/workspace/prompts`
(`MISO_PROMPTS_DIR`). The `.wav`/`.txt` files are gitignored; only this README is
tracked.
