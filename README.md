# autoedit

Auto-edit raw IRL footage into a DaVinci Resolve timeline.

Requires `ffmpeg`/`ffprobe` on PATH and Python 3.10+. For Resolve: enable
*Preferences > General > External scripting using: Local* and keep Resolve open.

```
python -m autoedit a.mp4 b.mp4 --name Vlog --target 60     # build timeline in Resolve
python -m autoedit a.mp4 --dry-run --edl cut.edl --json plan.json   # no Resolve; import EDL manually
```

What it does per clip: removes silence (with padding), black/frozen footage, splits long
takes at scene cuts, drops tiny fragments, and optionally keeps the best-scoring
(scene changes + loudness) segments to hit `--target` seconds.

Tests: `python -m pytest tests`
