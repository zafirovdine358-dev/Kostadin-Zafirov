"""Files you can import by hand when scripting is not available: an EDL per short and the captions as SRT.

Standard library only. The EDL carries the cuts (no framing, overlays or audio processing: those need `apply`).
"""
import os
import shutil

from . import srt, timebase
from .plan import EditPlan, Short


def _tc_frames(tc: str, fps: float) -> int:
    parts = [int(x) for x in tc.replace(";", ":").split(":")] if tc else [0, 0, 0, 0]
    h, m, s, f = (parts + [0, 0, 0, 0])[:4]
    return ((h * 60 + m) * 60 + s) * int(round(timebase.exact(fps))) + f


def edl(short: Short, tl_fps: float, title: str | None = None) -> str:
    """CMX3600 cut list of the short's pieces, source timecode = the clip's own start timecode + offset."""
    name = os.path.basename(short.source)
    reel = os.path.splitext(name)[0][:8].upper().replace(" ", "_")
    base = _tc_frames(short.stats.get("start_tc", ""), short.fps)
    lines = [f"TITLE: {title or short.name}", "FCM: NON-DROP FRAME", ""]
    bounds = short.frame_bounds(tl_fps)
    for k, p in enumerate(short.pieces):
        s_in = base + timebase.to_frame(p.src_in, short.fps)
        n = bounds[k + 1] - bounds[k]
        s_out = s_in + int(round(n * timebase.exact(short.fps) / timebase.exact(tl_fps)))
        lines.append(f"{k + 1:03d}  {reel:<8} AA/V  C        {timebase.timecode(s_in, short.fps)} "
                     f"{timebase.timecode(s_out, short.fps)} {timebase.timecode(bounds[k], tl_fps)} "
                     f"{timebase.timecode(bounds[k + 1], tl_fps)}")
        lines.append(f"* FROM CLIP NAME: {name}")
        if p.note:
            lines.append(f"* COMMENT: {p.note}")
    return "\n".join(lines) + "\n"


def write_all(plan: EditPlan, out_dir: str, shorts: list[str] | None = None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    fps = (plan.timeline or {}).get("fps", 29.97)
    out = []
    for s in plan.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        p = os.path.join(out_dir, f"{s.name}.edl")
        with open(p, "w") as f:
            f.write(edl(s, fps))
        out.append(p)
        caps = s.stats.get("caption_list")
        if caps:
            keep = [(a, b, t) for a, b, t, kind in caps if not (kind == "based" and s.overlays_have("based"))]
            out.append(srt.write(keep, os.path.join(out_dir, f"{s.name}.srt")))
        elif s.subtitles and os.path.isfile(s.subtitles):
            out.append(shutil.copy(s.subtitles, os.path.join(out_dir, f"{s.name}.srt")))
    return out
