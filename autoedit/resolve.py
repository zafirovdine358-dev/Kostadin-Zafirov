"""DaVinci Resolve integration (scripting API) plus an EDL fallback."""
import os
import sys

from .plan import Segment

_API_PATHS = [
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
    "/opt/resolve/Developer/Scripting/Modules",
]


def connect():
    env = os.environ.get("RESOLVE_SCRIPT_API")
    paths = ([os.path.join(env, "Modules")] if env else []) + _API_PATHS
    for p in paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    try:
        import DaVinciResolveScript as dvr
    except ImportError as e:
        raise SystemExit("Cannot import DaVinciResolveScript. Set RESOLVE_SCRIPT_API and "
                         "enable Preferences > General > External scripting using: Local.") from e
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise SystemExit("Resolve is not running (or external scripting is disabled).")
    return resolve


def build_timeline(resolve, name: str, clips: list[tuple[str, float, list[Segment]]], fps: float,
                   marker_scenes: bool = False):
    """clips: (path, source_fps, segments). Appends each segment as a subclip."""
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject() or pm.CreateProject(name)
    pool = project.GetMediaPool()
    items = pool.ImportMedia([c[0] for c in clips])
    if not items:
        raise SystemExit("Resolve could not import the media.")
    timeline = pool.CreateEmptyTimeline(name)
    project.SetCurrentTimeline(timeline)
    by_name = {os.path.basename(i.GetClipProperty("File Path")): i for i in items}
    infos = []
    for path, src_fps, segs in clips:
        item = by_name[os.path.basename(path)]
        for s in segs:
            infos.append({"mediaPoolItem": item,
                          "startFrame": int(round(s.start * src_fps)),
                          "endFrame": int(round(s.end * src_fps)) - 1})
    if not pool.AppendToTimeline(infos):
        raise SystemExit("AppendToTimeline failed.")
    return timeline


def write_edl(path: str, title: str, clips: list[tuple[str, float, list[Segment]]], fps: float):
    """CMX3600 EDL - File > Import > Timeline in Resolve if scripting isn't available."""
    def tc(sec: float) -> str:
        f = int(round(sec * fps))
        return f"{f // int(fps) // 3600:02d}:{f // int(fps) // 60 % 60:02d}:{f // int(fps) % 60:02d}:{f % int(fps):02d}"
    lines, n, rec = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""], 1, 0.0
    for clip, _, segs in clips:
        for s in segs:
            lines.append(f"{n:03d}  AX       V     C        {tc(s.start)} {tc(s.end)} {tc(rec)} {tc(rec + s.length)}")
            lines.append(f"* FROM CLIP NAME: {os.path.basename(clip)}")
            rec += s.length
            n += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
