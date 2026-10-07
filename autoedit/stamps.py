"""Batch stamp on every short (port of the stamp half of based-stamps-and-music).

The stamp itself comes from the BASED Editor Portal. Download the 9:16 PNG yourself and point stamps.file (or
--stamp) at it, or leave it in the Stamps folder. This tool never touches the portal or its link.
"""
import glob
import os

from . import config, media, sprites
from .plan import Overlay


def find_stamp(cfg: dict, explicit: str | None = None) -> str:
    c = cfg["stamps"]
    for cand in (explicit, c.get("file")):
        if cand and os.path.isfile(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    folder = config.path(cfg, "stamps")
    found = glob.glob(os.path.join(folder, "WM-*.png")) if os.path.isdir(folder) else []
    return max(found, key=os.path.getmtime) if found else ""


def stamp_window(short, c: dict) -> tuple[float, float, bool]:
    """(start, end, on_a_cut): the last ~30 s for shorts of a minute or more, else the last half, starting on a cut."""
    end = short.tl_duration()
    target = end - c["tail_s"] if end >= c["long_short_s"] else end / 2
    cuts = short.piece_starts()
    if len(cuts) < 2:
        return max(target, 0.0), end, False
    best = min(cuts[1:], key=lambda t: abs(t - target))
    return best, end, True


def run(plan_, cfg: dict, shorts=None, log=print, stamp: str | None = None, **_):
    c = cfg["stamps"]
    if not c["enabled"]:
        return
    path = find_stamp(cfg, stamp)
    if not path:
        log("  no stamp file: download the 9:16 stamp from the Editor Portal and pass --stamp, "
            f"or put WM-*.png in {config.path(cfg, 'stamps')}")
        return
    sel = [s for s in plan_.shorts if (not shorts or s.name in shorts) and s.enabled]
    wins = {s.name: stamp_window(s, c) for s in sel}
    master, err = path, ""
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and sel:
        try:
            master = sprites.still_master(path, max(b - a for a, b, _ in wins.values()),
                                          plan_.timeline or cfg["timeline"],
                                          os.path.join(os.path.expanduser(cfg["paths"]["work"]), "stills"))
        except (ImportError, media.MediaError) as ex:
            err = str(ex)
    for s in sel:
        s.clear_stage("stamps")
        a, b, on_cut = wins[s.name]
        if err:
            s.note(f"[stamps] could not turn the stamp into a clip: {err}")
        s.overlays.append(Overlay("stamp", master, a, b, cfg["tracks"]["stamp"], {"scaling": 2, "still": path},
                                  os.path.basename(path), "stamps"))
        s.stats["stamp"] = {"start": round(a, 2), "end": round(b, 2), "on_cut": on_cut}
        if not on_cut:
            s.note("[stamps] no cut near the target: the stamp starts mid-shot")
        log(f"  {s.name}: stamp {a:.1f}-{b:.1f}s" + ("" if on_cut else " (not on a cut)"))
