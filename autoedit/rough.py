"""Rough cut (port of based-rough-cut): cut every stretch where nobody is speaking."""
import math

from . import analysis, intervals as iv, speech
from .plan import Marker, Piece

MIN_SLIVER_FRAMES = 3


def speech_gaps(words: list, duration: float, c: dict) -> list[tuple[float, float, str]]:
    """Cut candidates: the head, gaps >= min_gap between speech, the tail. (start, end, kind)"""
    ws = sorted(words, key=lambda w: w.s)
    if not ws:
        return []
    pre, post, min_gap = c["pre"], c["post"], c["min_gap"]
    cuts = []
    if ws[0].s - pre > c["head_min"]:
        cuts.append((0.0, ws[0].s - pre, "head"))
    run_end = ws[0].e
    for w in ws[1:]:
        if w.s - run_end >= min_gap:
            cuts.append((run_end + post, w.s - pre, "gap"))
        run_end = max(run_end, w.e)
    if duration - (run_end + post) > c["head_min"]:
        cuts.append((run_end + post, duration, "tail"))
    return [(a, b, k) for a, b, k in cuts if b > a]


def snap_keep(keep: list[tuple[float, float]], fps: float, duration: float) -> list[tuple[float, float]]:
    """Kept ranges grow out to whole source frames, like the SOP's ceil/floor on the cuts."""
    out = []
    for a, b in keep:
        a2 = 0.0 if a <= 1e-6 else math.floor(a * fps + 1e-6) / fps
        b2 = duration if b >= duration - 1e-6 else min(math.ceil(b * fps - 1e-6) / fps, duration)
        out.append((a2, b2))
    return iv.merge(out)


def plan_cuts(an: analysis.Analysis, fps: float, c: dict) -> list[tuple[float, float, str]]:
    cuts = []
    for a, b, kind in speech_gaps(an.words, an.duration, c):
        a, b = speech.guard_cut(an.e, a, b, c["guard_db"], c["guard_quiet"])
        if b - a >= c["min_cut_frames"] / fps:
            cuts.append((a, b, kind))
    return cuts


def split_protected(pieces: list[Piece], protect: list[tuple[float, float]]) -> list[Piece]:
    """Cut pieces at the edges of kept 'showing' stretches and flag those as phone shots."""
    out = []
    for p in pieces:
        cur = p.src_in
        for a, b in sorted(protect):
            a, b = max(a, p.src_in), min(b, p.src_out)
            if b - a < 1e-3:
                continue
            if a > cur + 1e-3:
                out.append(Piece(cur, a, flags=list(p.flags)))
            out.append(Piece(a, b, flags=list(p.flags) + ["phone"]))
            cur = b
        if p.src_out > cur + 1e-3:
            out.append(Piece(cur, p.src_out, flags=list(p.flags)))
    return out


def rough_keep(an: analysis.Analysis, fps: float, c: dict, mode: str, judge=None):
    """-> (kept ranges, cuts applied, protected ranges). judge(ranges) -> ranges showing something."""
    cuts = plan_cuts(an, fps, c)
    protect: list[tuple[float, float]] = []
    applied = []
    long_gaps = [(a, b) for a, b, k in cuts if b - a >= c["review_min"] and (k == "gap" or mode == "claude")]
    if mode == "flag":
        protect = list(long_gaps)
    elif mode == "claude" and judge and long_gaps:
        protect = judge(long_gaps)
    for a, b, _ in cuts:
        pieces = iv.subtract([(a, b)], protect)
        applied += pieces
    keep = iv.invert(applied, 0.0, an.duration)
    keep = [k for k in snap_keep(keep, fps, an.duration) if k[1] - k[0] >= MIN_SLIVER_FRAMES / fps]
    return keep, applied, protect


def run(plan, cfg: dict, shorts=None, log=print, showing: str | None = None, **_):
    c = cfg["rough"]
    mode = showing or c["showing"]
    for s in plan.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        s.clear_from("rough")
        an = analysis.load(s, cfg, log)
        judge = None
        if mode == "claude":
            from . import showing as sh
            judge = lambda ranges, s=s: sh.judge(s, ranges, cfg, log)
        if not an.words:
            s.pieces = [Piece(0.0, s.duration)]
            s.note("[rough] no speech found, kept the whole clip")
        else:
            keep, applied, protect = rough_keep(an, s.fps, c, mode, judge)
            s.pieces = split_protected([Piece(a, b) for a, b in keep], protect)
            s.stats["protect"] = [list(p) for p in protect]
            for a, b in protect:
                t = s.src_to_tl(a)
                if t is not None and mode == "flag":
                    s.markers.append(Marker(t, "Yellow", "CHECK showing?",
                                            f"silent {b - a:.1f}s kept: phone or product shown? delete if not",
                                            stage="rough"))
        s.snapshot("rough")
        before, after = s.duration, s.tl_duration()
        s.stats["rough"] = {"before": round(before, 2), "after": round(after, 2), "pieces": len(s.pieces)}
        log(f"  {s.name}: {before:.1f}s -> {after:.1f}s, {len(s.pieces)} pieces"
            + (f", {len(s.stats.get('protect', []))} silent stretches kept for review" if s.stats.get("protect") else ""))
