"""Pure edit-decision logic: turn analysis results into keep-segments."""
from dataclasses import dataclass, field

Interval = tuple[float, float]


@dataclass
class Settings:
    pad: float = 0.15          # keep this much audio around speech
    min_clip: float = 1.0      # drop kept segments shorter than this
    max_clip: float = 12.0     # split segments longer than this (at scene cuts if any)
    merge_gap: float = 0.4     # bridge gaps shorter than this
    target_len: float | None = None  # trim to this total length (seconds), best-scored first


@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0

    @property
    def length(self) -> float:
        return self.end - self.start


def invert(cuts: list[Interval], duration: float) -> list[Interval]:
    """Complement of `cuts` within [0, duration]."""
    out, cur = [], 0.0
    for s, e in sorted(cuts):
        s, e = max(s, 0.0), min(e, duration)
        if e <= s:
            continue
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        out.append((cur, duration))
    return out


def merge(intervals: list[Interval], gap: float) -> list[Interval]:
    out: list[Interval] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def split_long(seg: Interval, scenes: list[float], max_len: float) -> list[Interval]:
    s, e = seg
    parts = []
    while e - s > max_len:
        cands = [t for t in scenes if s + max_len / 2 <= t <= s + max_len]
        cut = max(cands) if cands else s + max_len
        parts.append((s, cut))
        s = cut
    parts.append((s, e))
    return parts


def build_segments(duration: float, silences: list[Interval], dead: list[Interval],
                   scenes: list[float], loudness: list[float], cfg: Settings) -> list[Segment]:
    # shrink silences by pad so speech keeps breathing room
    cuts = [(s + cfg.pad, e - cfg.pad) for s, e in silences if e - s > 2 * cfg.pad]
    cuts += dead
    keep = merge(invert(cuts, duration), cfg.merge_gap)
    pieces = [p for k in keep for p in split_long(k, scenes, cfg.max_clip)]
    segs = [Segment(s, e) for s, e in pieces if e - s >= cfg.min_clip]
    for sg in segs:
        sg.score = _score(sg, scenes, loudness)
    if cfg.target_len is not None:
        segs = _fit_target(segs, cfg.target_len)
    return segs


def _score(seg: Segment, scenes: list[float], loudness: list[float]) -> float:
    """More scene changes + louder audio = more eventful."""
    n_scenes = sum(seg.start <= t < seg.end for t in scenes)
    lo, hi = int(seg.start), max(int(seg.end), int(seg.start) + 1)
    window = loudness[lo:hi]
    loud = (sum(window) / len(window) + 60) / 60 if window else 0.0  # dB -> ~0..1
    return n_scenes / max(seg.length, 1.0) + max(loud, 0.0)


def _fit_target(segs: list[Segment], target: float) -> list[Segment]:
    chosen, total = [], 0.0
    for sg in sorted(segs, key=lambda x: -x.score):
        if total + sg.length <= target:
            chosen.append(sg)
            total += sg.length
    return sorted(chosen, key=lambda x: x.start)
