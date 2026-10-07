"""Interval arithmetic on (start, end) tuples, seconds."""

Interval = tuple[float, float]


def merge(items: list[Interval], gap: float = 0.0) -> list[Interval]:
    out: list[Interval] = []
    for s, e in sorted(items):
        if e <= s:
            continue
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def invert(cuts: list[Interval], lo: float, hi: float) -> list[Interval]:
    """Complement of `cuts` inside [lo, hi]."""
    out, cur = [], lo
    for s, e in merge(cuts):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def subtract(keep: list[Interval], cuts: list[Interval], min_len: float = 0.0) -> list[Interval]:
    out = []
    for a, b in merge(keep):
        for s, e in invert(cuts, a, b):
            if e - s > max(min_len, 1e-6):
                out.append((s, e))
    return out


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out = []
    for s1, e1 in merge(a):
        for s2, e2 in merge(b):
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return sorted(out)


def total(items: list[Interval]) -> float:
    return sum(e - s for s, e in merge(items))


def contains(items: list[Interval], t: float) -> bool:
    return any(s <= t < e for s, e in items)


def overlaps(items: list[Interval], s: float, e: float) -> bool:
    return any(a < e and b > s for a, b in items)
