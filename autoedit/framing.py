"""Reframing maths (port of the coordinate section of based-camera-angles).

Premiere position -> where the source centre sits on the canvas; Resolve (Crop scaling, zoom z):
    canvas x = W/2 + pan  + (u - sw/2) * z
    canvas y = H/2 - tilt + (v - sh/2) * z
so pan = Px - W/2 and tilt = H/2 - Py.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Canvas:
    w: float = 1080.0
    h: float = 1920.0
    cx: float = 540.0      # where the head should be centred (between the blue guides)
    eye_y: float = 570.0
    margin: float = 3.0


def min_zoom(sw: float, sh: float, c: Canvas) -> float:
    """Smallest zoom at which the source still covers the canvas (no black bars)."""
    return max((c.w + 2 * c.margin) / sw, (c.h + 2 * c.margin) / sh)


def clamp(px: float, py: float, z: float, sw: float, sh: float, c: Canvas) -> tuple[float, float]:
    lo_x, hi_x = c.w - sw * z / 2 + c.margin, sw * z / 2 - c.margin
    lo_y, hi_y = c.h - sh * z / 2 + c.margin, sh * z / 2 - c.margin
    return min(max(px, lo_x), hi_x), min(max(py, lo_y), hi_y)


def frame_point(u: float, v: float, z: float, sw: float, sh: float, c: Canvas) -> tuple[float, float]:
    """Position that puts source point (u, v) at the target spot, clamped so no bars show."""
    return clamp(c.cx - (u - sw / 2) * z, c.eye_y - (v - sh / 2) * z, z, sw, sh, c)


def residual(u: float, v: float, z: float, sw: float, sh: float, c: Canvas) -> float:
    """How far (px) the point ends up from the target after clamping."""
    px, py = frame_point(u, v, z, sw, sh, c)
    return max(abs(px + (u - sw / 2) * z - c.cx), abs(py + (v - sh / 2) * z - c.eye_y))


def to_resolve(px: float, py: float, c: Canvas) -> tuple[float, float]:
    return px - c.w / 2, c.h / 2 - py


def steps(sw: float, sh: float, c: Canvas, base: list[float]) -> list[float]:
    """Allowed zoom steps; sources smaller than 4K start at the zoom that fills the canvas."""
    z0 = max(1.0, min_zoom(sw, sh, c))
    return [round(z0 * b, 4) for b in base]


def pick_step(face_w: float, sw: float, sh: float, c: Canvas, base: list[float], px: list[float]) -> int:
    """Index into steps(): 0 if the face is already big enough on the canvas, else the next zoom up."""
    z0 = max(1.0, min_zoom(sw, sh, c))
    shown = face_w * z0
    if shown >= px[0]:
        return 0
    if shown >= px[1]:
        return min(1, len(base) - 1)
    return min(2, len(base) - 1)


def dp(ts, U, V, tol: float) -> list[int]:
    """Douglas-Peucker on the (u, v) track: indices of the keys worth keeping."""
    keep = {0, len(ts) - 1}

    def rec(a: int, b: int):
        if b <= a + 1:
            return
        best, bi = 0.0, None
        for i in range(a + 1, b):
            r = (ts[i] - ts[a]) / (ts[b] - ts[a] + 1e-9)
            e = max(abs(U[i] - (U[a] + r * (U[b] - U[a]))), abs(V[i] - (V[a] + r * (V[b] - V[a]))) * 1.3)
            if e > best:
                best, bi = e, i
        if bi is not None and best > tol:
            keep.add(bi)
            rec(a, bi)
            rec(bi, b)
    if len(ts) > 1:
        rec(0, len(ts) - 1)
    return sorted(keep)


def median_smooth(x, k: int = 5):
    x = np.asarray(x, dtype=float)
    h = k // 2
    return np.array([np.median(x[max(0, i - h): i + h + 1]) for i in range(len(x))])


def chunk_track(ts, U, V, tols: list[float], min_chunk: float) -> list[tuple[float, float, float, float]]:
    """Split a moving track into steady chunks: (t0, t1, median u, median v). Larger tolerance until chunks are long enough."""
    ts = np.asarray(ts, dtype=float)
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    span = max(ts[-1] - ts[0], 1e-6)
    max_chunks = max(1, int(span / min_chunk))
    idx = [0, len(ts) - 1]
    for tol in tols:
        idx = dp(ts, U, V, tol)
        if len(idx) - 1 <= max_chunks:
            break
    cuts = [ts[0]]
    for i in idx[1:-1]:
        if ts[i] - cuts[-1] >= min_chunk and ts[-1] - ts[i] >= min_chunk:
            cuts.append(ts[i])
    cuts.append(ts[-1])
    while len(cuts) - 1 > max_chunks and len(cuts) > 2:          # still too many: drop the closest boundary
        gaps = [cuts[i + 1] - cuts[i - 1] for i in range(1, len(cuts) - 1)]
        del cuts[1 + int(np.argmin(gaps))]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        m = (ts >= a) & (ts <= b) if b >= cuts[-1] else (ts >= a) & (ts < b)
        if not m.any():
            m = np.ones(len(ts), bool)
        out.append((float(a), float(b), float(np.median(U[m])), float(np.median(V[m]))))
    return out
