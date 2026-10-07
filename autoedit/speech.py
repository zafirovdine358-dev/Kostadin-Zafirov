"""Speech/energy maths shared by the cut stages (ports of rc/gaps2.py, fcb/fclib.py, fcb/edge.py)."""
import numpy as np

HOP = 0.02            # 20 ms energy frames
FPS_E = 50            # frames per second of the energy track
ABS_FLOOR = -28.0     # never call anything quieter than this (dB below speech level) speech


def energy_db(x: np.ndarray, sr: int) -> np.ndarray:
    """20 ms RMS in dB relative to the speech level (median of the louder half), like fclib.energy."""
    h = int(sr * HOP)
    n = len(x) // h
    if n == 0:
        return np.zeros(0)
    fr = x[: n * h].reshape(n, h)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    return db - np.median(np.sort(db)[len(db) // 2:])


def smooth3(e: np.ndarray) -> np.ndarray:
    return np.convolve(e, np.ones(3) / 3, "same") if len(e) else e


def threshold(e: np.ndarray, t: float, frac: float = 0.35) -> float:
    """floor + frac*(speech - floor) from the 15th/85th percentiles within +-2 s."""
    if not len(e):
        return 0.0
    i = int(t * FPS_E)
    w = e[max(0, i - 100): i + 100]
    if not len(w):
        w = e
    fl, sp = np.percentile(w, 15), np.percentile(w, 85)
    return float(max(fl + frac * (sp - fl), ABS_FLOOR))


def _clamp(i: int, n: int) -> int:
    return max(0, min(i, n - 1))


def speech_end(e: np.ndarray, word_end: float, limit: float, frac: float = 0.35) -> float:
    """Where speech really stops near a word end: last loud frame, 160 ms of quiet ends the scan."""
    if not len(e):
        return word_end
    es, th, n = smooth3(e), threshold(e, word_end, frac), len(e)
    i, lim = _clamp(int((word_end - 0.2) * FPS_E), n), _clamp(int(limit * FPS_E), n)
    last, quiet = i, 0
    while i < lim:
        if es[i] > th:
            last, quiet = i, 0
        else:
            quiet += 1
            if quiet >= 8:
                break
        i += 1
    return (last + 1) / FPS_E


def speech_start(e: np.ndarray, word_start: float, limit: float, frac: float = 0.35) -> float:
    if not len(e):
        return word_start
    es, th, n = smooth3(e), threshold(e, word_start, frac), len(e)
    i, lim = _clamp(int((word_start + 0.15) * FPS_E), n), _clamp(int(limit * FPS_E), n)
    first, quiet = i, 0
    while i > lim:
        if es[i] > th:
            first, quiet = i, 0
        else:
            quiet += 1
            if quiet >= 8:
                break
        i -= 1
    return first / FPS_E


def loud_run(e: np.ndarray, x: float, y: float, frac: float = 0.5, need: int = 12) -> bool:
    """Is there sustained sound (>= 0.24 s above the local threshold) between x and y?"""
    if not len(e) or y <= x:
        return False
    es = smooth3(e)
    th = threshold(e, (x + y) / 2, frac)
    a = int(x * FPS_E) + 3
    b = max(int(y * FPS_E) - 3, a)
    return int((es[a:b] > th).sum()) >= need


def guard_cut(e: np.ndarray, a: float, b: float, guard_db: float = 8.0, quiet: float = 0.25) -> tuple[float, float]:
    """Pull a cut's edges inward while the audio there is still loud (missed first/last words).

    A quiet dip of up to `quiet` seconds does not stop the scan. Returns (a, a) if nothing is left.
    """
    n = len(e)
    if not n or b <= a:
        return a, b
    lim, qn = -guard_db, int(quiet * FPS_E)
    ia, ib = int(round(a * FPS_E)), min(int(round(b * FPS_E)), n)
    new_a, run, i = ia, 0, ia
    while i < ib:
        if e[i] > lim:
            new_a, run = i + 1, 0
        else:
            run += 1
            if run > qn:
                break
        i += 1
    new_b, run, j = ib, 0, ib - 1
    while j >= ia:
        if e[j] > lim:
            new_b, run = j, 0
        else:
            run += 1
            if run > qn:
                break
        j -= 1
    na = max(a, new_a / FPS_E) if new_a > ia else a
    nb = min(b, new_b / FPS_E) if new_b < ib else b
    return (na, nb) if nb > na else (a, a)


def island_words(e: np.ndarray, min_len: float = 0.12, bridge: float = 0.35, frac: float = 0.35):
    """No-ASR fallback: speech islands from energy alone, as (start, end) pseudo words."""
    if not len(e):
        return []
    es = smooth3(e)
    th = np.array([threshold(e, i / FPS_E, frac) for i in range(0, len(e), 25)])
    thr = np.repeat(th, 25)[: len(e)]
    on = es > thr
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append([i / FPS_E, j / FPS_E])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for s, t in out:
        if merged and s - merged[-1][1] <= bridge:
            merged[-1][1] = t
        else:
            merged.append([s, t])
    return [(s, t) for s, t in merged if t - s >= min_len]
