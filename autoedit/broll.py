"""B-roll over a product explanation (port of based-broll, local folder only: download from Frame.io yourself)."""
import os
import re

import numpy as np

from . import analysis, captions as cp, config, media, products as pr, turns as tn
from .plan import Marker, Overlay

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".mxf", ".avi")


def candidates(folder: str, product: str, model: str | None = None) -> list[str]:
    """Clips whose folder or file name mentions the product (and the model, if given)."""
    want = [w for w in re.split(r"[\s_\-]+", product.lower()) if w]
    out = []
    for root, _, files in os.walk(folder):
        for n in files:
            if os.path.splitext(n)[1].lower() not in VIDEO_EXT or n.startswith("."):
                continue
            hay = re.sub(r"[\s_\-]+", " ", os.path.join(root, n).lower())
            if all(w in hay for w in want) and (not model or model.lower() in hay):
                out.append(os.path.join(root, n))
    return sorted(out)


def clean_ranges(path: str, duration: float, fps: float = 4.0, bad_lo: float = 25.0, bad_hi: float = 235.0,
                 jump: float = 12.0, guard: float = 0.3, min_len: float = 1.0) -> list[tuple[float, float]]:
    """Stretches without fades to black/white (finished UGC clips fade between shots)."""
    means = []
    for t, img in media.grab_frames(path, 0, duration, fps, 64, 112):
        means.append(float(img.mean()))
    means = np.array(means)
    bad = (means < bad_lo) | (means > bad_hi)
    bad[1:] |= np.abs(np.diff(means)) > jump
    g = int(round(guard * fps))
    bad = np.convolve(bad.astype(int), np.ones(2 * g + 1, int), "same") > 0
    out, i = [], 0
    while i < len(bad):
        if not bad[i]:
            j = i
            while j < len(bad) and not bad[j]:
                j += 1
            if (j - i) / fps >= min_len:
                out.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return out


def cut_plan(window: float, lo: float, hi: float) -> list[float]:
    """Lengths of the cuts that fill the window, each between lo and hi seconds."""
    n = max(1, int(round(window / ((lo + hi) / 2))))
    while window / n > hi:
        n += 1
    while n > 1 and window / n < lo:
        n -= 1
    return [window / n] * n


def pick(clean: list[tuple[float, float]], lengths: list[float]) -> list[float]:
    """Source start for each cut, spread over the clean footage in order so the shots come from different moments."""
    total = sum(b - a for a, b in clean)
    if not clean or total < min(lengths):
        return []
    starts = []
    for i, ln in enumerate(lengths):
        pos = (i + 0.5) / len(lengths) * total
        for a, b in clean:
            if pos <= b - a:
                starts.append(min(a + max(pos - ln / 2, 0.0), b - ln))
                break
            pos -= b - a
        else:
            starts.append(clean[-1][1] - ln)
    return [max(s, 0.0) for s in starts]


def snap_to_captions(bounds: list[float], caps: list[float], lo: float, tol: float = 0.4) -> list[float]:
    """Move the cut points onto caption starts so a caption never names one thing over the next shot."""
    out = list(bounds)
    for k in range(1, len(out) - 1):
        near = min(caps, key=lambda c: abs(c - out[k]), default=None)
        if near is not None and abs(near - out[k]) <= tol and near - out[k - 1] >= lo and out[-1] - near >= lo:
            out[k] = near
    return out


def explanation_window(words: list, T: list, t_at: float, max_s: float) -> tuple[float, float]:
    """The sentences the product is being explained in: from the sentence at t_at while the same person talks."""
    sents = tn.sentences(words)
    k = next((i for i, s in enumerate(sents) if s.s - 0.05 <= t_at <= s.e + 0.05), None)
    if k is None:
        k = max((i for i, s in enumerate(sents) if s.s <= t_at), default=0)
    start = sents[k].s
    spk = tn.speaker_at(T, start) if T else ""
    end = sents[k].e
    for s in sents[k + 1:]:
        if (T and tn.speaker_at(T, s.s) != spk) or s.e - start > max_s or s.s - end > 1.2:
            break
        end = s.e
    return start, end


def run(plan_, cfg: dict, shorts=None, log=print, short: str | None = None, product: str | None = None,
        at: float | None = None, model: str | None = None, **_):
    c = cfg["broll"]
    sel = [s for s in plan_.shorts if (short and s.name == short) or (not short and (not shorts or s.name in shorts))]
    if not product:
        raise SystemExit("broll needs --product, e.g. --product 'curl cream'")
    folder = config.path(cfg, "broll")
    files = candidates(folder, product, model) if os.path.isdir(folder) else []
    if not files:
        raise SystemExit(f"no B-roll for '{product}' under {folder}: put the clips there (folder or file name "
                         "containing the product, e.g. B-roll/David/curl cream routine.mp4)")
    if not sel:
        raise SystemExit(f"no short named '{short}' in this plan")
    s = sel[0]
    an = analysis.load(s, cfg, log, need_text=True)
    T = [tn.Turn(float(a), float(b), sp) for a, b, sp in s.stats.get("turns", [])]
    words = cp.timeline_words(s, an, T)
    if at is None:
        ments = [m for m in pr.find_mentions(words, cfg["products"]["items"], 0.0, cfg["captions"].get("asr_fixes"))
                 if m.name == product or product in m.name]
        if not ments:
            raise SystemExit(f"'{product}' is never mentioned in {s.name}: pass --at SECONDS")
        at = ments[0].t
    t0, t1 = explanation_window(words, T, at, c["max_s"])
    lengths = cut_plan(t1 - t0, c["cut_min_s"], c["cut_max_s"])
    bounds = [t0 + sum(lengths[:i]) for i in range(len(lengths) + 1)]
    caps = [x[0] for x in s.stats.get("caption_list", [])]
    bounds = snap_to_captions(bounds, caps, c["cut_min_s"] * 0.8)
    s.clear_stage("broll")
    src = files[0]
    info = media.probe(src)
    clean = clean_ranges(src, info.duration)
    starts = pick(clean, [b - a for a, b in zip(bounds, bounds[1:])])
    if not starts:
        raise SystemExit(f"{os.path.basename(src)} has no clean stretch long enough for B-roll")
    for (a, b), st in zip(zip(bounds, bounds[1:]), starts):
        s.overlays.append(Overlay("broll", src, a, b, cfg["tracks"]["broll"],
                                  {"src_in": round(st, 3), "scaling": 3, "audio": False, "fps": info.fps}, os.path.basename(src), "broll"))
    s.markers.append(Marker(t0, "Blue", f"B-ROLL: {product}", f"{os.path.basename(src)}, {len(starts)} cuts", stage="broll"))
    s.stats["broll"] = {"product": product, "file": src, "window": [round(t0, 2), round(t1, 2)], "cuts": len(starts)}
    log(f"  {s.name}: {len(starts)} B-roll cut(s) over {t0:.1f}-{t1:.1f}s from {os.path.basename(src)}")
