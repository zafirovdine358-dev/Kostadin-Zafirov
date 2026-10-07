"""Find the newest dated shoot folder and describe its clips (port of import-based)."""
import datetime as dt
import os
import re
from dataclasses import dataclass

import numpy as np

from . import config, media
from .plan import EditPlan, Piece, Short

_DMY = re.compile(r"^(?P<a>\d{1,2})[:/._\-](?P<b>\d{1,2})[:/._\-](?P<y>\d{2,4})\s*(?P<rest>.*)$")
_ISO = re.compile(r"^(?P<y>\d{4})[-_.](?P<m>\d{1,2})[-_.](?P<d>\d{1,2})\s*(?P<rest>.*)$")


@dataclass
class Batch:
    date: dt.date
    host: str
    path: str


def parse_batch_name(name: str, order: str = "DMY") -> tuple[dt.date, str] | None:
    """'29:09:26 Lian' (Finder shows 29/09/26) -> (2026-09-29, 'Lian')."""
    m = _ISO.match(name)
    try:
        if m:
            return dt.date(int(m["y"]), int(m["m"]), int(m["d"])), m["rest"].strip()
        m = _DMY.match(name)
        if m:
            a, b, y = int(m["a"]), int(m["b"]), int(m["y"])
            y += 2000 if y < 100 else 0
            d, mo = (a, b) if order.upper().startswith("D") else (b, a)
            return dt.date(y, mo, d), m["rest"].strip()
    except ValueError:
        return None
    return None


def find_batches(footage: str, order: str = "DMY", skip: list[str] | None = None) -> list[Batch]:
    skip = [s.lower() for s in (skip or [])]
    out = []
    for n in os.listdir(footage):
        full = os.path.join(footage, n)
        if not os.path.isdir(full) or any(s in n.lower() for s in skip):
            continue
        parsed = parse_batch_name(n, order)
        if parsed:
            out.append(Batch(parsed[0], parsed[1], full))
    return sorted(out, key=lambda b: (b.date, b.path), reverse=True)


def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_media(folder: str, exts: list[str]) -> list[str]:
    names = [n for n in os.listdir(folder)
             if not n.startswith(".") and os.path.splitext(n)[1].lower() in exts]
    return [os.path.join(folder, n) for n in sorted(names, key=_natural)]


def analyze_audio(path: str, info: media.MediaInfo, empty_db: float = -60.0) -> dict:
    """Which channels carry sound, and is it one mic or two (host + guest)?"""
    chans = []  # (stream, channel, peak_db, samples)
    for a in info.audio:
        x = media.decode_audio(path, a.index, rate=8000, channels=a.channels)
        for c in range(a.channels):
            peak = float(np.abs(x[:, c]).max()) if len(x) else 0.0
            chans.append((a.index, c, 20 * np.log10(peak + 1e-9), x[:, c]))
    live = [c for c in chans if c[2] > empty_db]
    out = {"mic": "single", "a1": None, "a2": None,
           "empty": [[c[0], c[1]] for c in chans if c[2] <= empty_db],
           "peaks": {f"{c[0]}:{c[1]}": round(c[2], 1) for c in chans}}
    if not live:
        return out
    out["a1"] = [live[0][0], live[0][1]]
    if len(live) >= 2:
        a, b = live[0][3], live[1][3]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        same = n > 0 and float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) > 0.995 \
            and abs(np.sqrt((a ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-12) - 1) < 0.12
        if not same:
            out["mic"] = "two"
            out["a2"] = [live[1][0], live[1][1]]
    return out


def ingest(cfg: dict, folder: str | None = None, shorts: list[str] | None = None,
           log=print) -> tuple[EditPlan, str]:
    """Build a fresh plan for the newest batch (or `folder`). Returns (plan, work_dir)."""
    ing = cfg["ingest"]
    if folder:
        folder = os.path.expanduser(folder)
        parsed = parse_batch_name(os.path.basename(folder.rstrip("/\\")), ing["date_order"])
        batch = Batch(parsed[0] if parsed else dt.date.today(), parsed[1] if parsed else "", folder)
    else:
        root = config.path(cfg, "footage")
        if not os.path.isdir(root):
            raise SystemExit(f"footage folder not found: {root} (set paths.based_root / paths.footage)")
        found = find_batches(root, ing["date_order"], ing["skip_words"])
        if not found:
            raise SystemExit(f"no dated shoot folders in {root}")
        batch = found[0]
    files = list_media(batch.path, ing["media_ext"])
    if shorts:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in shorts]
    if not files:
        raise SystemExit(f"no media files in {batch.path}")
    slug = f"{batch.date:%Y-%m-%d} {batch.host}".strip()
    work = os.path.join(os.path.expanduser(cfg["paths"]["work"]), slug)
    plan = EditPlan(batch={"name": "", "folder": batch.path, "date": batch.date.isoformat(), "host": batch.host,
                           "slug": slug},
                    timeline=dict(cfg["timeline"]))
    for f in files:
        info = media.probe(f)
        audio = analyze_audio(f, info, ing["empty_db"]) if info.audio else {"mic": "none", "a1": None,
                                                                           "a2": None, "empty": []}
        s = Short(name=info.name, source=os.path.abspath(f), fps=info.fps, width=info.width,
                  height=info.height, duration=info.duration, audio=audio,
                  work=os.path.join(work, info.name))
        s.pieces = [Piece(0.0, info.duration)]
        if not audio["a1"]:
            s.enabled = False
            s.note("[ingest] no audible audio channel: left out of every stage")
        s.stats["start_tc"] = info.timecode
        plan.shorts.append(s)
        log(f"  {info.name}: {info.duration:.1f}s {info.width}x{info.height}@{info.fps:.2f} "
            f"audio={audio['mic']} empty={len(audio['empty'])}")
    plan.done("ingest")
    return plan, work
