"""One upbeat track per short (port of the music half of based-stamps-and-music).

Rules from the editor: Epidemic Sound (ES_) files only, no game music, never repeat a song within 10 shorts
(carried across batches in a log), loud from the first second, strong beat, major leaning, instrumental over
vocal, longer than the short, -18 dB under the voice. Picked by analysis, not by ear: the report lists them.
"""
import json
import os
import re

import numpy as np

from . import config, media

EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
SR = 11025
N_FFT, HOP = 1024, 256


def library(folder: str, prefix: str, ban: list[str]) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for n in sorted(os.listdir(folder)):
        low = n.lower()
        if not low.endswith(EXT) or not n.startswith(prefix) or any(b in low for b in ban):
            continue
        out.append(os.path.join(folder, n))
    return out


def title_key(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"\((instrumental( version)?|inst\.?)\)", "", base, flags=re.I)
    return re.sub(r"\s+", " ", base).strip().lower()


def is_instrumental(name: str) -> bool:
    return bool(re.search(r"\((instrumental|inst)", os.path.basename(name), re.I))


def analyze(path: str, seconds: int = 80) -> dict:
    info = media.probe(path)
    ch = info.audio[0].channels
    x = media.decode_audio(path, 0, rate=SR, dur=seconds, channels=ch,
                           filters="pan=mono|c0=0.5*c0+0.5*c1" if ch > 1 else None)[:, 0]
    dur = info.duration
    n = (len(x) - N_FFT) // HOP + 1
    if n < 20:
        return {"dur": dur, "ok": False}
    frames = x[np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]] * np.hanning(N_FFT)
    S = np.abs(np.fft.rfft(frames, axis=1))
    rms = np.sqrt((frames ** 2).mean(axis=1) / 0.375 + 1e-12)           # undo the window's power loss
    fr = SR / HOP
    flux = np.maximum(np.diff(np.log1p(10 * S), axis=0), 0).sum(axis=1)
    on = np.concatenate([[0], flux])
    ac = np.correlate(on - on.mean(), on - on.mean(), "full")[len(on) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(fr * 60 / 180), int(fr * 60 / 70)
    k = lo + int(np.argmax(ac[lo:hi]))
    tempo = 60 * fr / k
    f = np.fft.rfftfreq(N_FFT, 1 / SR)
    sel = (f > 55) & (f < 1760)
    pc = np.round(12 * np.log2(f[sel] / 261.6256)).astype(int) % 12
    chroma = np.array([(S[:, sel][:, pc == i] ** 2).sum() for i in range(12)])
    bm = max(np.corrcoef(np.roll(MAJ, i), chroma)[0, 1] for i in range(12))
    bn = max(np.corrcoef(np.roll(MIN, i), chroma)[0, 1] for i in range(12))
    def r_db(a: float, b: float) -> float:
        seg = rms[int(a * fr):max(int(b * fr), int(a * fr) + 1)]
        return float(20 * np.log10(seg.mean() + 1e-9)) if seg.size else -99.0
    lead = float(np.argmax(rms > 0.1 * rms.max()) / fr)
    return {"dur": dur, "ok": True, "tempo": float(tempo), "pulse": float(ac[k]), "onset": float(on.mean()),
            "r0_5": r_db(0, 5), "r5_15": r_db(5, 15), "major": bool(bm > bn), "lead": lead}


def features(paths: list[str], cache_file: str, log=print) -> dict[str, dict]:
    cache = {}
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    out, dirty = {}, False
    for p in paths:
        st = os.stat(p)
        key = f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            try:
                cache[key] = analyze(p)
            except media.MediaError:
                cache[key] = {"ok": False, "dur": 0}
            dirty = True
        out[p] = cache[key]
    if dirty:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    return out


def scores(feats: dict[str, dict], need: float, c: dict) -> dict[str, float]:
    """Higher is better; tracks that cannot work (too short, unreadable) are left out."""
    ok = {p: f for p, f in feats.items() if f.get("ok") and f["dur"] >= need + f["lead"] + 0.5}
    if not ok:
        return {}
    ons = sorted(f["onset"] for f in ok.values())
    twins = {}
    for p in feats:
        if is_instrumental(p):
            twins[title_key(p)] = True
    out = {}
    for p, f in ok.items():
        rank = ons.index(f["onset"]) / max(len(ons) - 1, 1)
        s = 3.0 * float(np.clip((f["r0_5"] - (c["first5_db"] - 12)) / 12, 0, 1))
        s += 2.0 * min(f["pulse"] / 0.5, 1.0) + 1.5 * rank
        s += 1.0 if f["major"] else 0.0
        s += 0.5 if 100 <= f["tempo"] <= 140 else 0.0
        if is_instrumental(p):
            s += 1.5
        elif twins.get(title_key(p)):
            s -= 2.0                                       # the plain version of a track with an instrumental twin has vocals
        if f["lead"] > 1.0:
            s -= 3.0
        out[p] = s
    return out


def choose(shorts: list, sc: dict[str, float], recent: list[str], no_repeat: int) -> list[str | None]:
    picks = []
    for _ in shorts:
        block = set(recent[-no_repeat:]) if no_repeat else set()
        best = next((p for p in sorted(sc, key=lambda p: (-sc[p], p)) if os.path.basename(p) not in block), None)
        picks.append(best)
        if best:
            recent.append(os.path.basename(best))
    return picks


def prepare(src: str, start: float, length: float, gain_db: float, out: str) -> str:
    af = f"volume={gain_db}dB,afade=t=out:st={max(length - 0.3, 0):.3f}:d=0.3"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    media.run([media.FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-vn",
               "-af", af, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", out])
    return out


def run(plan_, cfg: dict, shorts=None, log=print, **_):
    from .plan import Overlay
    c = cfg["music"]
    if not c["enabled"]:
        return
    folder = config.path(cfg, "music")
    work = os.path.expanduser(cfg["paths"]["work"])
    lib = library(folder, c["prefix"], c["ban"])
    if not lib:
        log(f"  no {c['prefix']}* music in {folder}")
        return
    sel = [s for s in plan_.shorts if (not shorts or s.name in shorts) and s.enabled and s.pieces]
    for s in sel:
        s.clear_stage("music")
    need = max((s.tl_duration() for s in sel), default=0.0)
    feats = features(lib, os.path.join(work, "music_features.json"), log)
    log_file = os.path.join(work, c["log"])
    recent = json.load(open(log_file))["recent"] if os.path.isfile(log_file) else []
    picks = choose(sel, scores(feats, need, c), recent, c["no_repeat"])
    for s, p in zip(sel, picks):
        if p is None:
            s.note("[music] no track long enough that has not been used in the last "
                   f"{c['no_repeat']} shorts")
            continue
        L = s.tl_duration()
        f = feats[p]
        out = prepare(p, f["lead"], L, c["gain_db"], os.path.join(s.work, "music.wav"))
        s.overlays.append(Overlay("music", out, 0.0, L, cfg["tracks"]["music"],
                                  {"src": p, "src_in": round(f["lead"], 2), "gain_db": c["gain_db"]},
                                  os.path.splitext(os.path.basename(p))[0], "music"))
        s.stats["music"] = os.path.basename(p)
        log(f"  {s.name}: {os.path.splitext(os.path.basename(p))[0]} ({f['tempo']:.0f} BPM, "
            f"{'major' if f['major'] else 'minor'}, {c['gain_db']:.0f} dB)")
    os.makedirs(work, exist_ok=True)
    with open(log_file, "w") as fh:
        json.dump({"recent": recent[-200:]}, fh)
