"""Voice clean-up and levelling (port of based-audio-fix) as an ffmpeg chain, rendered per short.

Resolve's scripting API cannot add Fairlight effects, so each short gets one finished dialogue track: the live
mic of each piece, rumble cut, light denoise, optional match-EQ to a reference voice, levelled to the target
speech level, pauses pulled down by a 2:1 expander, hard-limited, with 3-frame constant-power crossfades at cuts.
Claude (and this tool) cannot hear: everything here is measured, so the report lists the numbers. Your ears win.
"""
import os

import numpy as np

from . import media
from .plan import Short

SR = 48000
FC = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
      5000, 6300, 8000, 10000, 12500]
Q_THIRD_OCTAVE = 4.32


# ---------------------------------------------------------------- measuring (port of au/cmp.py)
def measure(x: np.ndarray, sr: int = SR) -> dict:
    w = int(sr * 0.03)
    n = len(x) // w
    if n < 5:
        return {"speech": -99.0, "peak": -99.0, "floor": [-99.0] * 3, "active": 0.0, "bands": np.zeros(len(FC)),
                "gap": 0.0}
    fr = x[: n * w].reshape(n, w)
    r = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    nz = r > -100
    top = np.percentile(r[nz], 95)
    act = r > top - 25
    sp = 20 * np.log10(np.sqrt((fr[act] ** 2).mean()) + 1e-9)
    fl = [float(np.percentile(r[nz], q)) for q in (1, 5, 10)]
    W, hop = 4096, 2048
    k = (len(x) - W) // hop
    b = np.zeros(len(FC))
    if k > 2:
        idx = np.arange(W)[None, :] + hop * np.arange(k)[:, None]
        F = x[idx] * np.hanning(W)
        e = np.sqrt((F ** 2).mean(axis=1))
        a2 = e > np.percentile(e, 95) * 10 ** (-25 / 20)
        S = (np.abs(np.fft.rfft(F[a2], axis=1)) ** 2).mean(axis=0)
        f = np.fft.rfftfreq(W, 1 / sr)
        b = np.array([10 * np.log10(S[(f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].sum() + 1e-20) for fc in FC])
        b = b - 10 * np.log10((10 ** (b / 10)).sum())
    return {"speech": float(sp), "peak": float(20 * np.log10(np.abs(x).max() + 1e-9)), "floor": fl,
            "active": float(act.mean()), "bands": b, "gap": float(fl[2] - sp)}


def match_eq(ref: np.ndarray, cur: np.ndarray, lo: float, hi: float, prev: np.ndarray | None = None) -> np.ndarray:
    """Per-band gain (dB) that moves the current tone toward the reference, smoothed and clamped (au/eq.py)."""
    d = cur - ref
    ds = np.convolve(np.pad(d, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    g = -ds
    if prev is not None:
        g = prev + 0.8 * g
    g = np.clip(g, lo, hi)
    g[np.array(FC) < 100] = 0.0
    return g


def eq_filters(gains: np.ndarray) -> list[str]:
    return [f"equalizer=f={fc}:width_type=q:w={Q_THIRD_OCTAVE}:g={g:.2f}" for fc, g in zip(FC, gains) if abs(g) >= 0.3]


def prep_chain(c: dict, gains: np.ndarray | None) -> str:
    parts = [f"highpass=f={c['hp_hz']:.0f}"]
    if c["denoise_nr"] > 0:
        parts.append(f"afftdn=nr={c['denoise_nr']:.1f}:nf=-50")
    if gains is not None:
        parts += eq_filters(gains)
    return ",".join(parts)


def finish_chain(c: dict, gain_db: float) -> str:
    """Level, expander (pauses 2:1 below the voice), limiter."""
    thr = c["target_rms"] - c["expander_db"]
    r = c["expander_ratio"]
    pts = f"-90/{thr + (-90 - thr) * r:.1f}|{thr:.1f}/{thr:.1f}|0/0"
    ceil = 10 ** (c["ceiling_db"] / 20)
    return (f"volume={gain_db:.2f}dB,compand=attacks=0.01:decays=0.2:points={pts}:soft-knee=2,"
            f"alimiter=limit={ceil:.4f}:attack=5:release=50:level=disabled")


# ---------------------------------------------------------------- assembling
def pick_channels(short: Short, roles: dict) -> list[str]:
    """Which mic is live for each piece: the one belonging to whoever is talking."""
    two = short.audio.get("mic") == "two" and short.audio.get("a2")
    if not two:
        return ["a1"] * len(short.pieces)
    by_spk = {v: k.lower() for k, v in roles.items()}          # {"G": "a1", "H": "a2"}
    return [by_spk.get(p.spk, "a1") for p in short.pieces]


def assemble(chans: dict[str, np.ndarray], pieces: list, pick: list[str], sr: int, xf_frames: int,
             fps: float) -> np.ndarray:
    """Cut the live mic of each piece together; constant-power crossfades centred on each real cut."""
    segs, ab = [], []
    for p, ch in zip(pieces, pick):
        a, b = int(round(p.src_in * sr)), int(round(p.src_out * sr))
        segs.append(chans[ch][a:b].copy())
        ab.append((a, b))
    y = np.concatenate(segs) if segs else np.zeros(0, np.float32)
    h = int(round(xf_frames / fps * sr / 2))
    pos = 0
    for k in range(len(pieces) - 1):
        pos += len(segs[k])
        contiguous = ab[k][1] == ab[k + 1][0] and pick[k] == pick[k + 1]
        if h < 2 or contiguous:
            continue
        ca, cb = chans[pick[k]], chans[pick[k + 1]]
        A = ca[ab[k][1] - h: ab[k][1] + h]
        B = cb[ab[k + 1][0] - h: ab[k + 1][0] + h]
        if len(A) == 2 * h and len(B) == 2 * h and pos - h >= 0 and pos + h <= len(y):
            t = np.linspace(0, np.pi / 2, 2 * h)
            y[pos - h: pos + h] = A * np.cos(t) + B * np.sin(t)
    return y


def mute(y: np.ndarray, spans: list, sr: int) -> np.ndarray:
    """Silence time spans (another brand's name) with 10 ms fades either side."""
    f = int(0.01 * sr)
    for a, b in spans:
        i, j = max(int(a * sr), 0), min(int(b * sr), len(y))
        if j <= i:
            continue
        a0, b1 = max(i - f, 0), min(j + f, len(y))
        y[a0:i] *= np.linspace(1, 0, i - a0)
        y[j:b1] *= np.linspace(0, 1, b1 - j)
        y[i:j] = 0.0
    return y


def raw_channels(short: Short) -> dict[str, np.ndarray]:
    out = {}
    for key in ("a1", "a2"):
        sel = short.audio.get(key)
        if sel:
            n = next((a.channels for a in media.probe(short.source).audio if a.index == sel[0]), 1)
            out[key] = media.decode_audio(short.source, sel[0], rate=SR, channels=n)[:, sel[1]].copy()
    return out


# ---------------------------------------------------------------- per short
def process(short: Short, cfg: dict, ref_bands: np.ndarray | None, fps: float) -> dict:
    c = cfg["audio"]
    chans = raw_channels(short)
    if not chans:
        raise media.MediaError(f"{short.name}: no audio")
    pick = pick_channels(short, cfg["angles"]["mics"])
    used = sorted(set(pick))
    before = measure(assemble(chans, short.pieces, pick, SR, 0, fps))

    gains, passes = None, (3 if ref_bands is not None else 1)
    for it in range(passes):                       # tone: measure, correct, measure again, refine (au/eq.py)
        prepped = {ch: media.filter_audio(chans[ch], SR, prep_chain(c, gains)) for ch in used}
        cur = measure(assemble(prepped, short.pieces, pick, SR, 0, fps))
        if ref_bands is not None and it < passes - 1:
            gains = match_eq(ref_bands, cur["bands"], c["eq_lo"], c["eq_hi"], gains)
    gain_db = float(np.clip(c["target_rms"] - cur["speech"], -24, 24))
    fin = finish_chain(c, gain_db)
    done = {ch: media.filter_audio(prepped[ch], SR, fin) for ch in used}
    y = assemble(done, short.pieces, pick, SR, c["crossfade_frames"], fps)
    y = mute(y, short.stats.get("mute", []), SR)
    out = os.path.join(short.work, "voice.wav")
    media.write_wav(out, y, SR)
    short.voice = out
    return {"before": before, "after": measure(y), "gain": gain_db,
            "eq": None if gains is None else [round(float(g), 1) for g in gains]}


def run(plan_, cfg: dict, shorts=None, log=print, reference: str | None = None, **_):
    c = cfg["audio"]
    if not c["enabled"]:
        return
    ref_path = reference or c.get("reference", "")
    ref_bands = None
    if ref_path:
        ref_path = os.path.expanduser(ref_path)
        x = media.decode_audio(ref_path, 0, rate=SR, channels=None)
        ref_bands = measure(x.mean(axis=1).astype(np.float32))["bands"]
        log(f"  matching tone to {os.path.basename(ref_path)}")
    fps = (plan_.timeline or cfg["timeline"])["fps"]
    for s in plan_.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        if "enhanced" in s.name.lower():
            log(f"  {s.name}: already enhanced, left alone")
            continue
        s.clear_stage("audio")
        try:
            r = process(s, cfg, ref_bands, fps)
        except media.MediaError as ex:
            s.note(f"[audio] skipped: {ex}")
            log(f"  {s.name}: skipped ({ex})")
            continue
        b, a = r["before"], r["after"]
        s.stats["audio"] = {"speech_before": round(b["speech"], 1), "speech_after": round(a["speech"], 1),
                            "peak_after": round(a["peak"], 1), "gap_after": round(a["gap"], 1),
                            "active_after": round(a["active"], 2), "gain": round(r["gain"], 1)}
        if abs(a["speech"] - c["target_rms"]) > 2.0:
            s.note(f"[audio] speech level {a['speech']:.1f} dB, target {c['target_rms']:.0f}: check by ear")
        log(f"  {s.name}: speech {b['speech']:.1f} -> {a['speech']:.1f} dB, peak {a['peak']:.1f}, "
            f"pauses {a['gap']:.0f} dB under speech, active {a['active']:.2f}")
