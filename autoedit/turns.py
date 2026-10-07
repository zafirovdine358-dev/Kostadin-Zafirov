"""Who is speaking when (port of the speaker-turn part of based-camera-angles)."""
import re
from dataclasses import dataclass

import numpy as np

from . import speech

SR = 16000
MIN_PITCH_RATIO = 1.18          # two cluster medians this far apart in Hz: different people
MIN_SHAPE_DISTANCE = 6.0         # or this far apart in tone colour (log-mel units)
HOST_CUES = re.compile(r"\b(based|tiktok|follow\w*|recommend\w*|product\w*|try|spray|cream|powder|mousse|clay|"
                       r"pomade|shampoo|conditioner|lotion|wash|live|giveaway\w*|discount\w*)\b", re.I)


@dataclass
class Sentence:
    s: float
    e: float
    first: int           # index of first word
    last: int            # index of last word (inclusive)
    text: str = ""


@dataclass
class Turn:
    s: float
    e: float
    spk: str


def sentences(words: list, max_gap: float = 0.7, max_len: float = 14.0) -> list[Sentence]:
    """Split words into sentences at . ? ! or a long pause."""
    out: list[Sentence] = []
    start = None
    for i, w in enumerate(words):
        if start is None:
            start = i
        end_here = bool(w.t) and w.t.rstrip('"\')').endswith((".", "?", "!"))
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = nxt is None or nxt.s - w.e >= max_gap
        long_ = words[i].e - words[start].s >= max_len
        if end_here or gap or long_:
            out.append(Sentence(words[start].s, w.e, start, i, " ".join(x.t for x in words[start:i + 1])))
            start = None
    return out


# ---------------------------------------------------------------- two mics
def mic_labels(sents: list[Sentence], chan: dict, roles: dict, min_margin: float = 3.0) -> list[tuple[str, float]]:
    """Per sentence: the louder mic wins. roles maps 'A1'/'A2' to 'H' or 'G'. Returns (label, dB margin).

    When the mics differ by less than min_margin dB (both hear both people) the previous speaker is kept.
    """
    def lvl(x):
        h = int(SR * speech.HOP)
        n = len(x) // h
        return 20 * np.log10(np.sqrt((x[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    l1, l2 = lvl(chan["a1"]), lvl(chan["a2"])
    out = []
    for s in sents:
        a, b = int(s.s * speech.FPS_E), max(int(s.e * speech.FPS_E), int(s.s * speech.FPS_E) + 1)
        x1, x2 = l1[a:b], l2[a:b]
        n = min(len(x1), len(x2))
        if n == 0:
            out.append((roles["A1"], 0.0))
            continue
        d = float(np.median(x1[:n] - x2[:n]))
        if abs(d) < min_margin and out:
            out.append((out[-1][0], abs(d)))
        else:
            out.append((roles["A1"] if d >= 0 else roles["A2"], abs(d)))
    return out


# ---------------------------------------------------------------- one mic
def _mel_fb(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)
    pts = np.linspace(mel(80), mel(sr / 2 - 100), n_mels + 2)
    hz = 700 * (10 ** (pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def _rows(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    n = (len(x) - size) // hop + 1
    if n <= 0:
        return np.zeros((0, size), dtype=x.dtype)
    return x[np.arange(size)[None, :] + hop * np.arange(n)[:, None]]


def _f0(x: np.ndarray, sr: int) -> float:
    """Median pitch of the louder voiced frames, 0 if none (FFT autocorrelation)."""
    size = 640
    rows = _rows(x, size, 320)
    if not len(rows):
        return 0.0
    en = (rows ** 2).mean(1)
    rows = rows[en >= np.percentile(en, 60)][:150]
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for fr in rows:
        fr = fr - fr.mean()
        r = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * size)) ** 2)[:size]
        if r[0] <= 1e-9:
            continue
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] / r[0] > 0.35:
            f0s.append(sr / k)
    return float(np.median(f0s)) if f0s else 0.0


def voice_features(x: np.ndarray, sr: int = SR, n_mels: int = 20) -> np.ndarray | None:
    """Spectral shape (loudness removed) plus median pitch of one stretch of speech."""
    n_fft, hop = 512, 160
    frames = _rows(x, n_fft, hop)
    if len(frames) < 8:
        return None
    pw = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    en = pw.sum(axis=1)
    keep = en > np.percentile(en, 40)
    if keep.sum() < 4:
        return None
    shape = np.log(pw[keep] @ _mel_fb(n_mels, n_fft, sr).T + 1e-10).mean(axis=0)
    return np.concatenate([shape - shape.mean(), [np.log(_f0(x, sr) + 1.0) * 3.0]])


def kmeans2(X: np.ndarray, iters: int = 30) -> np.ndarray:
    """Two clusters, deterministic start (the two furthest points)."""
    n = len(X)
    if n < 2:
        return np.zeros(n, dtype=int)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    c = np.stack([Z[i], Z[j]])
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        lab = ((Z[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([Z[lab == k].mean(0) if (lab == k).any() else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def two_voices(feats: np.ndarray, lab: np.ndarray) -> bool:
    """Is the split into two clusters a real difference between people, or one voice cut in half?

    Judged on the raw features, not the standardised ones (those make any tiny difference look big): a clearly
    different pitch, or a clearly different tone colour.
    """
    if min(int((lab == 0).sum()), int((lab == 1).sum())) < 2:
        return False
    shape_d = float(np.linalg.norm(feats[lab == 0, :-1].mean(0) - feats[lab == 1, :-1].mean(0)))
    f0 = np.exp(feats[:, -1] / 3.0) - 1.0
    med = [float(np.median(f0[(lab == k) & (f0 > 40)])) if ((lab == k) & (f0 > 40)).any() else 0.0 for k in (0, 1)]
    ratio = max(med) / min(med) if min(med) > 0 else 1.0
    return ratio >= MIN_PITCH_RATIO or shape_d >= MIN_SHAPE_DISTANCE


def host_score(texts: list[str]) -> float:
    return sum(1 for t in texts if HOST_CUES.search(t)) / max(len(texts), 1)


def voice_labels(sents: list[Sentence], x: np.ndarray, sr: int = SR) -> list[str]:
    """Cluster sentences into two voices and call the host the one with the host cues (else who spoke first)."""
    feats, idx = [], []
    for k, s in enumerate(sents):
        f = voice_features(x[int(s.s * sr): int(s.e * sr)], sr) if s.e - s.s >= 0.4 else None
        if f is not None:
            feats.append(f)
            idx.append(k)
    labels = [""] * len(sents)
    if len(feats) < 2:
        return ["H"] * len(sents)
    F = np.array(feats)
    lab = kmeans2(F)
    if not two_voices(F, lab):
        return ["H"] * len(sents)                  # one voice: do not invent a second speaker
    for k, l in zip(idx, lab):
        labels[k] = str(int(l))
    # sentences too short for features inherit the previous label
    last = labels[idx[0]]
    for k in range(len(labels)):
        labels[k] = labels[k] or last
        last = labels[k]
    texts = {c: [sents[k].text for k in range(len(sents)) if labels[k] == c] for c in ("0", "1")}
    s0, s1 = host_score(texts["0"]), host_score(texts["1"])
    host = ("0" if s0 > s1 else "1") if abs(s0 - s1) > 1e-9 else labels[0]
    return ["H" if l == host else "G" for l in labels]


# ---------------------------------------------------------------- turns
def merge_turns(items: list[tuple[float, float, str]]) -> list[Turn]:
    out: list[Turn] = []
    for s, e, l in items:
        if out and out[-1].spk == l:
            out[-1].e = max(out[-1].e, e)
        else:
            out.append(Turn(s, e, l))
    return out


def build_turns(sents: list[Sentence], labels: list[str], min_turn: float = 0.5) -> list[Turn]:
    """Merge same-speaker runs; absorb backchannels (< min_turn) and overlaps into the previous turn."""
    T = merge_turns([(s.s, s.e, l) for s, l in zip(sents, labels)])
    for _ in range(3):
        for i in range(1, len(T)):
            dur, overlap = T[i].e - T[i].s, T[i].s < T[i - 1].e - 0.05
            if dur < min_turn or overlap:
                if dur < min_turn or (i + 1 < len(T) and T[i + 1].spk == T[i - 1].spk):
                    T[i].spk = T[i - 1].spk
        T = merge_turns([(t.s, t.e, t.spk) for t in T])
    return T


def speaker_at(turns: list[Turn], t: float) -> str:
    sp = turns[0].spk if turns else ""
    for tr in turns:
        if tr.s - 0.1 <= t:
            sp = tr.spk
    return sp
