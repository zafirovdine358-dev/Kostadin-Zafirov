"""Faces in the footage: detection (YuNet/SFace via OpenCV), tracks, and who is the host."""
import os
from dataclasses import dataclass, field

import numpy as np

from . import media


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    ex: float                    # eye midpoint
    ey: float
    score: float = 1.0
    sim: float | None = None     # similarity to the host (SFace), if known
    mouth: float = 0.0           # lower-face motion minus upper-face motion since the previous sample
    track: int = -1
    emb: np.ndarray | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


@dataclass
class Sample:
    t: float
    faces: list = field(default_factory=list)
    H: Face | None = None
    G: Face | None = None


class YuNet:
    """OpenCV YuNet detector (+ SFace embeddings when a host vector is given). Frames in, source-pixel faces out."""

    def __init__(self, model: str, sface: str = "", host: np.ndarray | None = None, score: float = 0.6,
                 min_w: float = 35.0):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(os.path.expanduser(model), "", (960, 540), score)
        self.rec = cv2.FaceRecognizerSF.create(os.path.expanduser(sface), "") if sface else None
        self.host, self.min_w = host, min_w

    def embed(self, img, row) -> np.ndarray | None:
        if self.rec is None:
            return None
        e = self.rec.feature(self.rec.alignCrop(img, row)).flatten()
        return e / (np.linalg.norm(e) + 1e-9)

    def faces(self, img: np.ndarray, to_src: float = 1.0) -> list[Face]:
        cv2 = self.cv2
        h, w = img.shape[:2]
        dw, dh = 960, max(2, int(round(960 * h / w)))
        self.det.setInputSize((dw, dh))
        _, rows = self.det.detect(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA))
        out = []
        k = w / dw
        for r in (rows if rows is not None else []):
            if r[2] * k < self.min_w:
                continue
            big = r.copy()
            big[:14] *= k
            e = self.embed(img, big)
            sim = float(e @ self.host) if e is not None and self.host is not None else None
            out.append(Face(big[0] * to_src, big[1] * to_src, big[2] * to_src, big[3] * to_src,
                            (big[4] + big[6]) / 2 * to_src, (big[5] + big[7]) / 2 * to_src,
                            float(r[14]), sim, emb=e))
        return out


def mouth_motion(prev_gray: np.ndarray, gray: np.ndarray, f: Face, k: float = 1.0) -> float:
    """Lower-third change minus forehead change inside the face box (camera shake affects both)."""
    x0, y0, w, h = int(f.x * k), int(f.y * k), int(f.w * k), int(f.h * k)
    if w < 8 or h < 8:
        return 0.0
    cols = slice(max(x0 + int(0.2 * w), 0), max(x0 + int(0.8 * w), 1))

    def diff(a: float, b: float) -> float:
        rows = slice(max(y0 + int(a * h), 0), max(y0 + int(b * h), 1))
        p, c = prev_gray[rows, cols].astype(np.float32), gray[rows, cols].astype(np.float32)
        if p.size == 0 or p.shape != c.shape:
            return 0.0
        return float(np.abs(c - p).mean() / (c.mean() + 8.0))
    return max(0.0, diff(0.62, 1.0) - diff(0.10, 0.42))


def sample_faces(path: str, ranges: list[tuple[float, float]], fps: float, src_w: int, src_h: int, detector,
                 frame_w: int = 1920, log=None) -> list[Sample]:
    """Detect faces at `fps` inside each (start, end) of the source file."""
    fh = max(2, int(round(frame_w * src_h / src_w)))
    k = src_w / frame_w
    out: list[Sample] = []
    for a, b in ranges:
        prev, prev_faces = None, []
        for t, img in media.grab_frames(path, a, b - a, fps, frame_w, fh):
            faces = detector.faces(img, to_src=k)
            gray = img.mean(axis=2).astype(np.uint8) if img.ndim == 3 else img
            if prev is not None:
                for f in faces:
                    near = [p for p in prev_faces if abs(p.cx - f.cx) < f.w and abs(p.y - f.y) < f.h]
                    if near:
                        f.mouth = mouth_motion(prev, gray, f, 1.0 / k)
            prev, prev_faces = gray, faces
            out.append(Sample(round(t, 3), faces))
    return out


def build_tracks(samples: list[Sample], src_w: float, max_gap: float = 1.5) -> int:
    """Greedy nearest-neighbour tracking; sets Face.track. Returns the number of tracks."""
    last: list[tuple[float, float, float, float]] = []      # per track: t, cx, cy, w
    for s in samples:
        used = set()
        for f in sorted(s.faces, key=lambda f: -f.w):
            best, bd = -1, 1e18
            for i, (t, cx, cy, w) in enumerate(last):
                if i in used or s.t - t > max_gap:
                    continue
                d = np.hypot(f.cx - cx, f.y + f.h / 2 - cy)
                if d < max(1.2 * max(f.w, w), 0.1 * src_w) and d < bd:
                    best, bd = i, d
            if best < 0:
                last.append((s.t, f.cx, f.y + f.h / 2, f.w))
                best = len(last) - 1
            else:
                last[best] = (s.t, f.cx, f.y + f.h / 2, f.w)
            used.add(best)
            f.track = best
    return len(last)


def assign_roles(samples: list[Sample], turns: list, min_presence: float = 0.1, host_sim: float = 0.33) -> dict:
    """Decide which track is the host (H) and which the guest (G).

    With SFace similarities the host is the track that looks like the host; otherwise the track whose mouth
    moves while the host's voice is on (and not while the guest's is). Returns {"H": track|None, "G": track|None}.
    """
    by_track: dict[int, list[Face]] = {}
    for s in samples:
        for f in s.faces:
            by_track.setdefault(f.track, []).append(f)
    n = max(len(samples), 1)
    tracks = [t for t, fs in by_track.items() if len(fs) / n >= min_presence]
    if not tracks:
        return {"H": None, "G": None}
    sims = {t: [f.sim for f in by_track[t] if f.sim is not None] for t in tracks}
    if any(sims.values()):
        score = {t: (float(np.mean(v)) if v else -1.0) for t, v in sims.items()}
        host = max(score, key=score.get)
        H = host if score[host] >= host_sim else None
        rest = [t for t in tracks if t != H]
    else:
        votes = {t: 0.0 for t in tracks}
        for s in samples:
            spk = _speaker_at(turns, s.t)
            if not spk:
                continue
            sign = 1.0 if spk == "H" else -1.0
            for f in s.faces:
                if f.track in votes:
                    votes[f.track] += sign * f.mouth
        H = max(votes, key=votes.get) if votes and max(votes.values()) > 0 else None
        rest = [t for t in tracks if t != H]
    G = max(rest, key=lambda t: len(by_track[t])) if rest else None
    return {"H": H, "G": G}


def _speaker_at(turns: list, t: float) -> str:
    sp = ""
    for tr in turns:
        if tr.s - 0.1 <= t < tr.e + 0.3:
            sp = tr.spk
    return sp


def label_samples(samples: list[Sample], roles: dict) -> None:
    for s in samples:
        s.H = max((f for f in s.faces if roles["H"] is not None and f.track == roles["H"]),
                  key=lambda f: f.w, default=None)
        s.G = max((f for f in s.faces if roles["G"] is not None and f.track == roles["G"]),
                  key=lambda f: f.w, default=None)


def host_centroid(embeddings_by_short: dict[str, np.ndarray], min_shorts: int = 10, sim: float = 0.45) -> np.ndarray | None:
    """The host is the face that recurs across shorts (port of ca/host.py)."""
    names = list(embeddings_by_short)
    if len(names) < 2:
        return None
    E = np.vstack([embeddings_by_short[n] for n in names])
    owner = np.array([n for n in names for _ in range(len(embeddings_by_short[n]))])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    need = min(min_shorts, max(2, int(np.ceil(0.6 * len(names)))))
    cnt = np.array([sum(bool((S[i][owner == n] > sim).any()) for n in names if n != owner[i]) for i in range(len(E))])
    if not (cnt >= need).any():
        return None
    c = E[cnt >= need].mean(0)
    c /= np.linalg.norm(c) + 1e-9
    for _ in range(3):
        s = E @ c
        if not (s > sim).any():
            break
        c = E[s > sim].mean(0)
        c /= np.linalg.norm(c) + 1e-9
    return c


def save_samples(path: str, ranges: list[tuple[float, float]], samples: list[Sample]) -> None:
    rows, emb, idx = [], [], []
    for i, s in enumerate(samples):
        for f in s.faces:
            rows.append([f.x, f.y, f.w, f.h, f.ex, f.ey, f.score, f.mouth])
            emb.append(f.emb if f.emb is not None else np.zeros(0))
            idx.append(i)
    dim = max((len(e) for e in emb), default=0)
    E = np.array([e if len(e) == dim else np.zeros(dim) for e in emb]) if dim else np.zeros((len(rows), 0))
    np.savez_compressed(path, t=np.array([s.t for s in samples]), idx=np.array(idx, dtype=int),
                        rows=np.array(rows).reshape(-1, 8), emb=E, ranges=np.array(ranges).reshape(-1, 2))


def load_samples(path: str) -> tuple[list[tuple[float, float]], list[Sample]]:
    z = np.load(path)
    samples = [Sample(float(t)) for t in z["t"]]
    for i, r, e in zip(z["idx"], z["rows"], z["emb"] if z["emb"].shape[1] else [None] * len(z["idx"])):
        samples[int(i)].faces.append(Face(*[float(v) for v in r[:7]], None, float(r[7]),
                                          emb=e if e is not None else None))
    return [tuple(map(float, r)) for r in z["ranges"]], samples


def host_from_photo(detector, path: str) -> np.ndarray | None:
    """Embedding of the biggest face in a photo of the host."""
    import cv2
    img = cv2.imread(os.path.expanduser(path))
    if img is None:
        return None
    fs = detector.faces(img)
    return max(fs, key=lambda f: f.w).emb if fs else None
