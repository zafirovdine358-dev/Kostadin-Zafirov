"""Camera angles (port of based-camera-angles): cut on every speaker change, reframe onto the speaker."""
import os

import numpy as np

from . import analysis, faces as fc, framing as fr, speech, turns as tn
from .plan import Marker, Piece, Short

MAX_RESIDUAL_OK = 80.0       # px: good enough to call the head centred
MAX_RESIDUAL_BAD = 150.0     # px: beyond this the head is not where it should be


def canvas(cfg: dict, tl: dict | None = None) -> fr.Canvas:
    a = cfg["angles"]
    t = tl or cfg["timeline"]
    return fr.Canvas(float(t["width"]), float(t["height"]), a["center_x"], a["eye_y"], a["margin"])


# ------------------------------------------------------------------ turns -> pieces
def split_at(pieces: list[Piece], cuts: list[float], fps: float, min_gap_frames: int) -> list[Piece]:
    """Cut pieces at the given source times, skipping cuts within min_gap_frames of an existing edge."""
    out = []
    eps = min_gap_frames / fps
    for p in pieces:
        pts = sorted(c for c in cuts if p.src_in + eps < c < p.src_out - eps)
        edges = [p.src_in] + pts + [p.src_out]
        for a, b in zip(edges, edges[1:]):
            out.append(Piece(a, b, p.spk, p.who, p.zoom, p.pan, p.tilt, list(p.flags), p.note))
    return out


def turn_cuts(sents: list[tn.Sentence], words: list, T: list[tn.Turn], e: np.ndarray, fps: float,
              lead_frames: int) -> list[float]:
    """Where each new speaker starts: just before their first word."""
    cuts = []
    for tr in T[1:]:
        first = next((s for s in sents if s.s >= tr.s - 1e-6), None)
        if first is None:
            continue
        w0 = words[first.first]
        t0 = speech.speech_start(e, w0.s, max(w0.s - 0.4, 0.0))
        cuts.append(max(t0 - lead_frames / fps, 0.0))
    return cuts


def label_pieces(pieces: list[Piece], T: list[tn.Turn], fps: float) -> None:
    for p in pieces:
        p.spk = tn.speaker_at(T, p.src_in + 2 / fps) if T else ""


def speaker_turns(short: Short, an: analysis.Analysis, cfg: dict, log=print):
    """-> (sentences, turns). Two mics: the louder mic. One mic: voice clustering. labels.txt overrides."""
    sents = tn.sentences(an.words)
    if not sents:
        return [], []
    manual = os.path.join(short.work, "labels.txt")
    labels = None
    if os.path.isfile(manual):
        toks = open(manual).read().split()
        if len(toks) == len(sents) and set(toks) <= {"H", "G"}:
            labels = toks
            log(f"  {short.name}: using labels.txt")
        else:
            log(f"  {short.name}: labels.txt has {len(toks)} labels for {len(sents)} sentences, ignoring it")
    if labels is None and short.audio.get("mic") == "two" and {"a1", "a2"} <= set(an.chan):
        labels = [l for l, _ in tn.mic_labels(sents, an.chan, cfg["angles"]["mics"])]
    if labels is None:
        labels = tn.voice_labels(sents, an.audio)
    return sents, tn.build_turns(sents, labels, cfg["angles"]["min_turn"])


# ------------------------------------------------------------------ framing
def _track(samples: list[fc.Sample], who: str, lo: float, hi: float):
    pts = [(s.t, getattr(s, who)) for s in samples if lo - 0.05 <= s.t <= hi and getattr(s, who) is not None]
    return pts


def frame_pieces(pieces: list[Piece], samples: list[fc.Sample], sw: float, sh: float, cfg: dict,
                 c: fr.Canvas) -> tuple[list[Piece], list[tuple[float, str, str, str]]]:
    """Pick one zoom for the short, then centre the speaker's head in every piece (static or in steady chunks)."""
    a = cfg["angles"]
    k = sw / 3840.0                              # thresholds in the plugin are 4K pixels
    steps = fr.steps(sw, sh, c, a["scale_steps"])
    markers: list[tuple[float, str, str, str]] = []

    # 1. which face each piece is about, and where it is
    info = []
    for p in pieces:
        sp = p.spk or "H"
        other = "G" if sp == "H" else "H"
        rs = [s for s in samples if p.src_in - 0.05 <= s.t <= p.src_out]
        n = max(len(rs), 1)
        cov = {w: sum(getattr(s, w) is not None for s in rs) / n for w in ("H", "G")}
        who, note = sp, ""
        if cov[sp] < a["min_cover"]:
            if cov[other] >= 0.6:
                who, note = other, "speaker not visible, stayed on the other person"
            else:
                who, note = None, f"no usable face (cover {cov[sp]:.2f}/{cov[other]:.2f})"
        pts = _track(samples, who, p.src_in, p.src_out) if who else []
        if who and not pts:
            who, note = None, "no usable face"
        info.append({"p": p, "who": who, "note": note, "pts": pts})

    # 2. one zoom step for the whole short
    framed = [i for i in info if i["who"]]
    face_w = [np.median([f.w for _, f in i["pts"]]) for i in framed]
    dur = [i["p"].dur for i in framed]
    step = 0
    if framed:
        wmed = float(np.average(face_w, weights=dur))
        step = fr.pick_step(wmed, sw, sh, c, a["scale_steps"], a["face_px"])
        for cand in range(step, len(steps)):
            ok = 0.0
            for i, w in zip(framed, dur):
                u0 = np.median([f.cx for _, f in i["pts"][:8]])
                v0 = np.median([f.ey for _, f in i["pts"][:8]])
                if fr.residual(u0, v0, steps[cand], sw, sh, c) <= MAX_RESIDUAL_OK:
                    ok += w
            step = cand
            if ok >= 0.7 * sum(dur):
                break
    z_short = steps[step]

    # 3. frame every piece
    out: list[Piece] = []
    prev: Piece | None = None
    for i in info:
        p = i["p"]
        if i["who"] is None:
            if p.dur <= 0.67 and prev is not None:
                q = Piece(p.src_in, p.src_out, p.spk, prev.who, prev.zoom, prev.pan, prev.tilt,
                          list(p.flags), "tiny, copied the previous framing")
            else:
                q = Piece(p.src_in, p.src_out, p.spk, "", steps[0], 0.0, 0.0, p.flags + ["bad_angle"], i["note"])
                markers.append((p.src_in, "Red", "BAD ANGLE - fix", i["note"]))
            out.append(q)
            prev = q
            continue
        ts = np.array([t for t, _ in i["pts"]])
        U = fr.median_smooth([f.cx for _, f in i["pts"]])
        V = fr.median_smooth([f.ey for _, f in i["pts"]])
        z = z_short
        u0, v0 = float(np.median(U[:8])), float(np.median(V[:8]))
        note = i["note"]
        while fr.residual(u0, v0, z, sw, sh, c) > MAX_RESIDUAL_BAD and z != steps[-1]:
            z = steps[steps.index(z) + 1]
            note = (note + "; " if note else "") + f"zoomed to {z:.2f} to centre the head"
        moving = (U.max() - U.min() > a["static_dx"] * k) or (V.max() - V.min() > a["static_dy"] * k)
        flags = list(p.flags)
        if moving and len(ts) >= 4:
            chunks = fr.chunk_track(ts, U, V, [t * k for t in a["dp_tol"]] + [150 * k, 220 * k], a["chunk_min_s"])
        else:
            chunks = [(p.src_in, p.src_out, u0, v0)]
        if len(chunks) > 1:
            flags.append("stab")
            markers.append((p.src_in, "Green", "STABILISED", f"{len(chunks)} steady shots; add keyframes to refine"))
        edges = [p.src_in] + [ch[1] for ch in chunks[:-1]] + [p.src_out]
        for (a0, b0), ch in zip(zip(edges, edges[1:]), chunks):
            px, py = fr.frame_point(ch[2], ch[3], z, sw, sh, c)
            pan, tilt = fr.to_resolve(px, py, c)
            q = Piece(a0, b0, p.spk, i["who"], z, round(pan, 1), round(tilt, 1), list(flags), note)
            if fr.residual(ch[2], ch[3], z, sw, sh, c) > MAX_RESIDUAL_BAD:
                q.flags.append("bad_angle")
                q.note = (q.note + "; " if q.note else "") + "cannot centre the head"
                markers.append((a0, "Red", "BAD ANGLE - fix", "cannot centre the head"))
            out.append(q)
            prev = q
    return out, markers


# ------------------------------------------------------------------ detector + host
def make_detector(cfg: dict, host=None):
    m = os.path.expanduser(cfg["models"]["yunet"])
    if not (m and os.path.isfile(m)):
        return None
    try:
        import cv2  # noqa: F401  (only checking that it is installed)
    except ImportError:
        return None
    sface = os.path.expanduser(cfg["models"]["sface"])
    return fc.YuNet(m, sface if sface and os.path.isfile(sface) else "", host)


def _samples_for(short: Short, ranges: list[tuple[float, float]], det, cfg: dict, log) -> list[fc.Sample]:
    cache = os.path.join(short.work, "faces.npz")
    if os.path.isfile(cache):
        cached_ranges, samples = fc.load_samples(cache)
        if all(any(a <= lo + 0.01 and hi <= b + 0.01 for a, b in cached_ranges) for lo, hi in ranges):
            return samples
    log(f"  {short.name}: looking for faces ({sum(b - a for a, b in ranges):.0f}s of footage)")
    samples = fc.sample_faces(short.source, ranges, cfg["angles"]["sample_fps"], short.width, short.height, det)
    os.makedirs(short.work, exist_ok=True)
    fc.save_samples(cache, ranges, samples)
    return samples


def run(plan, cfg: dict, shorts=None, log=print, detector=None, **_):
    a = cfg["angles"]
    sel = [s for s in plan.shorts if (not shorts or s.name in shorts) and s.enabled]
    det = detector or make_detector(cfg)
    if det is None:
        log("  no face detector (set models.yunet and pip install opencv-python): cutting on speaker changes only")
    prepared = []
    for s in sel:
        s.clear_from("angles")
        an = analysis.load(s, cfg, log)
        sents, T = speaker_turns(s, an, cfg, log)
        pieces = s.pieces_before("angles")
        if sents:
            pieces = split_at(pieces, turn_cuts(sents, an.words, T, an.e, s.fps, a["cut_lead_frames"]),
                              s.fps, a["min_cut_gap_frames"])
        label_pieces(pieces, T, s.fps)
        s.stats["turns"] = [[round(t.s, 2), round(t.e, 2), t.spk] for t in T]
        prepared.append((s, pieces, T))
    samples_by = {}
    if det is not None:
        for s, pieces, T in prepared:
            samples_by[s.name] = _samples_for(s, [(p.src_in, p.src_out) for p in pieces], det, cfg, log)
        host = _host_vector(prepared, samples_by, det, cfg, log)
        if host is not None:
            for smp in samples_by.values():
                for sm in smp:
                    for f in sm.faces:
                        f.sim = float(f.emb @ host) if f.emb is not None else None
    for s, pieces, T in prepared:
        c = canvas(cfg, plan.timeline)
        if s.name in samples_by:
            smp = samples_by[s.name]
            fc.build_tracks(smp, s.width)
            roles = fc.assign_roles(smp, T)
            fc.label_samples(smp, roles)
            s.pieces, marks = frame_pieces(pieces, smp, s.width, s.height, cfg, c)
            s.stats["roles"] = {k: v for k, v in roles.items()}
        else:
            z0 = fr.steps(s.width, s.height, c, a["scale_steps"])[0]
            for p in pieces:
                p.zoom = z0
            s.pieces, marks = pieces, []
            s.note("[angles] no face detector: centre crop, only cut on speaker changes")
        for t_src, color, name, note in marks:
            t = s.src_to_tl(t_src)
            if t is not None:
                s.markers.append(Marker(t, color, name, note, stage="angles"))
        s.snapshot("angles")
        bad = sum("bad_angle" in p.flags for p in s.pieces)
        s.stats["angles"] = {"pieces": len(s.pieces), "bad": bad, "turns": len(T),
                             "zoom": sorted({p.zoom for p in s.pieces})}
        log(f"  {s.name}: {len(T)} turns, {len(s.pieces)} pieces, zoom {s.stats['angles']['zoom']}"
            + (f", {bad} bad angle(s) marked" if bad else ""))


def _host_vector(prepared, samples_by, det, cfg: dict, log):
    a = cfg["angles"]
    if getattr(det, "rec", None) is None:
        return None
    photo = a.get("host_photo", "")
    if photo and os.path.isfile(os.path.expanduser(photo)):
        return fc.host_from_photo(det, photo)
    emb = {}
    for name, smp in samples_by.items():
        e = [f.emb for sm in smp[::5] for f in sm.faces if f.emb is not None and f.w > 120]
        if e:
            emb[name] = np.vstack(e)
    c = fc.host_centroid(emb, a["host_min_shorts"])
    if c is None:
        log("  host not recognised across shorts: using who-talks-when to tell host from guest")
    return c
