import numpy as np

from autoedit import angles, config, faces as fc, turns as tn
from autoedit.analysis import Word
from autoedit.plan import Piece

CFG = config.merge(config.DEFAULTS, {})
C = angles.canvas(CFG)
SW, SH = 3840, 2160


def face(cx, ey=700, w=210):
    return fc.Face(cx - w / 2, ey - 0.4 * w, w, 1.25 * w, cx, ey)


def samples(spec, step=0.2):
    """spec: [(t0, t1, H_cx|None, G_cx|None)] -> labelled samples."""
    out = []
    for t0, t1, h, g in spec:
        for t in np.arange(t0, t1, step):
            s = fc.Sample(round(float(t), 3))
            s.H = face(h(t) if callable(h) else h) if h is not None else None
            s.G = face(g(t) if callable(g) else g) if g is not None else None
            out.append(s)
    return out


def test_split_at_skips_cuts_near_edges():
    ps = angles.split_at([Piece(0, 10)], [0.05, 4.0, 9.98], 30.0, 3)
    assert [(p.src_in, p.src_out) for p in ps] == [(0, 4.0), (4.0, 10)]


def test_turn_cuts_land_just_before_first_word():
    ws = [Word(1.0, 1.4, "Hi?"), Word(3.0, 3.5, "Yes.")]
    ss = tn.sentences(ws)
    e = np.full(300, -45.0)
    e[50:70] = 0.0
    e[150:175] = 0.0                       # 3.0 - 3.5 s
    T = [tn.Turn(1.0, 1.4, "H"), tn.Turn(3.0, 3.5, "G")]
    cuts = angles.turn_cuts(ss, ws, T, e, 30.0, 2)
    assert len(cuts) == 1 and 2.9 < cuts[0] < 3.0


def test_frame_pieces_centres_each_speaker_with_one_zoom():
    pcs = [Piece(0, 4, "H"), Piece(4, 8, "G")]
    smp = samples([(0, 8, 1200, 2700)])
    out, marks = angles.frame_pieces(pcs, smp, SW, SH, CFG, C)
    assert [p.who for p in out] == ["H", "G"] and len({p.zoom for p in out}) == 1
    for p, u in zip(out, (1200, 2700)):
        px = p.pan + 540
        z = p.zoom
        assert abs(px + (u - SW / 2) * z - 540) < 2          # head centred horizontally
        assert not p.flags and not marks


def test_zoom_goes_up_when_faces_are_small():
    smp = samples([(0, 8, 1200, 2700)])
    for s in smp:
        s.H, s.G = face(1200, w=120), face(2700, w=120)
    out, _ = angles.frame_pieces([Piece(0, 4, "H"), Piece(4, 8, "G")], smp, SW, SH, CFG, C)
    assert out[0].zoom == out[1].zoom and out[0].zoom >= 1.37


def test_missing_speaker_stays_on_the_other_person_and_bad_angle_marks():
    smp = samples([(0, 4, 1200, 2700), (4, 8, None, 2700), (8, 12, None, None)])
    out, marks = angles.frame_pieces([Piece(0, 4, "H"), Piece(4, 8, "H"), Piece(8, 12, "G")], smp, SW, SH, CFG, C)
    assert out[1].who == "G" and "stayed on the other person" in out[1].note
    assert out[2].who == "" and "bad_angle" in out[2].flags
    assert [m[2] for m in marks] == ["BAD ANGLE - fix"]


def test_moving_speaker_gets_steady_chunks_and_a_marker():
    drift = lambda t: 1200 + (0 if t < 4 else 400)
    smp = samples([(0, 8, drift, 2700)])
    out, marks = angles.frame_pieces([Piece(0, 8, "H")], smp, SW, SH, CFG, C)
    assert len(out) == 2 and all("stab" in p.flags for p in out)
    assert out[0].src_out == out[1].src_in and (out[0].src_in, out[1].src_out) == (0, 8)
    assert out[1].pan < out[0].pan                   # head moved right in the source -> picture moves left
    assert marks[0][2] == "STABILISED"


def test_samples_roundtrip(tmp_path):
    smp = samples([(0, 1, 1200, 2700)])
    smp[0].faces = [face(1200)]
    smp[0].faces[0].emb = np.arange(4, dtype=float)
    smp[1].faces = [face(1300)]
    smp[1].faces[0].emb = np.ones(4)
    p = str(tmp_path / "f.npz")
    fc.save_samples(p, [(0.0, 1.0)], smp)
    rng, back = fc.load_samples(p)
    assert rng == [(0.0, 1.0)] and len(back) == len(smp)
    assert back[0].faces[0].cx == 1200 and list(back[1].faces[0].emb) == [1, 1, 1, 1] and back[2].faces == []
