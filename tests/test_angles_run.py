import numpy as np

from autoedit import analysis, angles, config, faces as fc
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_speechy_clip


class FakeDetector:
    """Host on the left, guest on the right (coordinates in the 1920-wide frame, like YuNet returns them)."""
    rec = object()

    def faces(self, img, to_src=1.0):
        h = fc.Face(400 * to_src, 300 * to_src, 110 * to_src, 135 * to_src, 455 * to_src, 340 * to_src, 0.9, 0.7)
        g = fc.Face(1300 * to_src, 300 * to_src, 110 * to_src, 135 * to_src, 1355 * to_src, 340 * to_src, 0.9, 0.1)
        return [h, g]


def test_angles_run_two_mics_cuts_on_turns_and_frames_speakers(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "have_whisper", lambda: False)
    f = make_speechy_clip(str(tmp_path / "G1.mp4"), [[(1, 3), (7, 9)], [(4, 6), (10, 12)]], 13.0, audio_hz=(220, 330),
                          w=320, h=180)
    s = Short("G1", f, fps=30.0, width=3840, height=2160, duration=13.0,
              audio={"mic": "two", "a1": [0, 0], "a2": [0, 1], "empty": []}, work=str(tmp_path / "w" / "G1"),
              pieces=[Piece(0, 13.0)])
    plan = EditPlan(shorts=[s], timeline={"width": 1080, "height": 1920, "fps": 29.97})
    cfg = config.merge(config.DEFAULTS, {})

    def fake_grab(path, start, dur, fps, w, h):
        n = int(dur * fps)
        for i in range(n):
            yield start + i / fps, np.zeros((h, w, 3), np.uint8)
    monkeypatch.setattr(fc.media, "grab_frames", fake_grab)
    angles.run(plan, cfg, log=lambda *_: None, detector=FakeDetector())
    assert [p.spk for p in s.pieces] == ["G", "H", "G", "H"]
    assert [p.who for p in s.pieces] == ["G", "H", "G", "H"]
    assert [round(p.src_in, 1) for p in s.pieces] == [0.0, 3.9, 6.9, 9.9]
    pan = {p.spk: p.pan for p in s.pieces}
    assert pan["H"] > pan["G"]                       # host is on the left: picture moves right to centre him
    assert len({p.zoom for p in s.pieces}) == 1 and not any(p.flags for p in s.pieces)
    assert "angles" in s.snaps and s.stats["angles"]["turns"] == 4
    assert (tmp_path / "w" / "G1" / "faces.npz").exists()
    # second run reuses the face cache (no grabbing)
    monkeypatch.setattr(fc.media, "grab_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss")))
    angles.run(plan, cfg, log=lambda *_: None, detector=FakeDetector())
