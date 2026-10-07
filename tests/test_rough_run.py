from autoedit import analysis, config, rough
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_speechy_clip


def build(tmp_path):
    f = make_speechy_clip(str(tmp_path / "a" / "G1.mp4"), [(1.0, 3.0), (6.5, 8.5)], 11.0)
    s = Short("G1", f, fps=30.0, duration=11.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(tmp_path / "w" / "G1"), pieces=[Piece(0, 11.0)])
    return EditPlan(shorts=[s]), s


def test_rough_run_energy_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "have_whisper", lambda: False)
    plan, s = build(tmp_path)
    cfg = config.merge(config.DEFAULTS, {})
    rough.run(plan, cfg, shorts=None, log=lambda *_: None, showing="off")
    assert s.stats["asr"] == "energy"
    # speech 1-3 and 6.5-8.5 with pads: roughly 2.15 + 2.4 kept, head/gap/tail removed
    assert 4.0 < s.tl_duration() < 5.6, s.pieces
    assert s.pieces[0].src_in < 1.0 <= s.pieces[0].src_out
    assert "rough" in s.snaps
    # rerun is idempotent
    d1 = s.tl_duration()
    rough.run(plan, cfg, log=lambda *_: None, showing="off")
    assert abs(s.tl_duration() - d1) < 1e-6


def test_rough_run_flag_keeps_long_gap_with_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "have_whisper", lambda: False)
    plan, s = build(tmp_path)
    cfg = config.merge(config.DEFAULTS, {})
    rough.run(plan, cfg, log=lambda *_: None, showing="flag")
    assert any(m.name == "CHECK showing?" for m in s.markers)
    assert any("phone" in p.flags for p in s.pieces)
    assert s.tl_duration() > 7.0
