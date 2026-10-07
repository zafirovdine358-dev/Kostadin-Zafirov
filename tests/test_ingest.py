import datetime as dt

from autoedit import config, ingest
from conftest import make_clip


def test_parse_batch_names():
    p = ingest.parse_batch_name
    assert p("29:09:26 Lian") == (dt.date(2026, 9, 29), "Lian")
    assert p("2026-10-01 Erik") == (dt.date(2026, 10, 1), "Erik")
    assert p("Multi-Interview") is None
    assert p("5.10.26")[0] == dt.date(2026, 10, 5)
    assert p("05/10/26 X", order="MDY")[0] == dt.date(2026, 5, 10)


def test_newest_batch_ignores_done_and_loose_files(tmp_path):
    root = tmp_path / "New Vids"
    for n in ["28-09-26 Lian", "03-10-26 Erik", "05-10-26 DONE", "Multi-Interview"]:
        (root / n).mkdir(parents=True)
    (root / "notes.txt").write_text("x")
    found = ingest.find_batches(str(root), "DMY", ["DONE", "Multi-Interview"])
    assert [b.host for b in found] == ["Erik", "Lian"]


def test_audio_channel_detection(tmp_path):
    cases = {"single": "single", "identical": "single", "two": "two", "quad": "two"}
    for kind, want in cases.items():
        f = make_clip(str(tmp_path / f"{kind}.mp4"), dur=1.5, audio=kind)
        info = ingest.media.probe(f)
        a = ingest.analyze_audio(f, info)
        assert a["mic"] == want, (kind, a)
    quad = ingest.analyze_audio(str(tmp_path / "quad.mp4"), ingest.media.probe(str(tmp_path / "quad.mp4")))
    assert len(quad["empty"]) == 2 and quad["a2"] is not None


def test_ingest_builds_plan(tmp_path):
    cfg = config.merge(config.DEFAULTS, {"paths": {"based_root": str(tmp_path), "work": str(tmp_path / "work"),
                                                   "footage": "New Vids"}})
    d = tmp_path / "New Vids" / "29-09-26 Lian"
    make_clip(str(d / "G1292.mp4"), dur=1.5)
    make_clip(str(d / "G1291.mp4"), dur=1.5, audio="two")
    plan, work = ingest.ingest(cfg, log=lambda *_: None)
    assert [s.name for s in plan.shorts] == ["G1291", "G1292"]
    assert plan.batch["host"] == "Lian" and work.endswith("2026-09-29 Lian")
    assert plan.shorts[0].audio["mic"] == "two"
    assert plan.shorts[0].pieces[0].src_out > 1.4


def test_clip_without_sound_is_left_out(tmp_path):
    cfg = config.merge(config.DEFAULTS, {"paths": {"based_root": str(tmp_path), "work": str(tmp_path / "work"),
                                                   "footage": "New Vids"}})
    d = tmp_path / "New Vids" / "29-09-26 Lian"
    make_clip(str(d / "G1.mp4"), dur=1.5, audio="none")
    make_clip(str(d / "G2.mp4"), dur=1.5)
    plan, _ = ingest.ingest(cfg, log=lambda *_: None)
    assert [s.enabled for s in plan.shorts] == [False, True]
    assert "no audible audio" in plan.shorts[0].notes[0]
