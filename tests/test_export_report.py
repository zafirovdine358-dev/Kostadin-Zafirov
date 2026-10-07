import os

from autoedit import export, installer, report
from autoedit.plan import EditPlan, Marker, Overlay, Piece, Short


def sample():
    s = Short("G1291", "/footage/G1291.MP4", fps=29.97, duration=90.0,
              pieces=[Piece(2.0, 5.0, note="hook"), Piece(10.0, 12.0)])
    s.stats["start_tc"] = "01:00:00:00"
    s.stats["fine"] = {"before": 90, "after": 5, "cuts": {"pause": 4, "echo": 1}, "hook": "Are you following BASED?"}
    s.stats["captions"] = {"count": 12, "based": 2, "brands": 0}
    s.stats["caption_list"] = [[0.0, 1.0, "are you following", "text"], [1.0, 1.5, "based", "based"]]
    s.stats["products"] = [["pomade", 1.0, 4.0]]
    s.stats["stamp"] = {"start": 3.0, "end": 5.0, "on_cut": True}
    s.stats["music"] = "ES_Song.wav"
    s.stats["audio"] = {"speech_before": -25.0, "speech_after": -18.0, "peak_after": -3.1, "gap_after": -20.0}
    s.markers = [Marker(1.0, "Yellow", "CHECK showing?", "silent 3s", "rough")]
    s.notes = ["[fine] guest may not follow BASED ('no'): check this can be a short"]
    s.overlays = [Overlay("based", "g.mov", 1.0, 1.5, 7, {}, "BASED", "captions")]
    return EditPlan(batch={"name": "BATCH 4", "date": "2026-10-05", "host": "Lian"}, timeline={"fps": 29.97},
                    shorts=[s], stages=["ingest", "rough", "fine"])


def test_edl_uses_the_clips_own_timecode_and_continuous_record_time():
    plan = sample()
    edl = export.edl(plan.shorts[0], 29.97)
    lines = edl.splitlines()
    assert lines[0] == "TITLE: G1291" and lines[1] == "FCM: NON-DROP FRAME"
    ev = [l for l in lines if l[:3].isdigit()]
    assert len(ev) == 2
    a = ev[0].split()
    assert a[:4] == ["001", "G1291", "AA/V", "C"] and a[4] == "01:00:02:00"
    assert a[5] == "01:00:05:00" and a[6] == "00:00:00:00" and a[7] == "00:00:03:00"
    b = ev[1].split()
    assert b[4] == "01:00:10:00" and b[6] == "00:00:03:00" and b[7] == "00:00:05:00"
    assert "* FROM CLIP NAME: G1291.MP4" in lines and "* COMMENT: hook" in lines


def test_write_all_skips_the_based_caption_when_a_graphic_covers_it(tmp_path):
    plan = sample()
    out = export.write_all(plan, str(tmp_path))
    assert sorted(os.path.basename(p) for p in out) == ["G1291.edl", "G1291.srt"]
    assert "based" not in open(tmp_path / "G1291.srt").read() and "are you following" in open(tmp_path / "G1291.srt").read()
    assert export.write_all(plan, str(tmp_path), shorts=["other"]) == []


def test_report_table_and_markdown():
    plan = sample()
    t = report.table(plan)
    assert "G1291" in t and "1:30.0 -> 5.0s" in t
    assert "silent stretch" in t and "guest may not follow BASED" in t
    md = report.markdown(plan)
    assert md.startswith("# BATCH 4") and "Hook: Are you following BASED?" in md
    assert "- Cut: 1 echo, 4 pause" in md
    assert "pomade at 1.0s" in md and "ES_Song.wav" in md and "your ears decide" in md
    assert "guest may not follow" in md


def test_installer_writes_a_launcher(tmp_path):
    msg = installer.install_resolve_script({"_path": "/home/me/autoedit.json"}, folder=str(tmp_path / "Utility"))
    f = tmp_path / "Utility" / "BASED Auto Edit.py"
    assert f.exists() and "run_from_menu" in f.read_text() and "/home/me/autoedit.json" in f.read_text()
    assert "Workspace > Scripts" in msg
    compile(f.read_text(), str(f), "exec")
