import subprocess

from autoedit import broll, config, turns as tn
from autoedit.analysis import Word
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_speechy_clip


def routine(path, dur=14.0):
    """Clip that fades in from black over 1.5 s, flashes white at 9-9.5 s and fades out."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=d={dur}:s=144x256:r=30",
                    "-vf", f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,9,9.5)',"
                           f"fade=in:0:45,fade=out:st={dur - 1.5}:d=1.5", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", path], check=True)
    return path


def test_candidates_match_folder_or_file_names(tmp_path):
    (tmp_path / "David").mkdir()
    (tmp_path / "David" / "curl cream routine.mp4").write_bytes(b"x")
    (tmp_path / "Mia").mkdir()
    (tmp_path / "Mia" / "Curl_Cream_final.mov").write_bytes(b"x")
    (tmp_path / "Mia" / "texture powder.mp4").write_bytes(b"x")
    (tmp_path / "Mia" / "notes.txt").write_text("x")
    assert [p.split("/")[-1] for p in broll.candidates(str(tmp_path), "curl cream")] == ["curl cream routine.mp4", "Curl_Cream_final.mov"]
    assert len(broll.candidates(str(tmp_path), "curl cream", model="mia")) == 1
    assert broll.candidates(str(tmp_path), "pomade") == []


def test_clean_ranges_skip_fades_and_flashes(tmp_path):
    p = routine(str(tmp_path / "r.mp4"))
    clean = broll.clean_ranges(p, 14.0)
    assert clean and clean[0][0] >= 1.5 and clean[-1][1] <= 12.5
    assert not any(a < 9.4 and b > 9.1 for a, b in clean)            # the flash is excluded


def test_cut_plan_and_pick_spread_over_clean_footage():
    L = broll.cut_plan(6.0, 1.0, 2.0)
    assert len(L) == 4 and all(1.0 <= x <= 2.0 for x in L)
    assert broll.cut_plan(0.9, 1.0, 2.0) == [0.9]
    st = broll.pick([(1.5, 8.0), (10.0, 12.5)], L)
    assert len(st) == 4 and st == sorted(st) and all(b - a >= 1.0 for a, b in zip(st, st[1:]))
    assert all(any(a <= s and s + 1.5 <= b + 1e-9 for a, b in [(1.5, 8.0), (10.0, 12.5)]) for s in st)
    assert broll.pick([(0, 0.5)], [1.0]) == []


def test_snap_to_captions_keeps_cuts_long_enough():
    assert broll.snap_to_captions([0, 1.5, 3.0, 4.5], [1.7, 3.2, 9], 0.8) == [0, 1.7, 3.2, 4.5]
    assert broll.snap_to_captions([0, 1.5, 3.0], [0.4], 0.8) == [0, 1.5, 3.0]


def test_explanation_window_stops_at_speaker_change():
    ws = [Word(1.0, 1.4, "Curl"), Word(1.4, 1.8, "cream"), Word(1.8, 2.4, "is"), Word(2.4, 3.0, "thicker."),
          Word(3.1, 3.6, "It"), Word(3.6, 4.2, "holds."), Word(5.0, 5.4, "Okay.")]
    T = [tn.Turn(1.0, 4.2, "H"), tn.Turn(5.0, 5.4, "G")]
    assert broll.explanation_window(ws, T, 1.2, 8.0) == (1.0, 4.2)
    assert broll.explanation_window(ws, T, 1.2, 2.5)[1] == 3.0


def test_run_places_video_only_cuts(tmp_path, monkeypatch):
    import json
    folder = tmp_path / "B-roll" / "David"
    folder.mkdir(parents=True)
    routine(str(folder / "curl cream routine.mp4"))
    clip = make_speechy_clip(str(tmp_path / "G1.mp4"), [(1, 9)], 10.0)
    work = tmp_path / "w"
    work.mkdir()
    ws = [("The", 1.0, 1.1), ("curl", 1.2, 1.5), ("cream", 1.55, 1.9), ("is", 2.0, 2.1), ("lightweight.", 2.15, 3.0),
          ("You", 3.1, 3.3), ("just", 3.35, 3.6), ("scrunch", 3.65, 4.2), ("it", 4.25, 4.4), ("in.", 4.45, 5.0)]
    (work / "words.json").write_text(json.dumps({"engine": "whisper", "words": [
        {"s": s, "e": e, "t": t, "p": 1.0, "src": "a", "spk": ""} for t, s, e in ws]}))
    s = Short("G1", clip, fps=30.0, duration=10.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(work), pieces=[Piece(0, 10.0)])
    s.stats["turns"] = [[1.0, 5.0, "H"]]
    cfg = config.merge(config.DEFAULTS, {"paths": {"broll": str(tmp_path / "B-roll")}, "broll": {"enabled": True}})
    plan = EditPlan(shorts=[s], timeline={"fps": 29.97})
    broll.run(plan, cfg, log=lambda *_: None, product="curl cream")
    ov = [o for o in s.overlays if o.kind == "broll"]
    assert len(ov) >= 2 and abs(ov[0].t_in - 1.0) < 0.05 and abs(ov[-1].t_out - 5.0) < 0.05
    assert all(a.t_out == b.t_in for a, b in zip(ov, ov[1:]))
    assert all(1.0 <= o.t_out - o.t_in <= 2.5 for o in ov) and all(o.track == cfg["tracks"]["broll"] for o in ov)
    assert all(o.props["audio"] is False and o.props["scaling"] == 3 for o in ov)
    assert [o.props["src_in"] for o in ov] == sorted(o.props["src_in"] for o in ov)
    broll.run(plan, cfg, log=lambda *_: None, product="curl cream")        # rerun replaces, never stacks
    assert len([o for o in s.overlays if o.kind == "broll"]) == len(ov)
