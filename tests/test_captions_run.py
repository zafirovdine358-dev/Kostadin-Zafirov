import json
import os

from autoedit import captions as cp, config
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_speechy_clip


def build(tmp_path, graphic=False):
    f = make_speechy_clip(str(tmp_path / "G1.mp4"), [(1, 3), (9, 11)], 13.0)
    work = tmp_path / "w" / "G1"
    work.mkdir(parents=True)
    words = [("Are", 1.0, 1.2), ("you", 1.25, 1.4), ("following", 1.45, 1.9), ("base", 1.95, 2.3), ("on", 2.35, 2.5),
             ("TikTok?", 2.55, 3.0), ("Yes", 9.0, 9.3), ("I", 9.35, 9.45), ("do", 9.5, 9.8)]
    (work / "words.json").write_text(json.dumps({"engine": "whisper", "words": [
        {"s": s, "e": e, "t": t, "p": 1.0, "src": "a", "spk": ""} for t, s, e in words]}))
    s = Short("G1", f, fps=30.0, duration=13.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(work), pieces=[Piece(0.5, 4.0), Piece(8.5, 10.5)])
    s.stats["turns"] = [[1.0, 3.0, "H"], [9.0, 9.8, "G"]]
    cfg = config.merge(config.DEFAULTS, {})
    cfg["paths"]["work"] = str(tmp_path / "work")
    if graphic:
        from PIL import Image
        g = tmp_path / "based.png"
        Image.new("RGBA", (54, 96), (255, 215, 0, 255)).save(g)
        cfg["paths"]["based_graphic"] = str(g)
    return EditPlan(shorts=[s], timeline={"width": 108, "height": 192, "fps": 29.97}), s, cfg


def test_captions_map_through_the_cuts(tmp_path):
    plan, s, cfg = build(tmp_path)
    cp.run(plan, cfg, log=lambda *_: None)
    caps = s.stats["caption_list"]
    assert [c[2] for c in caps] == ["are you following", "based", "on tiktok?", "yes I do"]
    # second piece starts at timeline 3.5 s; 'Yes' was 0.5 s into it
    assert abs(caps[3][0] - 4.0) < 0.15
    assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(caps, caps[1:]))        # back to back
    assert os.path.isfile(s.subtitles) and "based" in open(s.subtitles).read()
    assert any("no BASED graphic" in n for n in s.notes)


def test_based_graphic_replaces_the_caption(tmp_path):
    plan, s, cfg = build(tmp_path, graphic=True)
    cp.run(plan, cfg, log=lambda *_: None)
    assert "based" not in open(s.subtitles).read()
    ov = [o for o in s.overlays if o.kind == "based"]
    assert len(ov) == 1 and ov[0].track == cfg["tracks"]["based"] and ov[0].t_out > ov[0].t_in
    assert ov[0].file.endswith("based_master.mov") and os.path.isfile(ov[0].file)
    # rerunning does not duplicate
    cp.run(plan, cfg, log=lambda *_: None)
    assert len([o for o in s.overlays if o.kind == "based"]) == 1
