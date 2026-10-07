import json
import os

import pytest

from autoedit import captions as cp, config, media, products as pr
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_speechy_clip

PIL = pytest.importorskip("PIL")
from PIL import Image

CFG = config.merge(config.DEFAULTS, {})
ITEMS = CFG["products"]["items"]


def W(text, t0=0.0, step=0.4):
    return [cp.TWord(w, t0 + i * step, t0 + i * step + 0.3) for i, w in enumerate(text.split())]


def test_first_mention_only_and_asr_repairs():
    ws = W("try the sea salt spray and the sea salt spray again") + W("the coke cream is great", t0=60)
    ms = pr.find_mentions(ws, ITEMS, 30.0)
    assert [(m.name, round(m.t, 1)) for m in ms] == [("sea salt spray", 0.8), ("curl cream", 60.4)]
    again = pr.find_mentions(W("sea salt spray") + W("sea salt spray", t0=40), ITEMS, 30.0)
    assert len(again) == 2                                   # comes back much later: shown again


def test_longest_phrase_wins():
    ms = pr.find_mentions(W("the skin revival spray is great"), ITEMS, 30.0)
    assert [m.name for m in ms] == ["skin revival spray"]
    ms = pr.find_mentions(W("leave-in conditioner and shampoo"), ITEMS, 30.0)
    assert [m.name for m in ms] == ["leave-in conditioner", "shampoo"]


def test_find_image_is_forgiving(tmp_path):
    Image.new("RGBA", (10, 10)).save(tmp_path / "Pomade.PNG")
    Image.new("RGBA", (10, 10)).save(tmp_path / "Curl Mousse (from webp).png")
    assert pr.find_image(str(tmp_path), "pomade.png").endswith("Pomade.PNG")
    assert pr.find_image(str(tmp_path), "Curl Mousse.webp").endswith("(from webp).png") is False  # different stem
    assert pr.find_image(str(tmp_path), "Curl Mousse (from webp).webp").endswith("(from webp).png")
    assert pr.find_image(str(tmp_path), "nope.png") == ""


def test_run_plans_and_bakes_slide_ins(tmp_path):
    prod = tmp_path / "BASED Products"
    prod.mkdir()
    im = Image.new("RGBA", (800, 1600), (0, 0, 0, 0))
    im.paste((200, 30, 30, 255), (100, 100, 700, 1500))
    im.save(prod / "SSS SANTAL TILTED V03 1.png")
    f = make_speechy_clip(str(tmp_path / "G1.mp4"), [(1, 6)], 8.0)
    work = tmp_path / "w" / "G1"
    work.mkdir(parents=True)
    ws = [("Try", 1.0, 1.2), ("the", 1.25, 1.4), ("sea", 1.5, 1.7), ("salt", 1.75, 1.95), ("spray", 2.0, 2.4)]
    (work / "words.json").write_text(json.dumps({"engine": "whisper", "words": [
        {"s": s, "e": e, "t": t, "p": 1.0, "src": "a", "spk": ""} for t, s, e in ws]}))
    s = Short("G1", f, fps=30.0, duration=8.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(work), pieces=[Piece(0.0, 8.0)])
    cfg = config.merge(CFG, {"paths": {"products": str(prod)}})
    plan = EditPlan(shorts=[s], timeline={"width": 270, "height": 480, "fps": 30.0})
    pr.run(plan, cfg, log=lambda *_: None)
    ov = s.overlays
    assert [o.kind for o in ov] == ["product"] and ov[0].props["name"] == "sea salt spray"
    assert ov[0].file.endswith(".mov") and os.path.isfile(ov[0].file) and ov[0].track == cfg["tracks"]["product"]
    assert 1.3 < ov[0].t_in < 1.6 and abs((ov[0].t_out - ov[0].t_in) - 3.0) < 1e-6
    info = media.probe(ov[0].file)
    assert (info.width, info.height) == (270, 480) and abs(info.duration - 3.0) < 0.15
    pr.run(plan, cfg, log=lambda *_: None)                         # idempotent
    assert len(s.overlays) == 1


def test_missing_image_is_noted_not_fatal(tmp_path):
    s = Short("G1", "x.mp4", duration=8.0, pieces=[Piece(0.0, 8.0)], work=str(tmp_path))
    ov = pr.plan(s, [pr.Mention("pomade", 1.0)], CFG, str(tmp_path), log=lambda *_: None)
    assert ov == [] and any("no image for 'pomade'" in n for n in s.notes)


def test_later_product_takes_over_the_slot():
    s = Short("G1", "x.mp4", duration=20.0, pieces=[Piece(0.0, 20.0)])
    import tempfile
    d = tempfile.mkdtemp()
    for n in ("Pomade.png", "Hair Clay.png"):
        Image.new("RGBA", (10, 10)).save(os.path.join(d, n))
    ov = pr.plan(s, [pr.Mention("pomade", 1.0), pr.Mention("hair clay", 2.5)], CFG, d)
    assert [(o.props["name"], round(o.t_out, 2)) for o in ov] == [("pomade", 2.4), ("hair clay", 5.4)]
