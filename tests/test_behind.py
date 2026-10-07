import numpy as np
import pytest

from autoedit import behind, config, media
from autoedit.plan import EditPlan, Overlay, Piece, Short
from conftest import make_speechy_clip

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")
from PIL import Image

TL = {"width": 270, "height": 480, "fps": 30.0}
CFG = config.merge(config.DEFAULTS, {})


def ellipse_matte(frame):
    H, W = frame.shape[:2]
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (W // 2, int(H * 0.5)), (int(W * 0.22), int(H * 0.35)), 0, 0, 360, 1.0, -1)
    return m


def empty_matte(frame):
    return np.zeros(frame.shape[:2], np.float32)


@pytest.fixture
def short(tmp_path):
    img = tmp_path / "prod.png"
    im = Image.new("RGBA", (400, 800), (0, 0, 0, 0))
    im.paste((220, 30, 30, 255), (20, 20, 380, 780))
    im.save(img)
    f = make_speechy_clip(str(tmp_path / "G1.mp4"), [(1, 9)], 10.0, w=960, h=540)
    s = Short("G1", f, fps=30.0, width=960, height=540, duration=10.0, work=str(tmp_path / "w"),
              pieces=[Piece(0, 4.0, "H", "H", 1.0, 0.0, 0.0), Piece(4.0, 7.0, "H", "H", 1.0, 0.0, 0.0),
                      Piece(7.0, 10.0, "G", "G", 1.0, 0.0, 0.0)])
    s.snapshot("fine")
    s.overlays.append(Overlay("product", str(img), 1.4, 4.4, 4,
                              {"name": "pomade", "image": str(img), "scale": 60, "mention": 1.5}, "pomade", "products"))
    s.overlays.append(Overlay("product", str(img), 6.4, 9.4, 4,
                              {"name": "hair clay", "image": str(img), "scale": 60, "mention": 6.5}, "hair clay", "products"))
    return EditPlan(shorts=[s], timeline=TL), s


def test_stay_end_follows_who_across_jump_cuts():
    s = Short("G1", "x", pieces=[Piece(0, 3, who="H"), Piece(3, 6, who="H"), Piece(6, 9, who="G"), Piece(9, 10, who="")])
    assert behind.stay_end(s, 1.0) == ("H", 6.0)
    assert behind.stay_end(s, 7.0) == ("G", 9.0)
    assert behind.stay_end(s, 9.5) == ("", 9.5)


def test_windows_need_two_seconds_on_the_same_person(short):
    plan, s = short
    ws = behind.windows(s, CFG, 30.0)
    assert [w.ov.props["name"] for w in ws] == ["pomade"]          # hair clay: guest shot ends 9.0-9.4? stays 2.5 s
    s.overlays[1].props["mention"] = 8.2                              # only 1.8 s of shot left
    assert [w.ov.props["name"] for w in behind.windows(s, CFG, 30.0)] == ["pomade"]
    s.pieces[2] = Piece(7.0, 8.5, "G", "G")
    assert behind.windows(s, CFG, 30.0)[0].ov.props["name"] == "pomade"


def test_run_bakes_three_layers_that_stack_correctly(short):
    plan, s = short
    s.overlays = s.overlays[:1]
    behind.run(plan, CFG, log=lambda *_: None, matte=ellipse_matte)
    kinds = sorted(o.kind for o in s.overlays)
    assert kinds == ["behind_person", "behind_product"] and s.stats["behind"] == 1
    prod = next(o for o in s.overlays if o.kind == "behind_product")
    person = next(o for o in s.overlays if o.kind == "behind_person")
    assert prod.track == CFG["tracks"]["behind_product"] and person.track == CFG["tracks"]["behind_person"]
    n = round((prod.t_out - prod.t_in) * 30)
    assert person.t_in == prod.t_in and n >= 60
    # pieces inside the window now use the baked, already framed picture
    baked = [p for p in s.pieces if p.media]
    assert baked and all(p.zoom == 1.0 and p.pan == 0.0 and "behind" in p.flags for p in baked)
    assert abs(sum(p.dur for p in s.pieces) - 10.0) < 1e-6
    assert round(baked[0].media_in, 3) == 0.0
    info = media.probe(baked[0].media)
    assert (info.width, info.height) == (270, 480) and abs(info.duration * 30 - n) < 1.5
    # decode the layers and check the stacking
    def decode(path):
        raw = media.run([media.FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, 480, 270, 4)
    P, V = decode(person.file), decode(prod.file)
    assert len(P) == n == len(V)
    mid = n // 2
    assert P[mid][240, 135, 3] > 240 and P[mid][5, 5, 3] == 0              # person opaque in the middle, nothing at the corner
    assert V[0][..., 3].max() == 0                                           # product starts fully hidden (below the frame)
    assert V[mid][..., 3].max() > 200                                        # and has come up beside the head by now
    ys, xs = np.nonzero(V[mid][..., 3] > 200)
    assert abs(xs.mean() - 135) > 40                                         # sits to one side of the head
    assert ((V[mid][..., 3] > 200) & (P[mid][..., 3] > 240)).any()           # and tucks behind the person
    # rerun restores the slide-in first, so nothing doubles up
    behind.run(plan, CFG, log=lambda *_: None, matte=ellipse_matte)
    assert sorted(o.kind for o in s.overlays) == ["behind_person", "behind_product"]


def test_person_who_leaves_keeps_the_slide_in(short):
    plan, s = short
    s.overlays = s.overlays[:1]
    behind.run(plan, CFG, log=lambda *_: None, matte=empty_matte)
    assert [o.kind for o in s.overlays] == ["product"] and not any(p.media for p in s.pieces)


def test_no_matte_leaves_everything_as_slide_ins(short, monkeypatch):
    plan, s = short
    monkeypatch.setattr(behind, "make_matte", lambda: None)
    behind.run(plan, CFG, log=lambda *_: None)
    assert [o.kind for o in s.overlays] == ["product", "product"]
