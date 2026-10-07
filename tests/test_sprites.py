import numpy as np
import pytest

from autoedit import media, sprites

PIL = pytest.importorskip("PIL")
from PIL import Image


@pytest.fixture
def product_png(tmp_path):
    p = tmp_path / "prod.png"
    im = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
    im.paste((255, 40, 40, 255), (20, 20, 180, 380))
    im.save(p)
    return str(p)


def test_state_slides_in_holds_and_drops_out():
    a = sprites.Anim(540, 1480, 300, from_dy=800, to_dy=300, dur=3.0, enter=0.5, exit=0.4, wiggle=0)
    x0, y0, _, o0 = sprites.state(a, 0.0)
    assert y0 == 1480 + 800 and o0 == 0.0                       # starts far below, invisible
    assert sprites.state(a, 1.5) == (540, 1480, 0, 1.0)         # resting
    x, y, r, o = sprites.state(a, 2.99)
    assert y > 1480 + 250 and o < 0.2                           # dropped and faded
    # behind-style: opaque from the first frame, no fade
    b = sprites.Anim(300, 900, 300, from_dy=1440, fade_in=0, fade_out=False, wiggle=0)
    assert sprites.state(b, 0.0)[3] == 1.0 and sprites.state(b, 2.9)[3] == 1.0


def test_follow_offsets_move_the_sprite():
    a = sprites.Anim(500, 900, 100, wiggle=0, enter=0, exit=0, fade_in=0, follow=[(0, 0, 0), (2, 100, -50)])
    x, y, *_ = sprites.state(a, 1.0)
    assert (round(x), round(y)) == (550, 875)


def test_bake_writes_a_transparent_prores_with_the_product_in_the_middle(tmp_path, product_png):
    a = sprites.Anim(270, 500, 150, from_dy=450, dur=1.0, enter=0.4, exit=0.3, to_dy=100, wiggle=0, shadow=True)
    out = sprites.bake(product_png, a, 540, 960, 30.0, str(tmp_path / "p.mov"))
    info = media.probe(out)
    assert (info.width, info.height) == (540, 960) and abs(info.duration - 1.0) < 0.1
    raw = media.run([media.FFMPEG, "-v", "error", "-i", out, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 960, 540, 4)
    assert len(fr) == 30
    assert fr[0][..., 3].max() == 0                       # nothing on screen at the first frame
    mid = fr[18]
    assert mid[500, 270, 3] > 200 and mid[500, 270, 0] > 200 and mid[500, 270, 1] < 120     # opaque red body in the middle
    assert mid[50, 270, 3] == 0                                                             # nothing at the top
    assert fr[-1][..., 3].max() < 40                      # faded out at the end
