"""Animated product overlays baked to transparent video (Resolve has no keyframe scripting)."""
import math
import os
from dataclasses import dataclass

import numpy as np

from . import media


@dataclass
class Anim:
    x: float                      # resting centre on the canvas, px
    y: float
    width: float                  # px
    rot: float = 0.0              # degrees at rest, positive = clockwise
    dur: float = 3.0
    enter: float = 0.5
    exit: float = 0.4
    from_dy: float = 0.0          # starts this far below its resting place (px)
    to_dy: float = 0.0            # leaves by dropping this far
    fade_in: float = 0.25         # 0 = fully opaque from the first frame
    fade_out: bool = True
    wiggle: float = 2.0           # degrees
    shadow: bool = True
    follow: list | None = None    # [(t, dx, dy)] extra offsets (px) so the sprite tucks behind a moving person


def ease_out_back(u: float) -> float:
    c1, c3 = 1.4, 2.4
    return 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2


def ease_in(u: float) -> float:
    return u * u


def state(a: Anim, t: float) -> tuple[float, float, float, float]:
    """(centre x, centre y, rotation, opacity) at time t."""
    u = min(max(t / a.enter, 0.0), 1.0) if a.enter > 0 else 1.0
    dy = a.from_dy * (1.0 - ease_out_back(u)) if a.from_dy else 0.0
    op = 1.0
    if a.fade_in > 0:
        op = min(op, t / a.fade_in)
    v = min(max((t - (a.dur - a.exit)) / a.exit, 0.0), 1.0) if a.exit > 0 else 0.0
    if v > 0:
        dy += a.to_dy * ease_in(v)
        if a.fade_out:
            op = min(op, 1.0 - v)
    x, y = a.x, a.y + dy
    if a.follow:
        ts = [f[0] for f in a.follow]
        x += float(np.interp(t, ts, [f[1] for f in a.follow]))
        y += float(np.interp(t, ts, [f[2] for f in a.follow]))
    rot = a.rot + a.wiggle * math.sin(2 * math.pi * 0.9 * t) * min(1.0, t / 0.6)
    return x, y, rot, max(0.0, min(1.0, op))


def load_sprite(path: str, width: float, shadow: bool):
    from PIL import Image, ImageFilter
    im = Image.open(path).convert("RGBA")
    h = max(1, int(round(im.height * width / im.width)))
    im = im.resize((max(1, int(round(width))), h), Image.LANCZOS)
    if not shadow:
        return im
    pad = 40
    base = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 255), (pad, pad), im.getchannel("A"))
    sh = sh.filter(ImageFilter.GaussianBlur(18))
    a = sh.getchannel("A").point(lambda v: int(v * 0.6))
    sh.putalpha(a)
    base = Image.alpha_composite(base, sh)
    base.alpha_composite(im, (pad, pad))
    return base


def paste_rgba(canvas, sprite, cx: float, cy: float) -> None:
    """alpha_composite that tolerates sprites hanging off the canvas."""
    x0, y0 = int(round(cx - sprite.width / 2)), int(round(cy - sprite.height / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(sprite.width, canvas.width - x0), min(sprite.height, canvas.height - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    canvas.alpha_composite(sprite.crop((sx0, sy0, sx1, sy1)), (x0 + sx0, y0 + sy0))


def frames(path: str, a: Anim, width: int, height: int, fps: float):
    """Raw RGBA bytes for each frame of the animation."""
    from PIL import Image
    sprite = load_sprite(path, a.width, a.shadow)
    n = max(1, int(round(a.dur * fps)))
    for k in range(n):
        t = k / fps
        cx, cy, rot, op = state(a, t)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if op > 0.003:
            sp = sprite.rotate(-rot, resample=Image.BICUBIC, expand=True) if abs(rot) > 0.05 else sprite
            if op < 0.997:
                alpha = sp.getchannel("A").point(lambda v, o=op: int(v * o))
                sp = sp.copy()
                sp.putalpha(alpha)
            paste_rgba(canvas, sp, cx, cy)
        yield canvas.tobytes()


def bake(path: str, a: Anim, width: int, height: int, fps: float, out: str) -> str:
    return media.encode_rgba(frames(path, a, width, height, fps), width, height, fps, out)


def still_master(img: str, needed_s: float, tl: dict, out_dir: str, fit: bool = True) -> str:
    """One transparent movie of a PNG, long enough for every use. Overlays take [0, n) frames of it, so a stamp
    for twenty shorts (or the BASED graphic for fifty captions) is encoded once, not once per use."""
    fps = tl["fps"]
    secs = 5.0 * math.ceil((max(needed_s, 0.1) + 0.5) / 5.0)
    name = os.path.splitext(os.path.basename(img))[0].replace(" ", "_")
    out = os.path.join(out_dir, f"{name}_master.mov")
    return media.still_to_video(img, secs, fps, int(tl["width"]), int(tl["height"]), out, fit)
