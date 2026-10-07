"""Products that slide in from behind the person (port of based-products-behind).

Resolve has no masks in its scripting API, so the layers are baked: the framed footage (V1), the product sliding
in (V2) and the footage again with the person cut out (V3, transparent everywhere else). Stacked, the product
appears from behind the speaker. V1 is baked too, so the three layers match pixel for pixel.
"""
import os
from dataclasses import asdict, dataclass

import numpy as np

from . import media, products as pr, sprites
from .plan import Overlay, Piece

PRESENT_FRAC = 0.02          # the person fills at least this much of the frame
GONE_S = 0.6                 # absent for this long = the shot left them


@dataclass
class Window:
    ov: Overlay
    who: str
    f_a: int
    f_b: int


def stay_end(short, t0: float) -> tuple[str, float]:
    """Who is on screen at t0, and until when does the picture stay on them (jump cuts on them count)."""
    starts = short.piece_starts()
    for k, p in enumerate(short.pieces):
        if starts[k] <= t0 < starts[k] + p.dur:
            if not p.who:
                return "", t0
            end = starts[k] + p.dur
            for q in short.pieces[k + 1:]:
                if q.who != p.who:
                    break
                end += q.dur
            return p.who, end
    return "", t0


def make_matte():
    """Person matte: MediaPipe selfie segmentation when installed."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        return None
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def matte(bgr: np.ndarray) -> np.ndarray:
        return seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
    return matte


def iter_framed(short, f_a: int, n: int, tl: dict):
    """BGR frames [H, W, 3] of the framed picture for timeline frames f_a .. f_a+n-1, piece by piece."""
    fps, W, H = tl["fps"], int(tl["width"]), int(tl["height"])
    bounds = short.frame_bounds(fps)
    for k, p in enumerate(short.pieces):
        lo, hi = max(f_a, bounds[k]), min(f_a + n, bounds[k + 1])
        if hi <= lo:
            continue
        yield from media.grab_framed(short.source, p.src_in + (lo - bounds[k]) / fps, hi - lo, fps,
                                     short.width, short.height, W, H, p.zoom, p.pan, p.tilt)


def person_masks(short, f_a: int, n: int, tl: dict, matte) -> list[np.ndarray]:
    """Half-resolution matte per frame (uint8), lightly smoothed over time."""
    import cv2
    W, H = int(tl["width"]), int(tl["height"])
    out, prev = [], None
    for fr in iter_framed(short, f_a, n, tl):
        m = np.asarray(matte(fr), dtype=np.float32)
        m = cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        if prev is not None:
            m = 0.65 * m + 0.35 * prev
        prev = m
        out.append(np.clip(m * 255, 0, 255).astype(np.uint8))
    return out


def present_until(masks: list[np.ndarray], fps: float) -> int:
    """First frame where the person has been gone for GONE_S seconds (or all frames)."""
    gone = int(GONE_S * fps)
    run = 0
    for i, m in enumerate(masks):
        if (m > 128).mean() < PRESENT_FRAC:
            run += 1
            if run >= gone:
                return i - run + 1
        else:
            run = 0
    return len(masks)


def head_track(masks: list[np.ndarray], W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Head centre per frame on the canvas, from the biggest person blob in the matte."""
    import cv2
    xs, ys = [], []
    for m in masks:
        b = (m > 128).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
        if n < 2:
            xs.append(xs[-1] if xs else W / 2)
            ys.append(ys[-1] if ys else H * 0.3)
            continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        top = st[k, cv2.CC_STAT_TOP]
        rows = lab[top: top + max(2, int(0.16 * b.shape[0]))] == k
        cx = float(np.nonzero(rows)[1].mean())
        xs.append(cx * (W / b.shape[1]))
        ys.append((top + 0.07 * b.shape[0]) * (H / b.shape[0]))
    k = 5
    pad = lambda a: np.pad(a, (k // 2, k // 2), mode="edge")
    sm = lambda a: np.convolve(pad(np.array(a)), np.ones(k) / k, mode="valid")
    return sm(xs), sm(ys)


def build_anim(prod: Overlay, head_x: np.ndarray, head_y: np.ndarray, side: int, tl: dict, c: dict, n: int) -> sprites.Anim:
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    w = pr.product_width(prod.props["image"], prod.props["scale"], W, c["min_width"]) * c["behind_scale"]
    w = min(w, 0.9 * W)
    x0, y0 = float(np.median(head_x[:5])), float(np.median(head_y[:5]))
    lo, hi = 0.12 * W, 0.88 * W
    rest_x = min(max(x0 + side * c["behind_dx"] * W, lo), hi)
    rest_y = y0 + 0.015 * H
    follow = []
    for i in range(0, len(head_x), 2):
        tx = min(max(float(head_x[i]) + side * c["behind_dx"] * W, lo), hi)
        follow.append((i / fps, tx - rest_x, float(head_y[i]) + 0.015 * H - rest_y))
    from PIL import Image
    with Image.open(prod.props["image"]) as im:
        h = im.height * w / im.width
    away = (H - rest_y) + h / 2 + 60                  # far enough down that the product starts and ends out of sight
    return sprites.Anim(rest_x, rest_y, w, side * c["behind_rot"], n / fps, c["behind_enter_f"] / fps,
                        c["behind_exit_f"] / fps, from_dy=away, to_dy=away, fade_in=0.0,
                        fade_out=False, wiggle=1.5, shadow=True, follow=follow)


def render_layers(short, win: Window, masks: list[np.ndarray], tl: dict, c: dict, out_dir: str, tag: str):
    """Pass 2: the framed footage and the cut-out person, written side by side."""
    import cv2
    W, H, fps = int(tl["width"]), int(tl["height"]), tl["fps"]
    n = len(masks)
    base = media.FrameWriter(W, H, fps, os.path.join(out_dir, f"{tag}_base.mov"))
    cut = media.FrameWriter(W, H, fps, os.path.join(out_dir, f"{tag}_person.mov"), alpha=True)
    k = max(1, int(round(c["feather"] / 2)))
    try:
        for fr, m in zip(iter_framed(short, win.f_a, n, tl), masks):
            base.write(fr.tobytes())
            a = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            a = cv2.GaussianBlur(a, (0, 0), k)
            rgba = np.dstack([fr[..., 2], fr[..., 1], fr[..., 0], a])
            cut.write(np.ascontiguousarray(rgba).tobytes())
    finally:
        b, p = base.close(), cut.close()
    return b, p


def split_pieces(short, f_a: int, f_b: int, fps: float) -> None:
    """Make piece boundaries at f_a and f_b so the baked layers replace whole pieces."""
    for f in (f_a, f_b):
        bounds = short.frame_bounds(fps)
        for k, p in enumerate(short.pieces):
            if bounds[k] < f < bounds[k + 1]:
                m = p.src_in + (f - bounds[k]) / fps
                a = Piece(p.src_in, m, p.spk, p.who, p.zoom, p.pan, p.tilt, list(p.flags), p.note, p.media, p.media_in)
                b = Piece(m, p.src_out, p.spk, p.who, p.zoom, p.pan, p.tilt, list(p.flags), p.note, p.media,
                          p.media_in + (m - p.src_in) if p.media else 0.0)
                short.pieces[k:k + 1] = [a, b]
                break


def bake_pieces(short, f_a: int, f_b: int, fps: float, base_file: str) -> None:
    bounds = short.frame_bounds(fps)
    for k, p in enumerate(short.pieces):
        if bounds[k] >= f_a and bounds[k + 1] <= f_b:
            p.media, p.media_in = base_file, (bounds[k] - f_a) / fps
            p.zoom, p.pan, p.tilt = 1.0, 0.0, 0.0
            p.flags = list(p.flags) + ["behind"]


def windows(short, cfg: dict, fps: float) -> list[Window]:
    """Slide-ins whose shot stays on one person for long enough, with non-overlapping windows."""
    c = cfg["products"]
    out: list[Window] = []
    for o in sorted((o for o in short.overlays if o.kind == "product"), key=lambda o: o.t_in):
        who, end = stay_end(short, o.props.get("mention", o.t_in))
        if not who:
            continue
        t_b = min(o.t_in + c["hold_s"], end)
        if t_b - o.props.get("mention", o.t_in) < c["behind_min_s"]:
            continue
        out.append(Window(o, who, int(round(o.t_in * fps)), int(round(t_b * fps))))
    for a, b in zip(out, out[1:]):
        if b.f_a < a.f_b:
            a.f_b = b.f_a
    return [w for w in out if (w.f_b - w.f_a) / fps >= 1.0]


def run(plan_, cfg: dict, shorts=None, log=print, matte=None, **_):
    c = cfg["products"]
    tl = dict(plan_.timeline or cfg["timeline"])
    fps = tl["fps"]
    matte = matte or make_matte()
    for s in plan_.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        # undo an earlier run: back to the slide-ins and the pieces as the cut stages left them
        restore = s.stats.pop("behind_src", [])
        s.clear_stage("behind")
        for d in restore:
            s.overlays.append(Overlay(**d))
        s.pieces = s.pieces_before("behind")
        wins = windows(s, cfg, fps)
        if not wins:
            log(f"  {s.name}: no product stays on one person for {c['behind_min_s']:.0f}s+, slide-ins stay at the bottom")
            continue
        if matte is None:
            log(f"  {s.name}: {len(wins)} product(s) could go behind the person, but there is no person matte "
                "(pip install mediapipe opencv-python); leaving them as slide-ins")
            continue
        out_dir = os.path.join(s.work, "behind")
        side, src, done = 1, [], 0
        for k, w in enumerate(wins):
            n = w.f_b - w.f_a
            masks = person_masks(s, w.f_a, n, tl, matte)
            n = present_until(masks, fps)
            if n / fps < c["behind_min_s"] - 0.3:
                log(f"  {s.name}: {w.ov.props['name']}: the person leaves the shot, staying a slide-in")
                continue
            masks = masks[:n]
            w.f_b = w.f_a + n
            hx, hy = head_track(masks, int(tl["width"]), int(tl["height"]))
            side = -side if k else (1 if hx[:5].mean() < tl["width"] / 2 else -1)
            tag = f"behind_{done:02d}_{w.ov.props['name'].replace(' ', '_')}"
            anim = build_anim(w.ov, hx, hy, side, tl, c, n)
            prod_file = sprites.bake(w.ov.props["image"], anim, int(tl["width"]), int(tl["height"]), fps,
                                     os.path.join(out_dir, f"{tag}_product.mov"))
            base_file, person_file = render_layers(s, w, masks, tl, c, out_dir, tag)
            split_pieces(s, w.f_a, w.f_b, fps)
            bake_pieces(s, w.f_a, w.f_b, fps, base_file)
            src.append(asdict(w.ov))
            s.overlays.remove(w.ov)
            t_a, t_b = w.f_a / fps, w.f_b / fps
            props = dict(w.ov.props, anim=asdict(anim), side=side)
            s.overlays.append(Overlay("behind_product", prod_file, t_a, t_b, cfg["tracks"]["behind_product"], props,
                                      w.ov.label, "behind"))
            s.overlays.append(Overlay("behind_person", person_file, t_a, t_b, cfg["tracks"]["behind_person"], {},
                                      "person", "behind"))
            done += 1
        s.stats["behind_src"] = src
        s.stats["behind"] = done
        log(f"  {s.name}: {done} product(s) behind the person, {len(wins) - done} left as slide-in(s)")
