"""Product slide-ins (port of the product half of based-subtitles): the product image at its first mention."""
import os
from dataclasses import asdict, dataclass

from . import analysis, captions as cp, config, sprites, turns as tn
from .plan import Overlay


@dataclass
class Mention:
    name: str
    t: float            # timeline seconds of the first word of the mention


def phrase_index(items: dict) -> list[tuple[list[str], str]]:
    out = []
    for name, it in items.items():
        for ph in it.get("phrases", [name]):
            out.append((ph.lower().split(), name))
    return sorted(out, key=lambda x: -len(x[0]))


def find_mentions(words: list, items: dict, repeat_gap: float, extra_fixes=None) -> list[Mention]:
    """First mention of each product (again only if it comes back much later)."""
    toks = []
    for w in words:
        lo, q = cp.norm_token(w.t)
        if lo and lo not in cp.FILLERS:
            toks.append(cp.Tok(lo, w.s, w.e, q))
    toks = cp.apply_fixes(toks, extra_fixes)
    idx = phrase_index(items)
    out: list[Mention] = []
    last: dict[str, float] = {}
    i = 0
    while i < len(toks):
        hit = next(((ph, n) for ph, n in idx if [t.t.lower() for t in toks[i:i + len(ph)]] == ph), None)
        if hit is None:
            i += 1
            continue
        ph, name = hit
        t = toks[i].s
        if name not in last or t - last[name] >= repeat_gap:
            out.append(Mention(name, t))
            last[name] = t
        i += len(ph)
    return out


def find_image(folder: str, filename: str) -> str:
    """Exact file, then case-insensitive, then the same name with another image extension."""
    if not filename:
        return ""
    p = os.path.join(folder, filename)
    if os.path.isfile(p):
        return p
    if not os.path.isdir(folder):
        return ""
    stem = os.path.splitext(filename)[0].lower()
    names = sorted(os.listdir(folder), key=lambda n: (os.path.splitext(n)[1].lower() != ".png", n))
    for n in names:
        if n.lower() == filename.lower() or (os.path.splitext(n)[0].lower() == stem
                                             and os.path.splitext(n)[1].lower() in (".png", ".webp", ".jpg", ".jpeg")):
            return os.path.join(folder, n)
    return ""


def product_width(img_path: str, scale: float, canvas_w: float, min_w: float) -> float:
    """Premiere's 'Scale %' is a percentage of the image's own pixels."""
    from PIL import Image
    with Image.open(img_path) as im:
        w = im.width * scale / 100.0
    return max(min(w, 0.9 * canvas_w), min_w * canvas_w)


def slide_in(img_path: str, scale: float, tl: dict, c: dict, dur: float) -> sprites.Anim:
    W, H = tl["width"], tl["height"]
    w = product_width(img_path, scale, W, c["min_width"])
    from PIL import Image
    with Image.open(img_path) as im:
        h = im.height * w / im.width
    y = c["y"] * H
    return sprites.Anim(W / 2, y, w, 0.0, dur, c["enter_s"], c["exit_s"], from_dy=(H - y) + h / 2 + 60,
                        to_dy=0.12 * H, fade_in=0.25, fade_out=True, wiggle=2.0, shadow=True)


def plan(short, mentions: list[Mention], cfg: dict, products_dir: str, log=print) -> list[Overlay]:
    c = cfg["products"]
    end = short.tl_duration()
    out: list[Overlay] = []
    for k, m in enumerate(sorted(mentions, key=lambda m: m.t)):
        it = c["items"][m.name]
        img = find_image(products_dir, it.get("image", ""))
        if not img:
            short.note(f"[products] no image for '{m.name}' ({it.get('image')}) in {products_dir}")
            continue
        t_in = max(0.0, m.t - 0.1)
        t_out = min(t_in + c["hold_s"], end)
        if t_out - t_in < 0.6:
            continue
        out.append(Overlay("product", img, t_in, t_out, cfg["tracks"]["product"],
                           {"name": m.name, "image": img, "scale": it.get("scale", 50), "mention": round(m.t, 3)},
                           m.name, "products"))
    # a later product takes over the slot from an earlier one that is still on screen
    for a, b in zip(out, out[1:]):
        if b.t_in < a.t_out:
            a.t_out = b.t_in
    return [o for o in out if o.t_out - o.t_in >= 0.6]


def bake_all(short, overlays: list[Overlay], cfg: dict, tl: dict, log=print) -> None:
    c = cfg["products"]
    fps = tl["fps"]
    for k, o in enumerate(overlays):
        a = slide_in(o.props["image"], o.props["scale"], tl, c, o.t_out - o.t_in)
        out = os.path.join(short.work, "overlays", f"product_{k:02d}_{o.props['name'].replace(' ', '_')}.mov")
        sprites.bake(o.props["image"], a, int(tl["width"]), int(tl["height"]), fps, out)
        o.props["anim"] = asdict(a)
        o.props["still"] = o.file
        o.file = out


def run(plan_, cfg: dict, shorts=None, log=print, **_):
    c = cfg["products"]
    if not c["enabled"]:
        return
    pdir = config.path(cfg, "products")
    tl = plan_.timeline or cfg["timeline"]
    try:
        import PIL  # noqa: F401  (only checking that it is installed)
    except ImportError:
        log("  Pillow is not installed: product images can't be sized or animated (pip install pillow)")
        return
    for s in plan_.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        s.clear_stage("products")
        s.clear_stage("behind")
        try:
            an = analysis.load(s, cfg, log, need_text=True)
        except SystemExit as ex:
            s.note(f"[products] skipped, needs word timings ({ex})")
            continue
        T = [tn.Turn(float(a), float(b), sp) for a, b, sp in s.stats.get("turns", [])]
        words = cp.timeline_words(s, an, T)
        ments = find_mentions(words, c["items"], c["repeat_gap_s"], cfg["captions"].get("asr_fixes"))
        ovs = plan(s, ments, cfg, pdir, log)
        bake_all(s, ovs, cfg, tl, log)
        s.overlays += ovs
        s.stats["products"] = [[o.props["name"], round(o.t_in, 2), round(o.t_out, 2)] for o in ovs]
        log(f"  {s.name}: {len(ovs)} product slide-in(s)" + (": " + ", ".join(o.props["name"] for o in ovs) if ovs else ""))
