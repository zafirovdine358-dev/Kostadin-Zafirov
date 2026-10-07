"""Subtitles (port of based-subtitles): word-for-word lowercase captions, max 3 words, BASED on its own."""
import os
import re
from dataclasses import dataclass

from . import analysis, media, speech, sprites, srt, timebase, turns as tn
from .plan import Overlay

WEAK = set("a an the to of my your our their his her and or but in on for with at is are was its it's that this "
           "so if as from about like just".split())
FILLERS = {"um", "uh", "hmm", "mm", "erm", "uh-huh", "umm", "uhh"}
I_FORMS = {"i", "i'm", "i've", "i'll", "i'd"}
ENDS = (".", "?", "!", ",")

# phrase-level ASR repairs: alternatives per position -> replacement words
DEFAULT_FIXES = [
    ([{"coke", "crow", "girl", "cruel"}, {"cream"}], ["curl", "cream"]),
    ([{"cruel", "chrome", "coral"}, {"move", "moves", "moose", "mousse", "most"}], ["curl", "mousse"]),
    ([{"curl", "curly"}, {"moose", "moves"}], ["curl", "mousse"]),
    ([{"leaving", "leave"}, {"in"}, {"conditioner"}], ["leave-in", "conditioner"]),
    ([{"leaving"}, {"conditioner"}], ["leave-in", "conditioner"]),
    ([{"skin"}, {"rib", "rivive", "revive", "rival"}, {"eye"}], ["skin", "revival"]),
    ([{"skin"}, {"rivive", "revive", "rival"}], ["skin", "revival"]),
    ([{"base", "bass", "vase", "basis", "bays", "bace", "based's"}], ["based"]),
]
BASED_BEFORE = {"following", "follow", "to", "about", "from", "with", "watching", "on"}
BASED_AFTER = {"is", "has", "are", "was", "does", "makes", "live", "tiktok", "page"}


@dataclass
class TWord:
    t: str
    s: float
    e: float
    spk: str = ""


@dataclass
class Tok:
    t: str
    s: float
    e: float
    q: bool = False
    n: int = 1


@dataclass
class Caption:
    s: float
    e: float
    text: str
    spk: str = ""
    kind: str = "text"        # text | based | brand


def norm_token(raw: str) -> tuple[str, bool]:
    q = raw.rstrip().endswith("?")
    t = re.sub(r"[^\w'’\-$/%]", "", raw).replace("’", "'")
    lo = t.lower()
    if lo in I_FORMS:
        lo = "I" + lo[1:]
    return lo, q


def apply_fixes(toks: list[Tok], extra: list[list[str]] | None = None) -> list[Tok]:
    """Repair known ASR mistakes on the token sequence, keeping the timing of the span."""
    rules = [(pat, rep) for pat, rep in DEFAULT_FIXES]
    for wrong, right in (extra or []):
        rules.append(([{w} for w in wrong.lower().split()], right.split()))
    out: list[Tok] = []
    i = 0
    while i < len(toks):
        hit = None
        for pat, rep in rules:
            if i + len(pat) <= len(toks) and all(toks[i + k].t.lower() in pat[k] for k in range(len(pat))):
                hit = (len(pat), rep)
                break
        if hit is None and toks[i].t.lower() in ("basic", "face", "grace", "case"):
            prev = out[-1].t.lower() if out else ""
            nxt = toks[i + 1].t.lower() if i + 1 < len(toks) else ""
            if prev in BASED_BEFORE or nxt in BASED_AFTER:
                hit = (1, ["based"])
        if hit is None:
            out.append(toks[i])
            i += 1
            continue
        n, rep = hit
        span = toks[i:i + n]
        s, e = span[0].s, span[-1].e
        for k, w in enumerate(rep):
            a = s + (e - s) * k / len(rep)
            b = s + (e - s) * (k + 1) / len(rep)
            out.append(Tok(w, a, b, span[-1].q and k == len(rep) - 1))
        i += n
    return out


def merge_brands(toks: list[Tok], brands: list[str]) -> tuple[list[Tok], list[tuple[float, float]]]:
    """Replace other brands with '*other brand*'; return the time spans whose audio must go."""
    out, spans, i = [], [], 0
    pats = sorted(([w for w in re.split(r"\s+", b.lower().replace("&", "and")) if w] for b in brands), key=len, reverse=True)
    while i < len(toks):
        hit = next((p for p in pats if [t.t.lower().replace("&", "and") for t in toks[i:i + len(p)]] == p), None)
        if hit:
            span = toks[i:i + len(hit)]
            out.append(Tok("*other brand*", span[0].s, span[-1].e, span[-1].q))
            spans.append((span[0].s, span[-1].e))
            i += len(hit)
        else:
            out.append(toks[i])
            i += 1
    return out, spans


def tokens(phrase: list[TWord], cfg_caps: dict) -> tuple[list[Tok], list[tuple[float, float]]]:
    toks: list[Tok] = []
    for w in phrase:
        lo, q = norm_token(w.t)
        if not lo or lo in FILLERS:
            continue
        if toks and toks[-1].t == lo and lo != "based":
            toks[-1].n += 1
            toks[-1].e, toks[-1].q = w.e, q
            continue
        toks.append(Tok(lo, w.s, w.e, q))
    toks = apply_fixes(toks, cfg_caps.get("asr_fixes"))
    return merge_brands(toks, cfg_caps["other_brands"])


DETERMINERS = {"the", "a", "an", "my", "your", "our"}
SPECIAL = ("based", "*other brand*")


def _units(run: list[Tok], groups: list[tuple]) -> list[list[Tok]]:
    """Single words, or product names (with their 'the') that must stay in one caption."""
    def group_at(i: int) -> int:
        return next((len(g) for g in groups if tuple(t.t for t in run[i:i + len(g)]) == g), 0)
    out, i = [], 0
    while i < len(run):
        g = group_at(i)
        if not g and run[i].t in DETERMINERS and 0 < group_at(i + 1) <= 2:
            g = 1 + group_at(i + 1)
        out.append(run[i:i + (g or 1)])
        i += g or 1
    return out


def _cost(units: list[list[Tok]], max_words: int) -> float:
    words = sum(len(u) for u in units)
    c = {1: 1.0, 2: 0.2}.get(words, 0.0) + 0.001 * (max_words - words)
    last = units[-1]
    if len(last) == 1 and last[0].t in WEAK:          # never end a caption on a/the/to/of...
        c += 4.0 + (1.5 if words == 1 else 0.0)
    return c


def chunk(toks: list[Tok], groups: list[list[str]], max_words: int = 3) -> list[list[Tok]]:
    """At most max_words per caption, BASED alone, product names whole, as few awkward endings as possible.

    The SOP's rules (4 -> 2+2, 5 -> 3+2, never end on a weak word) fall out of a small cost minimisation.
    """
    gs = sorted((tuple(g) for g in groups), key=len, reverse=True)
    out: list[list[Tok]] = []
    run: list[Tok] = []

    def flush():
        if not run:
            return
        units = _units(run, gs)
        n = len(units)
        best = [0.0] + [1e9] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i - 1, -1, -1):
                if sum(len(u) for u in units[j:i]) > max_words:
                    break
                c = best[j] + _cost(units[j:i], max_words)
                if c < best[i] - 1e-12:
                    best[i], back[i] = c, j
        cuts, i = [], n
        while i > 0:
            cuts.append((back[i], i))
            i = back[i]
        for a, b in reversed(cuts):
            out.append([t for u in units[a:b] for t in u])
        run.clear()

    for t in toks:
        if t.t in SPECIAL:
            flush()
            out.append([t])
        else:
            run.append(t)
    flush()
    return out


def split_long(group: list[Tok], groups: list[list[str]], long_chars: int, long_word: int) -> list[list[Tok]]:
    """'definitely recommend' is too long for one caption: break it up, but never inside a product name."""
    names = [t.t for t in group]
    if len(group) < 2 or any(names[i:i + len(g)] == list(g) for g in groups for i in range(len(names))):
        return [group]
    text = " ".join(t.t for t in group)
    two_long = sum(len(t.t) >= long_word for t in group) >= 2
    if len(text) < long_chars and not two_long:
        return [group]
    mid = (len(group) + 1) // 2
    if mid > 1 and group[mid - 1].t in WEAK:
        mid -= 1
    return [group[:mid], group[mid:]]


def build(words: list[TWord], cuts: list[float], c: dict) -> tuple[list[Caption], list[tuple[float, float]]]:
    """Captions on timeline seconds, plus time spans (timeline) of other brands to mute."""
    phrases, cur = [], []
    for w in words:
        if cur:
            p = cur[-1]
            brk = (w.spk != p.spk or p.t.rstrip().endswith(ENDS) or w.s - p.e > 0.3
                   or any(p.e - 0.02 <= x <= w.s + 0.02 for x in cuts))
            if brk:
                phrases.append(cur)
                cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)
    caps: list[Caption] = []
    mutes: list[tuple[float, float]] = []
    for ph in phrases:
        toks, spans = tokens(ph, c)
        mutes += spans
        for g in chunk(toks, c["groups"], c["max_words"]):
            for part in split_long(g, c["groups"], c["long_chars"], c["long_word"]):
                txt = " ".join(t.t + (f" x{t.n}" if t.n > 1 else "") for t in part)
                if part[-1].q and part[-1].t not in ("based", "*other brand*"):
                    txt += "?"
                kind = "based" if txt == "based" else ("brand" if txt.startswith("*other brand*") else "text")
                caps.append(Caption(part[0].s, part[-1].e, txt, ph[0].spk, kind))
    return caps, mutes


def retime(caps: list[Caption], fps: float, c: dict, end_limit: float) -> list[Caption]:
    """Back-to-back, frame-exact, at least min_dur each, the last one lingering a little."""
    if not caps:
        return caps
    mn = c["min_dur"]
    st = [x.s for x in caps]
    for k in range(1, len(st)):
        st[k] = max(st[k], st[k - 1] + 0.07)
    last_end = max(x.e for x in caps)
    st.append(max(st[-1] + 0.4, min(last_end + c["end_pad"], end_limit)))
    for _ in range(200):                                    # borrow time from the roomier neighbour
        changed = False
        for k in range(len(caps)):
            d = st[k + 1] - st[k]
            if d < mn - 1e-6:
                prev = st[k] - st[k - 1] if k else 0.0
                nxt = st[k + 2] - st[k + 1] if k + 2 < len(st) else 9.0
                if k and prev - mn >= nxt - mn and prev > mn:
                    st[k] -= min(mn - d, prev - mn)
                    changed = True
                elif nxt > mn:
                    st[k + 1] += min(mn - d, nxt - mn)
                    changed = True
        if not changed:
            break
    frames = [timebase.to_frame(t, fps) for t in st]
    for k in range(1, len(frames)):
        frames[k] = max(frames[k], frames[k - 1] + 1)
    lim = timebase.to_frame(end_limit, fps)
    if frames[-1] > lim > frames[-2]:
        frames[-1] = lim
    return [Caption(timebase.to_sec(frames[k], fps), timebase.to_sec(frames[k + 1], fps), x.text, x.spk, x.kind)
            for k, x in enumerate(caps)]


srt_time = srt.srt_time


def write_srt(caps: list[Caption], path: str, skip_kinds: tuple = ()) -> str:
    return srt.write([(x.s, x.e, x.text) for x in caps if x.kind not in skip_kinds], path)


def timeline_words(short, an: analysis.Analysis, T: list, refine: bool = True) -> list[TWord]:
    """Whisper words that survived the cuts, on timeline seconds, with who said them."""
    words = [w for w in an.words if w.t]
    out: list[TWord] = []
    pos = 0.0
    for p in short.pieces:
        for k, w in enumerate(words):
            mid = (w.s + w.e) / 2
            if not (p.src_in <= mid < p.src_out):
                continue
            s0 = w.s
            if refine:
                lim = max(w.s - 0.25, words[k - 1].e if k else 0.0, p.src_in)
                s0 = min(w.s, max(speech.speech_start(an.e, w.s, lim), lim))
            s, e = max(s0, p.src_in), min(w.e, p.src_out)
            out.append(TWord(w.t, pos + s - p.src_in, pos + e - p.src_in, tn.speaker_at(T, w.s) if T else ""))
        pos += p.dur
    return sorted(out, key=lambda w: w.s)


def run(plan, cfg: dict, shorts=None, log=print, **_):
    c = cfg["captions"]
    graphic = os.path.expanduser(cfg["paths"]["based_graphic"])
    has_graphic = bool(graphic) and os.path.isfile(graphic)
    fps = plan.timeline.get("fps", 29.97)
    for s in plan.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        s.clear_stage("captions")
        try:
            an = analysis.load(s, cfg, log, need_text=True)
        except SystemExit as ex:
            s.note(f"[captions] skipped, needs word timings ({ex})")
            log(f"  {s.name}: skipped (no word timings)")
            continue
        T = [tn.Turn(float(a), float(b), sp) for a, b, sp in s.stats.get("turns", [])]
        words = timeline_words(s, an, T)
        cuts = s.piece_starts()[1:]
        caps, mutes = build(words, cuts, c)
        caps = retime(caps, fps, c, s.tl_duration() - 0.02)
        if not has_graphic and any(x.kind == "based" for x in caps):
            s.note("[captions] no BASED graphic file (paths.based_graphic): 'based' stays as caption text")
        srt = os.path.join(s.work, f"{s.name}.srt")
        write_srt(caps, srt, skip_kinds=("based",) if has_graphic else ())
        s.subtitles = srt
        s.stats["captions"] = {"count": len(caps), "based": sum(x.kind == "based" for x in caps),
                               "brands": sum(x.kind == "brand" for x in caps)}
        s.stats["caption_list"] = [[round(x.s, 3), round(x.e, 3), x.text, x.kind] for x in caps]
        s.stats["mute"] = [[round(a, 3), round(b, 3)] for a, b in mutes]
        if has_graphic:
            based = [x for x in caps if x.kind == "based"]
            file = graphic
            if based and graphic.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    file = sprites.still_master(graphic, max(x.e - x.s for x in based), plan.timeline or cfg["timeline"],
                                                os.path.join(os.path.expanduser(cfg["paths"]["work"]), "stills"))
                except (ImportError, media.MediaError) as ex:
                    s.note(f"[captions] could not turn the BASED graphic into a clip: {ex}")
            for x in based:
                s.overlays.append(Overlay("based", file, x.s, x.e, cfg["tracks"]["based"], {"still": graphic}, "BASED",
                                          "captions"))
        log(f"  {s.name}: {len(caps)} captions, {s.stats['captions']['based']} BASED, {s.stats['captions']['brands']} other brand(s)")
