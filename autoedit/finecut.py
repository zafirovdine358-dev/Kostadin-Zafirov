"""Fine cut (port of based-fine-cut): same conversation start to finish, just tighter and cleaner."""
import re
from dataclasses import dataclass

import numpy as np

from . import analysis, intervals as iv, llm, speech, turns as tn
from .plan import Piece

MIN_SLIVER_FRAMES = 3
ENDS = (".", "?", "!", ",")


@dataclass
class Cut:
    s: float
    e: float
    reason: str


def norm(t: str) -> str:
    return re.sub(r"[^\w' ]", "", t.lower()).strip()


def sentence_speakers(sents: list, T: list) -> list[str]:
    return [tn.speaker_at(T, s.s) if T else "" for s in sents]


def snap_cut(words: list, e: np.ndarray, a: float, b: float) -> tuple[float, float]:
    """Move a removal to the real speech edges around it (port of fcplan.snap)."""
    prev = [w for w in words if w.e <= a + 0.15]
    nxt = [w for w in words if w.s >= b - 0.15]
    x, y = a, b
    if prev:
        se = speech.speech_end(e, prev[-1].e, a + 0.4) + 0.08
        if abs(se - a) < 0.3:
            x = max(se, prev[-1].e + 0.05)
    if nxt:
        ss = speech.speech_start(e, nxt[0].s, b - 0.4)
        if abs(ss - b) < 0.3:
            y = ss
    return (round(x, 3), round(y, 3)) if y > x else (a, b)


def has_speech(words: list, e: np.ndarray, x: float, y: float) -> bool:
    for w in words:
        if x + 0.06 < (w.s + w.e) / 2 < y - 0.06 and w.e - w.s < 1.5:
            return True
    return speech.loud_run(e, x, y)


# ---------------------------------------------------------------- individual rules
def filler_cuts(words: list, e: np.ndarray, fillers: set[str]) -> list[Cut]:
    out = []
    for i, w in enumerate(words):
        if norm(w.t) in fillers:
            lo = words[i - 1].e + 0.02 if i else 0.0
            hi = words[i + 1].s - 0.02 if i + 1 < len(words) else w.e + 0.1
            a, b = max(w.s - 0.02, lo), min(w.e + 0.04, hi)
            if b > a:
                out.append(Cut(a, b, "filler"))
    return out


def stutter_cuts(words: list, stutter_words: set[str]) -> list[Cut]:
    out = []
    for w1, w2 in zip(words, words[1:]):
        n1, n2 = norm(w1.t), norm(w2.t)
        if n1 and n1 == n2 and n1 in stutter_words and w2.s - w1.e < 0.5 and not w1.t.rstrip().endswith(ENDS[:3]):
            out.append(Cut(w1.s, w2.s - 0.02, "stutter"))
    return out


def duplicate_sentence_cuts(sents: list, spk: list[str]) -> list[Cut]:
    out = []
    for k in range(1, len(sents)):
        a, b = sents[k - 1], sents[k]
        if spk[k - 1] == spk[k] and norm(a.text) and norm(a.text) == norm(b.text) and b.s - a.e < 3.0:
            out.append(Cut(a.s, b.s - 0.02, "repeat"))
    return out


def echo_cuts(sents: list, spk: list[str], echo_words: set[str]) -> list[Cut]:
    """Host repeating the guest's last words ('Sea salt spray.') or a bare 'Really?'."""
    out = []
    for k in range(1, len(sents) - 1):                 # never the last line: it can be the handover
        s, prev = sents[k], sents[k - 1]
        if spk[k] != "H" or spk[k - 1] != "G" or s.s - prev.e > 2.5:
            continue
        mine = norm(s.text).split()
        theirs = norm(prev.text).split()[-5:]
        if not mine or len(mine) > 3:
            continue
        if all(w in theirs for w in mine) or (len(mine) == 1 and mine[0] in echo_words):
            out.append(Cut(s.s, s.e + 0.05, "echo"))
    return out


def find_hook(sents: list, patterns: list[str], start: float, max_lead: float) -> int | None:
    for k, s in enumerate(sents):
        if s.s - start > max_lead:
            break
        if any(re.search(p, s.text.lower()) for p in patterns):
            return k
    return None


def ending_cuts(sents: list, spk: list[str], closers: str, keep_thanks: int) -> list[tuple[int, str]]:
    """Trailing thank-yous and goodbyes after the last real exchange: [(sentence index, why)]."""
    rx = re.compile(closers, re.I)
    k = len(sents)
    while k > 1 and rx.match(norm(sents[k - 1].text)):
        k -= 1
    tail = list(range(k, len(sents)))
    if not tail:
        return []
    keep = []
    for i in tail:
        if len(keep) < keep_thanks and norm(sents[i].text).startswith("thank") and spk[i] != "H":
            keep.append(i)
    return [(i, "goodbye") for i in tail if i not in keep]


def pause_cuts(words: list, e: np.ndarray, ranges: list[tuple[float, float]], c: dict) -> list[Cut]:
    """Remove silence between sentences/clauses and at the edges of each kept range, leaving a natural tail
    (port of fcb/edge.tighten2, plus 'shave the head'). The last range keeps its ending: see plan_cuts."""
    out = []
    for n, (ka, kb) in enumerate(ranges):
        ws = [w for w in words if w.s >= ka - 0.05 and w.e <= kb + 0.05]
        if not ws:
            if kb - ka >= c["min_remove"] and not has_speech(words, e, ka, kb):
                out.append(Cut(round(ka, 2), round(kb, 2), "silence"))       # nothing left to say here
            continue
        onset = speech.speech_start(e, ws[0].s, ka)
        if onset - ka > c["head_pad"] + c["min_remove"] and not has_speech(words, e, ka, onset - c["head_pad"]):
            out.append(Cut(round(ka, 2), round(onset - c["head_pad"], 2), "head"))
        for w1, w2 in zip(ws, ws[1:]):
            gap = w2.s - w1.e
            if gap < c["min_gap"]:
                continue
            if not w1.t.strip().endswith(ENDS) and gap < c["long_gap"]:
                continue
            se = speech.speech_end(e, w1.e, w2.s + 0.3)
            ss = speech.speech_start(e, w2.s, se)
            tail = c["tail_q"] if w1.t.strip().endswith(("?", "!")) else c["tail"]
            x, y = max(se + tail, ka), min(ss, kb)
            if y - x >= c["min_remove"] and not has_speech(words, e, x, y):
                out.append(Cut(round(x, 2), round(y, 2), "pause"))
        if n < len(ranges) - 1:
            se = speech.speech_end(e, ws[-1].e, kb)
            tail = c["tail_q"] if ws[-1].t.strip().endswith(("?", "!")) else c["tail"]
            if kb - (se + tail) >= c["min_remove"] + 0.1 and not has_speech(words, e, se + tail, kb):
                out.append(Cut(round(se + tail, 2), round(kb, 2), "tail"))
    return out


# ---------------------------------------------------------------- orchestration
def plan_cuts(an: analysis.Analysis, pieces: list[Piece], T: list, protect: list[tuple[float, float]], cfg: dict,
              ask=None) -> tuple[list[Cut], dict]:
    c = cfg["fine"]
    words = [w for w in an.words if w.t]
    info: dict = {"hook": "", "flags": [], "llm": 0}
    if not words:
        return [], info
    sents = tn.sentences(words)
    spk = sentence_speakers(sents, T)
    kept = [(p.src_in, p.src_out) for p in pieces]
    k0 = kept[0][0]
    cuts: list[Cut] = []

    h = find_hook(sents, c["hook_patterns"], max(k0, 0.0), c["hook_max_lead"])
    if h is not None:
        info["hook"] = sents[h].text
        if h > 0:
            first = words[sents[h].first]
            cuts.append(Cut(k0, speech.speech_start(an.e, first.s, first.s - 0.4) - c["head_pad"], "hook"))
        nxt = next((i for i in range(h + 1, len(sents)) if spk[i] == "G"), None)
        if nxt is not None and re.search(r"\b(no|nope|not really|don'?t|haven'?t|never)\b", sents[nxt].text.lower()) \
                and not re.search(r"\b(yes|yeah|yep|i do|i am)\b", sents[nxt].text.lower()):
            info["flags"].append(f"guest may not follow BASED ({sents[nxt].text!r}): check this can be a short")

    cuts += filler_cuts(words, an.e, set(c["fillers"]))
    cuts += stutter_cuts(words, set(c.get("stutter_words", ["i", "the", "a", "to", "and", "it", "that", "we",
                                                            "you", "my", "is", "so", "but", "like"])))
    cuts += duplicate_sentence_cuts(sents, spk)
    for cut in echo_cuts(sents, spk, set(c["echo_words"])):
        cuts.append(Cut(*snap_cut(words, an.e, cut.s, cut.e), cut.reason))
    for i, why in ending_cuts(sents, spk, c["closers"], c["keep_thanks"]):
        cuts.append(Cut(sents[i].s - 0.02, sents[i].e + 0.05, why))

    if ask is not None:
        cuts += _llm_cuts(words, an, kept, cuts, T, ask, cfg, info)

    # end of the short: a natural tail after the last thing that stays
    removed = iv.merge([(x.s, x.e) for x in cuts])
    last_word = next((w for w in reversed(words) if not iv.overlaps(removed, w.s + 0.01, w.e - 0.01)), None)
    if last_word is not None:
        end = speech.speech_end(an.e, last_word.e, last_word.e + 0.6) + c["end_pad"]
        if end < kept[-1][1] - 0.2:
            cuts.append(Cut(end, kept[-1][1], "end"))

    # tighten what is left
    keep_now = iv.subtract(kept, [(x.s, x.e) for x in cuts])
    cuts += pause_cuts(words, an.e, keep_now, c)

    # never cut into a kept "showing" stretch
    final = []
    for x in cuts:
        for a, b in iv.subtract([(x.s, x.e)], protect):
            final.append(Cut(a, b, x.reason))
    return final, info


def _llm_cuts(words, an, kept, cuts, T, ask, cfg, info) -> list[Cut]:
    """Claude proposes word ranges to drop; only validated, speech-snapped ranges survive."""
    gone = iv.merge([(x.s, x.e) for x in cuts])
    live = [(i, w) for i, w in enumerate(words) if not iv.overlaps(gone, w.s + 0.01, w.e - 0.01)
            and iv.contains(kept, (w.s + w.e) / 2)]
    if len(live) < 6:
        return []
    ss = tn.sentences([w for _, w in live])
    text = llm.numbered_transcript([[tn.speaker_at(T, x.s) if T else "", [(live[j][0], live[j][1].t)
                                                                           for j in range(x.first, x.last + 1)]]
                                    for x in ss])
    try:
        out = ask(text, len(words))
    except Exception as ex:                      # no key, no network, refusal, bad JSON: the rules above still stand
        info["flags"].append(f"Claude pass skipped: {ex}")
        return []
    by = {i: w for i, w in live}
    res: list[Cut] = []
    for c in out["cuts"]:
        ws = [by[i] for i in range(c["start"], c["end"] + 1) if i in by]
        if ws:
            a, b = snap_cut(words, an.e, ws[0].s - 0.02, ws[-1].e + 0.05)
            res.append(Cut(a, b, c["reason"]))
    if out.get("hook_word") in by and by[out["hook_word"]].s > kept[0][0] + 0.3:
        res.append(Cut(kept[0][0], speech.speech_start(an.e, by[out["hook_word"]].s, by[out["hook_word"]].s - 0.4)
                       - cfg["fine"]["head_pad"], "hook"))
    if out.get("end_word") in by:
        w = by[out["end_word"]]
        end = speech.speech_end(an.e, w.e, w.e + 0.6) + cfg["fine"]["end_pad"]
        if end < kept[-1][1] - 0.2:
            res.append(Cut(end, kept[-1][1], "end"))
    total = sum(x.e - x.s for x in res)
    if total > 0.45 * sum(b - a for a, b in kept):
        info["flags"].append(f"Claude proposed cutting {total:.0f}s (too much): ignored its cuts")
        return []
    info["llm"] = len(res)
    info["flags"] += [f"Claude: {f['kind']}: {f['note']}" for f in out.get("flags", [])]
    return res


def apply_cuts(pieces: list[Piece], cuts: list[Cut], fps: float) -> list[Piece]:
    """Remove the cut ranges from the pieces; what is left keeps its framing."""
    rm = iv.merge([(c.s, c.e) for c in cuts])
    out = []
    for p in pieces:
        for a, b in iv.invert(rm, p.src_in, p.src_out):
            if b - a >= MIN_SLIVER_FRAMES / fps:
                out.append(Piece(a, b, p.spk, p.who, p.zoom, p.pan, p.tilt, list(p.flags), p.note, p.media))
    return out


def run(plan, cfg: dict, shorts=None, log=print, **_):
    for s in plan.shorts:
        if shorts and s.name not in shorts or not s.enabled:
            continue
        s.clear_from("fine")
        pieces = s.pieces_before("fine")
        s.pieces = pieces
        try:
            an = analysis.load(s, cfg, log, need_text=True)
        except SystemExit as ex:
            s.note(f"[fine] skipped, needs word timings ({ex})")
            s.snapshot("fine")
            log(f"  {s.name}: skipped (no word timings)")
            continue
        T = [tn.Turn(float(a), float(b), sp) for a, b, sp in s.stats.get("turns", [])]
        protect = [tuple(p) for p in s.stats.get("protect", [])]
        ask = None
        if cfg["llm"]["enabled"]:
            ask = lambda text, n: llm.propose_cuts(cfg, text, n)
        cuts, info = plan_cuts(an, pieces, T, protect, cfg, ask)
        before = s.tl_duration()
        s.pieces = apply_cuts(pieces, cuts, s.fps)
        s.snapshot("fine")
        after = s.tl_duration()
        by = {}
        for x in cuts:
            by[x.reason] = by.get(x.reason, 0) + 1
        s.stats["fine"] = {"before": round(before, 2), "after": round(after, 2), "cuts": by, "hook": info["hook"]}
        if info["hook"]:
            s.note(f"[fine] hook: {info['hook']}")
        for f in info["flags"]:
            s.note(f"[fine] {f}")
        log(f"  {s.name}: {before:.1f}s -> {after:.1f}s  " + ", ".join(f"{n} {r}" for r, n in sorted(by.items())))
