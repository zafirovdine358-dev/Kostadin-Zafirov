import numpy as np

from autoedit import analysis, config, finecut as fcut, intervals as iv, turns as tn
from autoedit.analysis import Word
from autoedit.plan import Piece

CFG = config.merge(config.DEFAULTS, {})


def script(lines, dur):
    """lines: [(speaker, start, [(word, dur), ...])] -> words, energy, turns."""
    words, e, T = [], np.full(int(dur * 50), -45.0), []
    for spk, t, ws in lines:
        t0 = t
        for w, d in ws:
            words.append(Word(t, t + d, w))
            e[int(t * 50): int((t + d) * 50)] = 0.0
            t += d + 0.04
        T.append(tn.Turn(t0, t, spk))
    an = analysis.Analysis(words=words, e=e, duration=dur, engine="whisper")
    return an, T


LINES = [
    ("H", 1.0, [("Hey", 0.3), ("guys.", 0.5)]),
    ("H", 3.0, [("Are", .2), ("you", .2), ("following", .4), ("BASED", .4), ("on", .2), ("TikTok?", .5)]),
    ("G", 5.8, [("Yes,", .3), ("I", .15), ("do.", .3)]),
    ("G", 9.0, [("Sea", .3), ("salt", .3), ("spray.", .4)]),
    ("H", 10.4, [("Sea", .3), ("salt", .3), ("spray.", .4)]),
    ("G", 12.0, [("Um", .3), ("I", .15), ("I", .15), ("think", .3), ("so.", .3)]),
    ("G", 20.0, [("Thank", .3), ("you.", .4)]),
    ("H", 21.5, [("Thank", .3), ("you!", .4)]),
    ("H", 22.6, [("Bye.", .4)]),
]


def run_plan(llm=None):
    an, T = script(LINES, 26.0)
    pieces = [Piece(0, 26.0)]
    cuts, info = fcut.plan_cuts(an, pieces, T, [], CFG, llm)
    return an, T, pieces, cuts, info


def reasons(cuts):
    return sorted({c.reason for c in cuts})


def test_every_rule_fires_on_a_street_interview():
    an, T, pieces, cuts, info = run_plan()
    assert {"hook", "echo", "filler", "stutter", "goodbye", "pause", "end"} <= set(reasons(cuts)), reasons(cuts)
    assert info["hook"].startswith("Are you following BASED")
    out = fcut.apply_cuts(pieces, cuts, 30.0)
    kept = [(p.src_in, p.src_out) for p in out]

    def survives(word):
        w = next(x for x in an.words if x.t == word)
        return iv.contains(kept, (w.s + w.e) / 2)
    assert survives("following") and survives("TikTok?") and survives("do.")
    assert not survives("guys.")                       # before the hook
    assert survives("Thank") and not any(iv.contains(kept, (w.s + w.e) / 2) for w in an.words if w.t == "Bye.")
    # one 'Sea salt spray' stays (the guest's), the host echo goes
    echoes = [w for w in an.words if w.t == "spray."]
    assert iv.contains(kept, (echoes[0].s + echoes[0].e) / 2) and not iv.contains(kept, (echoes[1].s + echoes[1].e) / 2)
    # the host's closing thank-you is cut, the guest's kept
    thanks = [w for w in an.words if w.t == "Thank"]
    assert iv.contains(kept, (thanks[0].s + thanks[0].e) / 2) and not iv.contains(kept, (thanks[1].s + thanks[1].e) / 2)


def test_natural_tail_after_the_last_word_and_after_questions():
    an, T, pieces, cuts, info = run_plan()
    out = fcut.apply_cuts(pieces, cuts, 30.0)
    last = out[-1].src_out
    guest_thanks_end = [w for w in an.words if w.t == "you."][-1].e
    assert 0.5 <= last - guest_thanks_end <= 0.9
    # after the hook question there is a >= 0.28 s breath before the guest answers
    q_end = [w for w in an.words if w.t == "TikTok?"][0].e
    pause = [c for c in cuts if c.reason == "pause" and c.s >= q_end - 0.01][0]
    assert pause.s - q_end >= 0.27


def test_protected_ranges_are_never_cut():
    an, T = script(LINES, 26.0)
    cuts, _ = fcut.plan_cuts(an, [Piece(0, 26.0)], T, [(6.8, 8.9)], CFG)
    assert not any(c.s < 8.9 and c.e > 6.8 for c in cuts)


def test_does_not_cut_inside_a_sentence_and_flags_non_followers():
    lines = [("H", 1.0, [("Are", .2), ("you", .2), ("following", .4), ("BASED?", .5)]),
             ("G", 2.5, [("No,", .3), ("I", .15), ("only", .2), ("use", .2), ("YouTube.", .5)])]
    an, T = script(lines, 6.0)
    cuts, info = fcut.plan_cuts(an, [Piece(0, 6.0)], T, [], CFG)
    assert any("may not follow BASED" in f for f in info["flags"])
    assert not any(c.reason in ("stutter", "echo", "filler") for c in cuts)


def test_llm_cuts_are_validated_and_snapped():
    seen = {}

    def ask(text, n):
        seen["text"] = text
        i = [k for k, w in enumerate(an.words) if w.t == "Yes,"][0]
        return {"cuts": [{"start": i, "end": i + 1, "reason": "unclear"}], "hook_word": None, "end_word": None,
                "flags": [{"kind": "other", "note": "check ending"}]}
    an, T = script(LINES, 26.0)
    cuts, info = fcut.plan_cuts(an, [Piece(0, 26.0)], T, [], CFG, ask)
    assert "[H]" in seen["text"] and "[8] Yes," in seen["text"] and "Hey" not in seen["text"]   # hook cut already removed it
    assert any(c.reason == "unclear" for c in cuts) and info["llm"] >= 1 and info["flags"][-1].startswith("Claude: other")


def test_llm_that_wants_to_cut_half_the_short_is_ignored():
    def ask(text, n):
        return {"cuts": [{"start": 0, "end": n - 1, "reason": "unsure"}], "hook_word": None, "end_word": None, "flags": []}
    an, T = script(LINES, 26.0)
    cuts, info = fcut.plan_cuts(an, [Piece(0, 26.0)], T, [], CFG, ask)
    assert not any(c.reason == "unsure" for c in cuts) and any("too much" in f for f in info["flags"])


def test_apply_cuts_keeps_framing_and_drops_slivers():
    p = Piece(0, 10, "H", "H", 1.2, -300.0, 40.0, ["stab"])
    out = fcut.apply_cuts([p], [fcut.Cut(2, 4, "x"), fcut.Cut(4.0, 4.05, "y"), fcut.Cut(9.97, 10, "z")], 30.0)
    assert [(q.src_in, q.src_out) for q in out] == [(0, 2), (4.05, 9.97)]
    assert all(q.zoom == 1.2 and q.pan == -300.0 and q.flags == ["stab"] for q in out)
