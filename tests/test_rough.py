import numpy as np

from autoedit import analysis, intervals as iv, rough
from autoedit.analysis import Word
from autoedit.plan import Piece

CFG = {"pre": 0.15, "post": 0.25, "min_gap": 0.9, "head_min": 0.3, "guard_db": 8.0, "guard_quiet": 0.25,
       "min_cut_frames": 4, "showing": "flag", "review_min": 2.5}


def make_an(words, dur, loud=()):
    """Energy track: loud (0 dB) under `loud` ranges, -45 dB elsewhere."""
    e = np.full(int(dur * 50), -45.0)
    for a, b in loud:
        e[int(a * 50): int(b * 50)] = 0.0
    return analysis.Analysis(words=[Word(*w) for w in words], e=e, duration=dur, engine="whisper")


def test_head_gap_tail_are_cut_with_pads():
    an = make_an([(3.0, 3.4, "hello"), (3.5, 4.0, "there"), (7.0, 7.5, "bye")], 12.0,
                 loud=[(3.0, 4.0), (7.0, 7.5)])
    cuts = rough.plan_cuts(an, 30.0, CFG)
    kinds = [k for *_, k in cuts]
    assert kinds == ["head", "gap", "tail"]
    head, gap, tail = cuts
    assert head[0] == 0.0 and abs(head[1] - 2.85) < 0.03
    assert abs(gap[0] - 4.25) < 0.03 and abs(gap[1] - 6.85) < 0.03
    assert abs(tail[0] - 7.75) < 0.03 and tail[1] == 12.0


def test_short_gaps_are_not_cut():
    an = make_an([(0.0, 1.0, "a"), (1.5, 2.0, "b")], 2.4, loud=[(0, 2)])
    assert rough.plan_cuts(an, 30.0, CFG) == []


def test_energy_guard_extends_into_clipped_word():
    an = make_an([(1.0, 2.0, "a"), (8.0, 9.0, "b")], 10.0, loud=[(1, 2.5), (7.4, 9)])
    cuts = rough.plan_cuts(an, 30.0, CFG)
    gap = [c for c in cuts if c[2] == "gap"][0]
    assert gap[0] >= 2.5 - 0.03 and gap[1] <= 7.4 + 0.03


def test_keep_snaps_to_frames_and_drops_slivers():
    keep = rough.snap_keep([(0.0, 2.01), (5.0, 5.05), (9.0, 12.0)], 30.0, 12.0)
    assert keep[0][0] == 0.0
    assert abs(keep[0][1] * 30 - round(keep[0][1] * 30)) < 1e-6
    assert keep[-1][1] == 12.0


def test_flag_mode_keeps_long_silent_gaps_and_marks_them():
    an = make_an([(1.0, 2.0, "a"), (8.0, 9.0, "b")], 10.0, loud=[(1, 2), (8, 9)])
    keep, applied, protect = rough.rough_keep(an, 30.0, CFG, "flag")
    assert len(protect) == 1 and protect[0][1] - protect[0][0] > 2.5
    assert iv.total(keep) > 7.0          # gap kept
    keep2, _, protect2 = rough.rough_keep(an, 30.0, CFG, "off")
    assert protect2 == [] and iv.total(keep2) < 4.0


def test_claude_mode_uses_judge_to_keep_only_showing_part():
    an = make_an([(1.0, 2.0, "a"), (8.0, 9.0, "b")], 10.0, loud=[(1, 2), (8, 9)])
    keep, applied, protect = rough.rough_keep(an, 30.0, CFG, "claude", judge=lambda r: [(4.0, 5.0)])
    assert protect == [(4.0, 5.0)]
    assert any(a <= 4.0 and b >= 5.0 for a, b in keep)
    assert iv.total(keep) < 5.0


def test_split_protected_flags_phone_piece():
    ps = rough.split_protected([Piece(0, 10)], [(4.0, 5.0)])
    assert [(p.src_in, p.src_out, "phone" in p.flags) for p in ps] == [(0, 4.0, False), (4.0, 5.0, True), (5.0, 10, False)]
