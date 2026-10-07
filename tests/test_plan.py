from autoedit.plan import Settings, build_segments, invert, merge, split_long


def test_invert_and_merge():
    assert invert([(2, 3), (2.5, 4)], 10) == [(0, 2), (4, 10)]
    assert merge([(0, 1), (1.2, 2), (5, 6)], 0.4) == [(0, 2), (5, 6)]


def test_split_long_prefers_scene():
    assert split_long((0, 20), [9.0], 12) == [(0, 9.0), (9.0, 20)]


def test_silence_removed_with_padding():
    cfg = Settings(pad=0.5, min_clip=1, merge_gap=0.1)
    segs = build_segments(20, [(5, 10)], [], [], [], cfg)
    assert [(s.start, s.end) for s in segs] == [(0, 5.5), (9.5, 20)]


def test_target_length():
    cfg = Settings(min_clip=1, target_len=6, max_clip=100)
    segs = build_segments(30, [(5, 10), (15, 20)], [], [2.0, 2.5, 3.0], [], Settings(pad=0, min_clip=1, target_len=6, max_clip=100, merge_gap=0))
    assert sum(s.length for s in segs) <= 6
