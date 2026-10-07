import numpy as np

from autoedit import framing as fr

C = fr.Canvas()


def test_min_zoom_4k_and_1080p():
    assert fr.min_zoom(3840, 2160, C) < 1.0           # 4K covers a vertical canvas at native pixels
    assert abs(fr.min_zoom(1920, 1080, C) - (1920 + 6) / 1080) < 1e-9


def test_frame_point_centres_face_and_matches_premiere_formula():
    # face centre u=1500, eyes v=700 on a 4K source, zoom 1.0: Px = 540 - (1500-1920), Py = 570 - (700-1080)
    px, py = fr.frame_point(1500, 700, 1.0, 3840, 2160, C)
    assert (px, py) == (540 + 420, 570 + 380)
    pan, tilt = fr.to_resolve(px, py, C)
    assert (pan, tilt) == (420, 960 - 950)
    # the point really lands on the target
    assert abs(px + (1500 - 1920) - 540) < 1e-9 and abs(py + (700 - 1080) - 570) < 1e-9
    # eyes high in the source cannot be pulled down to the eye line at zoom 1.0 (the clamp pins Py)
    assert fr.frame_point(1500, 900, 1.0, 3840, 2160, C)[1] == 1920 - 1080 + 3


def test_clamp_never_shows_bars():
    # face at the far left edge of the frame: cannot be centred, position is clamped
    px, py = fr.frame_point(100, 1080, 1.0, 3840, 2160, C)
    assert px == 1920 - 3
    assert fr.residual(100, 1080, 1.0, 3840, 2160, C) > 400
    # at zoom 1.0 the clamp is exactly the Premiere one
    lo, hi = fr.clamp(-1e9, -1e9, 1.2, 3840, 2160, C), fr.clamp(1e9, 1e9, 1.2, 3840, 2160, C)
    assert lo == (1080 - 1920 * 1.2 + 3, 1920 - 1080 * 1.2 + 3) and hi == (1920 * 1.2 - 3, 1080 * 1.2 - 3)


def test_pick_step_by_face_size():
    base, px = [1.0, 1.2, 1.37, 1.5], [190, 150]
    assert fr.pick_step(220, 3840, 2160, C, base, px) == 0
    assert fr.pick_step(170, 3840, 2160, C, base, px) == 1
    assert fr.pick_step(100, 3840, 2160, C, base, px) == 2
    assert fr.steps(1920, 1080, C, base)[0] > 1.7


def test_dp_keeps_corners_only():
    ts = np.linspace(0, 10, 11)
    U = np.array([0, 10, 20, 30, 40, 40, 40, 40, 40, 40, 40.0])
    idx = fr.dp(ts, U, np.zeros(11), 5)
    assert idx == [0, 4, 10]


def test_chunk_track_steady_and_moving():
    ts = np.arange(0, 10, 0.2)
    steady = fr.chunk_track(ts, np.full(len(ts), 1000.0), np.full(len(ts), 800.0), [45, 80], 1.5)
    assert len(steady) == 1 and steady[0][0] == 0.0
    U = np.where(ts < 5, 1000.0, 1300.0)
    moving = fr.chunk_track(ts, U, np.full(len(ts), 800.0), [45, 80], 1.5)
    assert len(moving) == 2 and abs(moving[0][2] - 1000) < 1 and abs(moving[1][2] - 1300) < 1
    assert moving[0][1] == moving[1][0]
    # long chunks only: a jittery track must not explode into tiny chunks
    rng = np.random.default_rng(1)
    jit = fr.chunk_track(ts, 1000 + rng.normal(0, 80, len(ts)), np.full(len(ts), 800.0), [45, 60, 80, 110, 150], 1.5)
    assert all(b - a >= 1.0 for a, b, *_ in jit) and len(jit) <= 6
