"""Frame/time helpers. 29.97 means 30000/1001, never 29.97 exactly."""

_EXACT = {23.976: 24000 / 1001, 29.97: 30000 / 1001, 47.952: 48000 / 1001,
          59.94: 60000 / 1001, 119.88: 120000 / 1001}


def exact(fps: float) -> float:
    for k, v in _EXACT.items():
        if abs(fps - k) < 0.01:
            return v
    return float(fps)


def to_frame(t: float, fps: float) -> int:
    return int(round(t * exact(fps)))


def to_sec(frame: int, fps: float) -> float:
    return frame / exact(fps)


def snap(t: float, fps: float) -> float:
    return to_sec(to_frame(t, fps), fps)


def timecode(frame: int, fps: float) -> str:
    """Non-drop timecode, enough for EDLs."""
    base = int(round(exact(fps)))
    s, f = divmod(int(frame), base)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"
