"""Is someone showing a phone or product to the camera during a silent stretch? (vision check, optional)"""
from . import intervals as iv, llm, media

STEP = 0.75
PER_CALL = 12


def sample_times(a: float, b: float, step: float = STEP) -> list[float]:
    n = max(1, int((b - a) / step))
    return [a + (i + 0.5) * (b - a) / n for i in range(n)]


def runs(times: list[float], flags: list[bool], lo: float, hi: float, step: float = STEP,
         pad: float = 0.15) -> list[tuple[float, float]]:
    out = []
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            out.append((max(lo, times[i] - step / 2 - pad), min(hi, times[j] + step / 2 + pad)))
            i = j + 1
        else:
            i += 1
    return [(a, b) for a, b in iv.merge(out) if b - a >= 0.3]


def judge(short, ranges: list[tuple[float, float]], cfg: dict, log=print, ask=None, grab=None):
    """Return the parts of `ranges` where something is shown. ask(jpegs)->[bool] and grab(t)->jpeg are injectable."""
    c = cfg["llm"]
    ask = ask or (lambda jpegs: llm.showing_frames(cfg, jpegs))
    grab = grab or (lambda t: media.frame_jpeg(short.source, t, c["frame_height"]))
    budget, found = c["max_frames"], []
    for a, b in ranges:
        times = sample_times(a, b)
        if len(times) > budget:
            times = times[:budget]
        if not times:
            log(f"  {short.name}: frame budget used up, leaving {b - a:.1f}s silence to cut")
            continue
        budget -= len(times)
        flags: list[bool] = []
        try:
            for k in range(0, len(times), PER_CALL):
                flags += ask([grab(t) for t in times[k:k + PER_CALL]])
        except Exception as ex:                  # no key, no network, refusal: treat the stretch as nothing shown
            log(f"  {short.name}: could not check frames ({ex}); cutting that silence")
            continue
        found += runs(times, flags, a, b)
    if found:
        log(f"  {short.name}: something shown to camera in {len(found)} silent stretch(es)")
    return found
