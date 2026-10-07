import numpy as np

from autoedit import audiofix as af, config, media
from autoedit.plan import EditPlan, Piece, Short
from conftest import make_clip

CFG = config.merge(config.DEFAULTS, {})
SR = af.SR


def voice(dur=12.0, level=0.05, noise=0.004, seed=0, bursts=((0.5, 3.5), (4.5, 7.5), (8.5, 11.5))):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for a, b in bursts:
        m = (t >= a) & (t < b)
        f0 = 150 + 20 * np.sin(2 * np.pi * 0.7 * t[m])
        ph = 2 * np.pi * np.cumsum(f0) / SR
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]) ** 2
        x[m] = sum(np.sin(h * ph) / h for h in range(1, 12)) * env
    x = x / np.abs(x).max() * level * 6
    return (x + noise * rng.standard_normal(len(t)) + 0.01 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)


def test_measure_reports_speech_gap_peak_and_tone():
    x = voice()
    m = af.measure(x)
    assert -35 < m["speech"] < -15 and m["gap"] < -12 and m["peak"] > m["speech"]
    assert 0.4 < m["active"] < 0.9
    assert m["bands"].shape == (len(af.FC),) and abs(10 * np.log10((10 ** (m["bands"] / 10)).sum())) < 0.1
    assert af.measure(np.zeros(100, np.float32))["speech"] == -99.0


def test_match_eq_pushes_toward_reference_and_clamps():
    ref = np.zeros(len(af.FC))
    cur = np.zeros(len(af.FC))
    cur[af.FC.index(4000)] = -10           # the clip is dull around 4 kHz
    cur[af.FC.index(160)] = 30             # and muddy: would need a huge cut
    g = af.match_eq(ref, cur, -12.0, 4.0)
    assert g[af.FC.index(4000)] > 0.5 and g[af.FC.index(4000)] <= 4.0
    assert g[af.FC.index(160)] == -12.0
    assert (g[np.array(af.FC) < 100] == 0).all()
    refined = af.match_eq(ref, cur - g * 0.5, -12.0, 4.0, g)
    assert refined.shape == g.shape


def test_chain_strings():
    c = CFG["audio"]
    prep = af.prep_chain(c, np.where(np.arange(len(af.FC)) == 12, 3.0, 0.0))
    assert prep.startswith("highpass=f=90,afftdn=nr=10.0") and "equalizer=f=1000:width_type=q:w=4.32:g=3.00" in prep
    assert "equalizer" not in af.prep_chain(c, None)
    fin = af.finish_chain(c, 5.5)
    assert fin.startswith("volume=5.50dB,compand=") and "-27.0/-27.0" in fin and "alimiter=limit=0.7079" in fin and "level=disabled" in fin


def test_assemble_crossfades_only_real_cuts():
    x = np.ones(SR * 4, np.float32)
    pcs = [Piece(0.0, 1.0), Piece(2.0, 3.0), Piece(3.0, 3.5)]
    y = af.assemble({"a1": x}, pcs, ["a1"] * 3, SR, 3, 29.97)
    assert len(y) == int(2.5 * SR)
    h = int(round(3 / 29.97 * SR / 2))
    cut = SR                                                      # first cut, at 1.0 s on the timeline
    assert np.allclose(y[:cut - h - 2], 1) and np.allclose(y[cut + h + 2:], 1)
    mid = y[cut - h: cut + h]
    assert abs(mid.max() - 1.4142) < 0.02                         # constant power: two correlated signals add up
    # contiguous pieces (3.0 -> 3.0) get no crossfade
    z = af.assemble({"a1": np.arange(SR * 4, dtype=np.float32)}, pcs[1:], ["a1"] * 2, SR, 3, 29.97)
    assert np.array_equal(z, np.arange(2 * SR, int(3.5 * SR), dtype=np.float32))


def test_mute_zeroes_the_span_with_short_fades():
    y = np.ones(SR, np.float32)
    y = af.mute(y, [(0.4, 0.5)], SR)
    assert (y[int(0.4 * SR): int(0.5 * SR)] == 0).all() and y[0] == 1 and y[-1] == 1
    assert 0 < y[int(0.4 * SR) - 10] < 1


def test_process_levels_cleans_and_aligns(tmp_path):
    x = voice()
    clip = make_clip(str(tmp_path / "G1.mp4"), dur=12.0, audio="none")
    wav = tmp_path / "src.wav"
    media.write_wav(str(wav), x, SR)
    import subprocess
    mp4 = str(tmp_path / "G1v.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", clip, "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", mp4], check=True)
    s = Short("G1", mp4, fps=30.0, duration=12.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(tmp_path / "w"), pieces=[Piece(0.3, 3.8, "H"), Piece(4.3, 7.8, "H"), Piece(8.3, 11.8, "H")])
    s.stats["mute"] = [[1.0, 1.2]]
    plan = EditPlan(shorts=[s], timeline={"fps": 29.97})
    af.run(plan, CFG, log=lambda *_: None)
    y, sr = media.read_wav(s.voice)
    assert sr == SR and abs(len(y) / sr - 10.5) < 0.02
    m = af.measure(y)
    assert abs(m["speech"] - (-18.0)) < 2.0 and m["peak"] <= -2.9
    before = af.measure(af.assemble({"a1": x}, s.pieces, ["a1"] * 3, SR, 0, 29.97))
    assert m["gap"] < before["gap"] - 3                           # pauses came down relative to the voice
    assert np.abs(y[int(1.0 * sr): int(1.2 * sr)]).max() < 1e-3   # the muted span
    assert "audio" in s.stats and s.stats["audio"]["speech_after"] == round(m["speech"], 1)
    # crossfade: no click at the first cut (3.5 s of audio -> timeline 3.5 s)
    i = int(3.5 * sr)
    assert np.abs(np.diff(y[i - 200: i + 200])).max() < 5 * np.abs(np.diff(y)).std() + 0.05


def bright_voice(dur=12.0, seed=1):
    """Same kind of voice but with a lot more energy up high (the 'reference' tone)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for a, b in ((0.5, 3.5), (4.5, 7.5), (8.5, 11.5)):
        m = (t >= a) & (t < b)
        ph = 2 * np.pi * 150 * t[m]
        x[m] = sum(np.sin(h * ph) / np.sqrt(h) for h in range(1, 40)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]) ** 2)
    x = x / np.abs(x).max() * 0.3
    return (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def test_reference_tone_is_approached_but_clamped(tmp_path):
    import subprocess
    x, ref = voice(), bright_voice()
    media.write_wav(str(tmp_path / "ref.wav"), ref, SR)
    media.write_wav(str(tmp_path / "src.wav"), x, SR)
    clip = make_clip(str(tmp_path / "G1.mp4"), dur=12.0, audio="none")
    mp4 = str(tmp_path / "G1v.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", clip, "-i", str(tmp_path / "src.wav"), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", mp4], check=True)
    s = Short("G1", mp4, fps=30.0, duration=12.0, audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
              work=str(tmp_path / "w"), pieces=[Piece(0.3, 11.8, "H")])
    plan = EditPlan(shorts=[s], timeline={"fps": 29.97})
    flat = config.merge(CFG, {"audio": {"reference": ""}})
    matched = config.merge(CFG, {"audio": {"reference": str(tmp_path / "ref.wav")}})
    af.run(plan, flat, log=lambda *_: None)
    y0, _ = media.read_wav(s.voice)
    af.run(plan, matched, log=lambda *_: None)
    y1, _ = media.read_wav(s.voice)
    rb = af.measure(ref)["bands"]
    hi = np.array(af.FC) >= 400
    d0 = np.abs(af.measure(y0)["bands"] - rb)[hi].mean()
    d1 = np.abs(af.measure(y1)["bands"] - rb)[hi].mean()
    assert d1 < d0 - 0.5, (d0, d1)
