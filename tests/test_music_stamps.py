import os
import subprocess

import numpy as np

from autoedit import config, media, music, stamps
from autoedit.plan import EditPlan, Piece, Short


def synth(path, expr, dur=40.0, lead=0.0, sr=22050):
    af = f"adelay={int(lead * 1000)}|{int(lead * 1000)}" if lead else "anull"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s={sr}:d={dur}", "-af", af,
                    "-ac", "2", path], check=True)
    return path


KICK = "0.8*sin(2*PI*60*t)*exp(-14*mod(t,0.5))"                                   # 120 BPM
MAJOR = "0.08*(sin(2*PI*261.63*t)+sin(2*PI*329.63*t)+sin(2*PI*392*t))"
MINOR = "0.08*(sin(2*PI*261.63*t)+sin(2*PI*311.13*t)+sin(2*PI*392*t))"


def make_library(tmp_path):
    d = tmp_path / "Music"
    d.mkdir()
    synth(str(d / "ES_Beat (Instrumental Version).wav"), f"{KICK}+{MAJOR}")
    synth(str(d / "ES_Beat.wav"), f"{KICK}+{MAJOR}")
    synth(str(d / "ES_Dull Pad.wav"), "0.03*sin(2*PI*220*t)")
    synth(str(d / "ES_Epic Trailer Rise.wav"), f"{KICK}+{MAJOR}")
    synth(str(d / "Mario Kart Theme.wav"), f"{KICK}+{MAJOR}")
    synth(str(d / "ES_Tiny.wav"), f"{KICK}+{MAJOR}", dur=3.0)
    return d


def test_library_filters_prefix_banned_words(tmp_path):
    d = make_library(tmp_path)
    names = [os.path.basename(p) for p in music.library(str(d), "ES_", ["trailer", "mario"])]
    assert names == ["ES_Beat (Instrumental Version).wav", "ES_Beat.wav", "ES_Dull Pad.wav", "ES_Tiny.wav"]
    assert music.title_key("ES_Beat (Instrumental Version).wav") == music.title_key("ES_Beat.wav")
    assert music.is_instrumental("ES_Beat (Inst).wav") and not music.is_instrumental("ES_Beat.wav")


def test_analysis_finds_tempo_mode_loudness_and_lead(tmp_path):
    a = music.analyze(synth(str(tmp_path / "a.wav"), f"{KICK}+{MAJOR}"))
    assert a["ok"] and abs(a["tempo"] - 120) < 8 and a["pulse"] > 0.3 and a["major"]
    assert a["lead"] < 0.3 and a["r0_5"] > -30
    m = music.analyze(synth(str(tmp_path / "m.wav"), f"{MINOR}+0.002*sin(2*PI*50*t)"))
    assert m["major"] is False
    quiet = music.analyze(synth(str(tmp_path / "q.wav"), "0.02*sin(2*PI*220*t)"))
    assert quiet["r0_5"] < a["r0_5"] - 10
    late = music.analyze(synth(str(tmp_path / "l.wav"), f"{KICK}+{MAJOR}", dur=30, lead=2.0))
    assert 1.5 < late["lead"] < 2.5


def test_scores_prefer_loud_beaty_instrumental_and_drop_short_tracks(tmp_path):
    d = make_library(tmp_path)
    lib = music.library(str(d), "ES_", ["trailer", "mario"])
    feats = music.features(lib, str(tmp_path / "f.json"))
    sc = music.scores(feats, 30.0, config.DEFAULTS["music"])
    best = sorted(sc, key=lambda p: -sc[p])
    assert os.path.basename(best[0]) == "ES_Beat (Instrumental Version).wav"
    assert os.path.basename(best[1]) == "ES_Beat.wav"          # vocal twin ranks below
    assert "ES_Tiny.wav" not in [os.path.basename(p) for p in sc]
    assert os.path.isfile(tmp_path / "f.json")
    again = music.features(lib, str(tmp_path / "f.json"))        # cached
    assert again == feats


def test_no_repeat_within_ten_shorts():
    sc = {f"/m/ES_{i}.wav": 10 - i for i in range(12)}
    recent: list[str] = []
    picks = music.choose([None] * 11, sc, recent, 10)
    names = [os.path.basename(p) for p in picks]
    assert len(set(names[:10])) == 10 and names[10] == "ES_10.wav"          # the 11th may not be any of the last 10
    assert all(names[i] not in names[max(0, i - 10):i] for i in range(len(names)))
    # a log from the previous batch is respected
    prev = [f"ES_{i}.wav" for i in range(5)]
    out = music.choose([None], sc, list(prev), 10)
    assert os.path.basename(out[0]) == "ES_5.wav"
    assert music.choose([None], {}, [], 10) == [None]


def test_run_places_a_quiet_flush_track_on_each_short(tmp_path):
    d = make_library(tmp_path)
    cfg = config.merge(config.DEFAULTS, {"paths": {"music": str(d), "work": str(tmp_path / "work")}})
    shorts = [Short(f"G{i}", "x.mp4", duration=20, pieces=[Piece(0, 18.0 + i)], work=str(tmp_path / "w" / f"G{i}")) for i in range(2)]
    plan = EditPlan(shorts=shorts, timeline={"fps": 29.97})
    music.run(plan, cfg, log=lambda *_: None)
    ov = [o for s in shorts for o in s.overlays if o.kind == "music"]
    assert len(ov) == 2 and ov[0].label != ov[1].label                      # never the same song twice in a row
    for s, o in zip(shorts, ov):
        assert o.t_in == 0 and abs(o.t_out - s.tl_duration()) < 1e-9 and o.track == 3
        info = media.probe(o.file)
        assert abs(info.duration - s.tl_duration()) < 0.1
        x, _ = media.read_wav(o.file)
        src = media.decode_audio(o.props["src"], 0, rate=22050, dur=10).mean(axis=1)
        ratio = 20 * np.log10(np.sqrt((x[: 48000 * 5] ** 2).mean()) / np.sqrt((src[: 22050 * 5] ** 2).mean()))
        assert abs(ratio - (-18.0)) < 1.0
    assert os.path.isfile(tmp_path / "work" / "music_log.json")
    # a second run in the same work folder avoids the songs from the log
    music.run(plan, cfg, log=lambda *_: None)
    again = [o.label for s in shorts for o in s.overlays if o.kind == "music"]
    assert again == ["ES_Dull Pad"] and any("no track long enough" in n for n in shorts[1].notes)


def test_stamp_starts_on_the_cut_nearest_the_last_30_seconds(tmp_path):
    s = Short("G1", "x.mp4", pieces=[Piece(0, 20), Piece(30, 52), Piece(60, 75), Piece(80, 100)])
    c = config.DEFAULTS["stamps"]
    a, b, on_cut = stamps.stamp_window(s, c)
    assert on_cut and b == 77.0 and a in s.piece_starts() and abs(a - (77 - 30)) <= 12     # target 47: cut at 42 or 57
    short = Short("G2", "x.mp4", pieces=[Piece(0, 10), Piece(20, 31)])
    a, b, on_cut = stamps.stamp_window(short, c)
    assert (a, b, on_cut) == (10.0, 21.0, True)                  # < 1 min: last half, start on the cut nearest 10.5
    one = Short("G3", "x.mp4", pieces=[Piece(0, 40)])
    a, b, on_cut = stamps.stamp_window(one, c)
    assert (a, b, on_cut) == (20.0, 40.0, False)


def test_stamp_run_uses_newest_wm_png_and_makes_one_master_clip(tmp_path):
    from PIL import Image
    d = tmp_path / "Stamps"
    d.mkdir()
    old, new = d / "WM-AAAA-old.png", d / "WM-BBBB-new.png"
    for p in (old, new):
        Image.new("RGBA", (54, 96), (255, 255, 255, 200)).save(p)
    os.utime(old, (1, 1))
    cfg = config.merge(config.DEFAULTS, {"paths": {"stamps": str(d), "work": str(tmp_path / "work")}})
    tl = {"width": 108, "height": 192, "fps": 29.97}
    shorts = [Short(f"G{i}", "x.mp4", pieces=[Piece(0, 40 + 10 * i), Piece(50, 80)]) for i in range(2)]
    plan = EditPlan(shorts=shorts, timeline=tl)
    stamps.run(plan, cfg, log=lambda *_: None)
    ov = [o for s in shorts for o in s.overlays]
    assert len(ov) == 2 and ov[0].file == ov[1].file and ov[0].file.endswith("WM-BBBB-new_master.mov")
    assert all(o.track == cfg["tracks"]["stamp"] for o in ov)
    info = media.probe(ov[0].file)
    assert (info.width, info.height) == (108, 192) and info.duration >= max(o.t_out - o.t_in for o in ov)
    stamps.run(plan, cfg, log=lambda *_: None, stamp=str(old))
    assert [o.props["still"] for s in shorts for o in s.overlays] == [str(old)] * 2


def test_unreadable_stamp_is_noted_not_hung(tmp_path):
    bad = tmp_path / "WM-BAD.png"
    bad.write_bytes(b"x")
    cfg = config.merge(config.DEFAULTS, {"paths": {"work": str(tmp_path / "work")}})
    s = Short("G1", "x.mp4", pieces=[Piece(0, 40), Piece(50, 80)])
    stamps.run(EditPlan(shorts=[s], timeline={"width": 108, "height": 192, "fps": 29.97}), cfg,
               log=lambda *_: None, stamp=str(bad))
    assert any("could not turn the stamp into a clip" in n for n in s.notes)
