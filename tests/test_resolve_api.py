import pytest

from autoedit import config, resolve_api as ra
from autoedit.plan import EditPlan, Marker, Overlay, Piece, Short
from fake_resolve import Resolve

CFG = config.merge(config.DEFAULTS, {})
TL = {"width": 1080, "height": 1920, "fps": 29.97, "gap_s": 60.06}


def touch(tmp_path, name, data=b"x"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


@pytest.fixture
def world(tmp_path):
    srcs = [touch(tmp_path, "G1.mp4"), touch(tmp_path, "G2.mp4")]
    voice = touch(tmp_path, "voice.wav")
    music = touch(tmp_path, "music.wav")
    prod = touch(tmp_path, "prod.mov")
    stamp = touch(tmp_path, "stamp.mov")
    srt = touch(tmp_path, "G1.srt")
    broll = touch(tmp_path, "roll.mp4")
    baked = touch(tmp_path, "baked.mov")
    durs = {"G1.mp4": 2000, "G2.mp4": 2000, "voice.wav": 400, "music.wav": 600, "prod.mov": 100, "stamp.mov": 900,
            "roll.mp4": 900, "baked.mov": 100}
    s1 = Short("G1", srcs[0], fps=29.97, width=3840, height=2160, duration=60.0,
               audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []}, voice=voice, subtitles=srt,
               pieces=[Piece(1.0, 4.0, "H", "H", 1.2, -300.0, 40.0), Piece(10.0, 12.5, "G", "G", 1.2, 410.0, 40.0)])
    s1.overlays = [Overlay("product", prod, 1.0, 3.0, 4, {"name": "pomade"}, "pomade", "products"),
                   Overlay("stamp", stamp, 3.0, 5.5, 6, {"scaling": 2}, "stamp", "stamps"),
                   Overlay("broll", broll, 2.0, 3.0, 5, {"src_in": 4.0, "scaling": 3, "fps": 30.0}, "roll", "broll"),
                   Overlay("music", music, 0.0, 5.5, 3, {"gain_db": -18}, "Song", "music")]
    s1.markers = [Marker(0.5, "Green", "STABILISED", "x", "angles"), Marker(2.0, "Red", "BAD ANGLE - fix", "y", "angles")]
    s2 = Short("G2", srcs[1], fps=29.97, width=3840, height=2160, duration=60.0,
               audio={"mic": "single", "a1": [0, 0], "a2": None, "empty": []},
               pieces=[Piece(0.0, 2.0, "H", "H", 1.0, 0.0, 0.0), Piece(2.0, 4.0, "H", "H", 1.0, 0.0, 0.0, media=baked, media_in=0.0)])
    plan = EditPlan(batch={"name": ""}, timeline=dict(TL), shorts=[s1, s2])
    return plan, durs


def run(plan, durs, **kw):
    knobs = {k: kw.pop(k) for k in list(kw) if k in ("ignore_record_frame", "record_absolute", "ignore_start_tc",
                                                    "support_subtitles", "audio_in_samples", "ignore_timeline_settings",
                                                    "fail_append", "allow_edl_import")}
    r = Resolve(durs, **knobs)
    out = ra.apply(plan, CFG, resolve=r, log=lambda *_: None, **kw)
    return r, out


def items(r, name, kind, track):
    tl = next(t for t in r.project.timelines if t.name == name)
    return tl, tl.GetItemListInTrack(kind, track)


def test_builds_a_verified_timeline_per_short(world):
    plan, durs = world
    r, out = run(plan, durs)
    assert out["timelines"] == ["BATCH 1 - G1", "BATCH 1 - G2"] and out["issues"] == [], out["issues"]
    assert plan.batch["name"] == "BATCH 1"
    tl, v1 = items(r, "BATCH 1 - G1", "video", 1)
    assert tl.GetSetting("timelineResolutionWidth") == "1080" and tl.GetSetting("timelineFrameRate") == "29.97"
    assert tl.start_frame == 0
    # V1: two pieces back to back, native pixels (Crop scaling) with the planned framing
    bounds = plan.shorts[0].frame_bounds(29.97)
    assert [(i.start, i.dur) for i in v1] == [(0, bounds[1]), (bounds[1], bounds[2] - bounds[1])]
    assert v1[0].props == {"Scaling": 1, "ZoomX": 1.2, "ZoomY": 1.2, "Pan": -300.0, "Tilt": 40.0}
    assert v1[1].props["Pan"] == 410.0
    assert v1[0].src_start == round(1.0 * 29.97)
    # the cleaned voice replaces the camera audio: V1 is picture only, A1 holds the voice from frame 0
    assert [i.kind for i in tl.tracks["video"][1]] == ["video", "video"]
    a1 = tl.GetItemListInTrack("audio", 1)
    assert [(i.mpi.GetName(), i.start, i.dur) for i in a1] == [("voice.wav", 0, 400 if bounds[2] > 400 else bounds[2])]
    # overlays on their own tracks at the right frames
    _, v4 = items(r, "BATCH 1 - G1", "video", 4)
    assert (v4[0].start, v4[0].dur) == (round(1.0 * 29.97), round(3.0 * 29.97) - round(1.0 * 29.97))
    _, v6 = items(r, "BATCH 1 - G1", "video", 6)
    assert v6[0].start == round(3.0 * 29.97) and v6[0].mpi.GetName() == "stamp.mov"
    _, v5 = items(r, "BATCH 1 - G1", "video", 5)
    assert v5[0].props == {"Scaling": 3} and v5[0].src_start == 120           # 4.0 s into a 30 fps clip
    _, a3 = items(r, "BATCH 1 - G1", "audio", 3)
    assert a3[0].mpi.GetName() == "music.wav" and a3[0].start == 0
    # captions and markers
    assert tl.GetItemListInTrack("subtitle", 1)[0].mpi.GetName() == "G1.srt"
    assert sorted(tl.markers) == [round(0.5 * 29.97), round(2.0 * 29.97)] and tl.markers[round(0.5 * 29.97)][0] == "Green"


def test_baked_pieces_keep_the_original_sound_when_there_is_no_voice(world):
    plan, durs = world
    r, out = run(plan, durs)
    tl, v1 = items(r, "BATCH 1 - G2", "video", 1)
    assert [i.mpi.GetName() for i in v1] == ["G2.mp4", "baked.mov"]
    assert v1[1].props == {}                                    # already framed: no Resolve transform on top
    kinds = [(i.mpi.GetName(), i.start) for i in tl.GetItemListInTrack("audio", 1)]
    bounds = plan.shorts[1].frame_bounds(29.97)
    assert ("G2.mp4", 0) in kinds and ("G2.mp4", bounds[1]) in kinds       # linked audio of piece 1, audio-only for baked piece 2


def test_batch_numbering_unique_names_and_media_reuse(world):
    plan, durs = world
    r = Resolve(durs)
    from fake_resolve import Timeline
    for n in ("BATCH 3", "BATCH 4 - G1", "Something else"):
        r.project.timelines.append(Timeline(n, r.project.pool))
    out = ra.apply(plan, CFG, resolve=r, log=lambda *_: None)
    assert plan.batch["name"] == "BATCH 5" and out["timelines"][0] == "BATCH 5 - G1"
    n_imports = r.project.pool.imports
    plan.batch["name"] = "BATCH 5"
    out2 = ra.apply(plan, CFG, resolve=r, log=lambda *_: None)
    assert out2["timelines"][0] == "BATCH 5 - G1 v2"            # never replaces a timeline you may be editing
    assert r.project.pool.imports == n_imports                   # media already in the batch folder is reused


def test_batch_layout_spaces_shorts_a_minute_apart(world):
    plan, durs = world
    r, out = run(plan, durs, layout="batch")
    assert out["timelines"] == ["BATCH 1"] and out["issues"] == []
    tl, v1 = items(r, "BATCH 1", "video", 1)
    b1 = plan.shorts[0].frame_bounds(29.97)[-1]
    assert len(v1) == 4 and v1[2].start == b1 + 1800            # 60.06 s at 29.97 = 1800 frames
    assert tl.GetItemListInTrack("subtitle", 1)[0].start == 0


def test_shorts_filter_and_disabled_shorts(world):
    plan, durs = world
    plan.shorts[1].enabled = False
    r, out = run(plan, durs)
    assert out["timelines"] == ["BATCH 1 - G1"]
    plan.shorts[1].enabled = True
    r, out = run(plan, durs, shorts=["G2"])
    assert out["timelines"] == ["BATCH 1 - G2"]


def test_api_quirks_are_survived_or_reported(world):
    plan, durs = world
    # recordFrame ignored: clips pile up at the end of the track, V1 still right (back to back), overlays are wrong
    r, out = run(plan, durs, ignore_record_frame=True)
    assert any("V4" in i or "stamp" in i or "pomade" in i for i in out["issues"]), out["issues"]
    # audio clip frames counted in samples: the call with start/end fails, the plain one works
    plan2, _ = world
    r, out = run(plan2, durs, audio_in_samples=True)
    tl, a1 = items(r, "BATCH 1 - G1", "audio", 1)
    assert [i.mpi.GetName() for i in a1] == ["voice.wav"]


def test_timeline_settings_that_do_not_stick_are_reported(world):
    plan, durs = world
    r, out = run(plan, durs, ignore_timeline_settings=True)
    assert any("timelineResolutionWidth" in i for i in out["issues"])


def test_missing_subtitle_support_tells_you_what_to_do(world):
    plan, durs = world
    r, out = run(plan, durs, support_subtitles=False)
    assert any("captions were not placed" in i and "G1.srt" in i for i in out["issues"])


def test_absolute_record_frames_are_detected_and_compensated(world):
    plan, durs = world
    r, out = run(plan, durs, record_absolute=True, ignore_start_tc=True)
    tl, v1 = items(r, "BATCH 1 - G1", "video", 1)
    assert tl.start_frame == 108000 and v1[0].start == 108000                 # first clip sits at the timeline start
    _, v4 = items(r, "BATCH 1 - G1", "video", 4)
    assert v4[0].start == 108000 + round(1.0 * 29.97)
    assert out["issues"] == [], out["issues"]


def test_old_resolve_is_flagged(world):
    plan, durs = world
    r = Resolve(durs, version=(18, 1, 0, 1, ""))
    out = ra.apply(plan, CFG, resolve=r, log=lambda *_: None)
    assert any("18.5" in i for i in out["issues"])


def test_no_project_open():
    class NoProject(Resolve):
        def GetProjectManager(self):
            class M:
                def GetCurrentProject(self):
                    return None
            return M()
    with pytest.raises(ra.ResolveUnavailable):
        ra.Builder(NoProject(), CFG)


def test_if_clips_cannot_be_appended_the_cut_list_is_imported_instead(world):
    plan, durs = world
    r, out = run(plan, durs, fail_append=True)
    assert any("imported the cut list as 'BATCH 1 - G1 (cuts only)'" in i for i in out["issues"]), out["issues"]
    path, opts = r.project.pool.imported[0]
    assert path.endswith("G1.edl") and opts["timelineName"] == "BATCH 1 - G1 (cuts only)"
    assert "FROM CLIP NAME: G1.mp4" in open(path).read()
    r2, out2 = run(plan, durs, fail_append=True, allow_edl_import=False)
    assert any("import G1.edl by hand" in i for i in out2["issues"])
