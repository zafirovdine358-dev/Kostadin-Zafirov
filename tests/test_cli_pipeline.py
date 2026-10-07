import json
import os
import subprocess

import pytest

from autoedit import analysis, cli, media, resolve_api
from autoedit.plan import EditPlan
from conftest import make_speechy_clip
from fake_resolve import Resolve

PIL = pytest.importorskip("PIL")
from PIL import Image


class AnyLength(dict):
    def get(self, key, default=None):
        return 100000


def words_file(path, spec):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "words.json"), "w") as f:
        json.dump({"engine": "whisper", "words": [{"s": s, "e": e, "t": t, "p": 1.0, "src": "a", "spk": ""}
                                                  for t, s, e in spec]}, f)


def sentence(text, t0, step=0.3):
    out, t = [], t0
    for w in text.split():
        out.append((w, t, t + step - 0.05))
        t += step
    return out


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "have_whisper", lambda: False)
    shoot = tmp_path / "BASED" / "New Vids" / "29-09-26 Lian"
    clips = {"G1291": [(1.0, 3.0), (4.0, 5.2), (8.0, 10.0)], "G1292": [(1.0, 3.2), (4.5, 6.0)]}
    for name, bursts in clips.items():
        make_speechy_clip(str(shoot / f"{name}.mp4"), bursts, 12.0, w=640, h=360)
    work = tmp_path / "work"
    slug = work / "2026-09-29 Lian"
    words_file(str(slug / "G1291"), sentence("Are you following BASED on TikTok?", 1.0) + sentence("Yes I do.", 4.0, 0.4)
               + sentence("Try the sea salt spray.", 8.0, 0.4))
    words_file(str(slug / "G1292"), sentence("Do you know BASED is live?", 1.0, 0.4) + sentence("Oh nice.", 4.5, 0.5))
    music = tmp_path / "BASED" / "Music"
    music.mkdir(parents=True)
    for n in ("ES_One.wav", "ES_Two (Instrumental Version).wav", "ES_Three.wav"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "aevalsrc='0.8*sin(2*PI*60*t)*exp(-14*mod(t,0.5))+0.08*sin(2*PI*261.63*t)':s=22050:d=30",
                        "-ac", "2", str(music / n)], check=True)
    stamps = tmp_path / "BASED" / "Stamps"
    stamps.mkdir()
    Image.new("RGBA", (54, 96), (255, 255, 255, 180)).save(stamps / "WM-ABCD-test.png")
    prod = tmp_path / "BASED" / "Products"
    prod.mkdir()
    Image.new("RGBA", (200, 400), (200, 30, 30, 255)).save(prod / "SSS SANTAL TILTED V03 1.png")
    cfg = {"paths": {"based_root": str(tmp_path / "BASED"), "footage": "New Vids", "music": "Music", "stamps": "Stamps",
                     "products": "Products", "work": str(work)},
           "timeline": {"width": 108, "height": 192, "fps": 29.97},
           "rough": {"showing": "off"}}
    cfg_path = tmp_path / "autoedit.json"
    cfg_path.write_text(json.dumps(cfg))
    return str(cfg_path), str(slug), tmp_path


def test_whole_pipeline_to_a_plan_then_exports_then_resolve(studio, monkeypatch, capsys):
    cfg_path, slug, tmp = studio
    cli.main(["--config", cfg_path, "run", "--dry-run"])
    plan = EditPlan.load(os.path.join(slug, "edit.json"))
    assert [s.name for s in plan.shorts] == ["G1291", "G1292"]
    assert {"ingest", "rough", "angles", "fine", "captions", "products", "behind", "stamps", "music", "audio"} <= set(plan.stages)
    g1 = plan.short("G1291")
    # silence cut, the hook found, the guest's yes kept
    assert 6.0 < g1.tl_duration() < 11.0 and g1.stats["fine"]["hook"].startswith("Are you following")
    kinds = sorted(o.kind for o in g1.overlays)
    assert kinds.count("music") == 1 and kinds.count("stamp") == 1 and "product" in kinds
    assert os.path.isfile(g1.subtitles) and "sea salt spray" in open(g1.subtitles).read()
    assert os.path.isfile(g1.voice) and media.probe(g1.voice).duration == pytest.approx(g1.tl_duration(), abs=0.1)
    assert plan.short("G1291").stats["music"] != plan.short("G1292").stats["music"]
    assert os.path.isfile(os.path.join(slug, "report.md"))
    out = capsys.readouterr().out
    assert "G1291" in out and "G1292" in out

    # files for a Free edition without scripting
    cli.main(["--config", cfg_path, "apply", "--no-resolve"])
    assert os.path.isfile(os.path.join(slug, "G1291.edl")) and os.path.isfile(os.path.join(slug, "G1291.srt"))
    edl = open(os.path.join(slug, "G1291.edl")).read()
    assert edl.startswith("TITLE: G1291") and "FROM CLIP NAME: G1291.mp4" in edl

    # and the real thing against the stand-in Resolve
    fake = Resolve(AnyLength())
    monkeypatch.setattr(resolve_api, "connect", lambda: fake)
    cli.main(["--config", cfg_path, "apply"])
    names = [t.name for t in fake.project.timelines]
    assert names == ["BATCH 1 - G1291", "BATCH 1 - G1292"]
    tl = fake.project.timelines[0]
    assert len(tl.GetItemListInTrack("video", 1)) == len(g1.pieces)
    assert tl.GetItemListInTrack("audio", 1)[0].mpi.GetName() == "voice.wav"
    assert EditPlan.load(os.path.join(slug, "edit.json")).batch["name"] == "BATCH 1"


def test_stages_can_be_run_one_by_one_like_the_skills(studio, capsys):
    cfg_path, slug, tmp = studio
    cli.main(["--config", cfg_path, "ingest"])
    cli.main(["--config", cfg_path, "rough-cut", "--showing", "off"])
    cli.main(["--config", cfg_path, "camera-angles"])
    cli.main(["--config", cfg_path, "fine-cut"])
    cli.main(["--config", cfg_path, "subtitles"])
    cli.main(["--config", cfg_path, "stamps-and-music"])
    cli.main(["--config", cfg_path, "audio-fix"])
    cli.main(["--config", cfg_path, "status"])
    plan = EditPlan.load(os.path.join(slug, "edit.json"))
    assert plan.stages[:3] == ["ingest", "rough", "angles"] and "audio" in plan.stages
    first = plan.short("G1291").tl_duration()
    # re-running rough-cut invalidates what came after it, and fine-cut brings the length back down
    cli.main(["--config", cfg_path, "rough-cut", "--showing", "off"])
    plan = EditPlan.load(os.path.join(slug, "edit.json"))
    g = plan.short("G1291")
    assert g.overlays == [] and g.subtitles == "" and g.voice == "" and "fine" not in g.snaps
    assert not {"captions", "products", "music", "stamp", "audio", "fine", "turns"} & set(g.stats)   # nothing stale
    assert g.tl_duration() >= first
