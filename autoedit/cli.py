"""python -m autoedit clip1.mp4 clip2.mp4 --name MyVlog [--target 60] [--edl out.edl] [--dry-run]"""
import argparse
import json

from . import analyze, plan, resolve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autoedit", description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="AutoEdit")
    ap.add_argument("--noise-db", type=float, default=-30.0, help="silence threshold (dB)")
    ap.add_argument("--min-silence", type=float, default=0.6)
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--min-clip", type=float, default=1.0)
    ap.add_argument("--max-clip", type=float, default=12.0)
    ap.add_argument("--target", type=float, help="target total length in seconds")
    ap.add_argument("--no-dead-video", action="store_true", help="keep black/frozen footage")
    ap.add_argument("--edl", help="write a CMX3600 EDL here")
    ap.add_argument("--json", help="write the edit plan as JSON here")
    ap.add_argument("--dry-run", action="store_true", help="analyze only; don't touch Resolve")
    a = ap.parse_args(argv)

    cfg = plan.Settings(pad=a.pad, min_clip=a.min_clip, max_clip=a.max_clip, target_len=a.target)
    clips = []
    for f in a.files:
        info = analyze.probe(f)
        segs = plan.build_segments(
            info.duration,
            analyze.detect_silence(f, a.noise_db, a.min_silence),
            [] if a.no_dead_video else analyze.detect_dead_video(f),
            analyze.detect_scenes(f, a.scene_threshold),
            analyze.loudness_profile(f),
            cfg)
        kept = sum(s.length for s in segs)
        print(f"{f}: {info.duration:.1f}s -> {kept:.1f}s in {len(segs)} clips")
        clips.append((f, info.fps, segs))

    fps = clips[0][1]
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([{"file": f, "fps": r, "segments": [vars(s) for s in sg]} for f, r, sg in clips], fh, indent=2)
    if a.edl:
        resolve.write_edl(a.edl, a.name, clips, fps)
    if not a.dry_run:
        resolve.build_timeline(resolve.connect(), a.name, clips, fps)
        print(f"Timeline '{a.name}' created in Resolve.")
