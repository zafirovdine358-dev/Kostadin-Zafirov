"""autoedit <command>: the BASED IRL editing skills, planned in Python and built in DaVinci Resolve."""
import argparse
import glob
import importlib
import os
import sys

from . import config
from .plan import STAGES, EditPlan

# command -> (stages it runs, help). Mirrors the skills: /based-rough-cut, /based-camera-angles, ...
COMMANDS = {
    "rough-cut": (["rough"], "cut every stretch where nobody is speaking"),
    "camera-angles": (["angles"], "cut on speaker changes, reframe onto the speaker"),
    "fine-cut": (["fine"], "tighten each short start to finish"),
    "subtitles": (["captions", "products"], "word-for-word captions, BASED graphic, product slide-ins"),
    "products-behind": (["behind"], "products that slide in from behind the person"),
    "broll": (["broll"], "B-roll of a model using the product over its explanation"),
    "stamps-and-music": (["stamps", "music"], "batch stamp and one upbeat track per short"),
    "audio-fix": (["audio"], "clean and level the voices"),
}
STAGE_MODULE = {"rough": "rough", "angles": "angles", "fine": "finecut", "captions": "captions",
                "products": "products", "behind": "behind", "broll": "broll", "stamps": "stamps",
                "music": "music", "audio": "audiofix"}
DEFAULT_RUN = ["rough", "angles", "fine", "captions", "products", "behind", "stamps", "music", "audio"]


def _plan_path(args, cfg) -> str:
    if args.plan:
        return os.path.expanduser(args.plan)
    found = glob.glob(os.path.join(os.path.expanduser(cfg["paths"]["work"]), "*", "edit.json"))
    if not found:
        raise SystemExit("no edit plan yet: run `autoedit ingest` first (or pass --plan)")
    return max(found, key=os.path.getmtime)


def _load(args, cfg) -> tuple[EditPlan, str]:
    path = _plan_path(args, cfg)
    return EditPlan.load(path), path


def run_stages(plan: EditPlan, cfg: dict, stages: list[str], shorts: list[str] | None, log=print,
               **opts) -> None:
    for st in stages:
        mod = importlib.import_module(f"autoedit.{STAGE_MODULE[st]}")
        log(f"== {st}")
        mod.run(plan, cfg, shorts=shorts, log=log, **opts)
        plan.done(st)


def cmd_ingest(args, cfg):
    from . import ingest
    shorts = args.shorts.split(",") if args.shorts else None
    plan, work = ingest.ingest(cfg, folder=args.folder, shorts=shorts)
    path = plan.save(os.path.join(work, "edit.json"))
    print(f"edit plan: {path}")


def cmd_stage(args, cfg):
    plan, path = _load(args, cfg)
    shorts = args.shorts.split(",") if args.shorts else None
    opts = {k: getattr(args, k) for k in ("showing", "product", "at", "short", "stamp", "reference", "model")
            if getattr(args, k, None) is not None}
    if getattr(args, "llm", False):
        cfg["llm"]["enabled"] = True
    if args.command == "audio-fix":
        cfg["audio"]["enabled"] = True
    run_stages(plan, cfg, COMMANDS[args.command][0], shorts, **opts)
    plan.save(path)
    print(f"saved {path}  (next: autoedit status, then autoedit apply)")


def _write_report(plan: EditPlan, path: str) -> None:
    from . import report
    md = os.path.join(os.path.dirname(path), "report.md")
    with open(md, "w") as f:
        f.write(report.markdown(plan))
    print("report:", md)


def _build(plan, path, cfg, shorts, layout):
    """Build the timelines in Resolve. When Resolve cannot be reached (not running, no scripting in this edition),
    write the EDL + SRT files instead and exit with 3."""
    from . import export, resolve_api
    try:
        res = resolve_api.apply(plan, cfg, shorts=shorts, layout=layout)
    except resolve_api.ResolveUnavailable as e:
        print(f"Could not build in Resolve: {e}")
        for p in export.write_all(plan, os.path.dirname(path), shorts):
            print("wrote", p)
        print("Import each .edl with File > Import > Timeline and each .srt with File > Import > Subtitle "
              "(cuts and captions only: no framing, overlays or cleaned audio).")
        sys.exit(3)
    plan.save(path)
    _write_report(plan, path)
    if res["issues"]:
        sys.exit(2)


def cmd_apply(args, cfg):
    from . import export
    plan, path = _load(args, cfg)
    shorts = args.shorts.split(",") if args.shorts else None
    out = os.path.dirname(path)
    if args.edl or args.no_resolve:
        for p in export.write_all(plan, out, shorts):
            print("wrote", p)
    if not args.no_resolve:
        _build(plan, path, cfg, shorts, args.layout or cfg["timeline"]["layout"])


def cmd_status(args, cfg):
    from . import report
    plan, path = _load(args, cfg)
    print(path)
    print(report.table(plan))
    for s in plan.shorts:
        for n in s.notes:
            print(f"  {s.name}: {n}")


def cmd_run(args, cfg):
    from . import ingest, report
    shorts = args.shorts.split(",") if args.shorts else None
    plan, work = ingest.ingest(cfg, folder=args.folder, shorts=shorts)
    path = os.path.join(work, "edit.json")
    stages = args.stages.split(",") if args.stages else DEFAULT_RUN
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s): {', '.join(bad)} (known: {', '.join(STAGES)})")
    if args.llm:
        cfg["llm"]["enabled"] = True
    run_stages(plan, cfg, stages, shorts)
    plan.save(path)
    print(report.table(plan))
    _write_report(plan, path)
    if not args.dry_run:
        _build(plan, path, cfg, shorts, cfg["timeline"]["layout"])
    else:
        print("plan only: build it in Resolve with `autoedit apply` (Resolve Studio), or `autoedit apply --no-resolve` "
              "for EDL + SRT files that any edition can import")


def cmd_doctor(args, cfg):
    from . import doctor
    sys.exit(doctor.run(cfg))


def cmd_init(args, cfg):
    if args.based_root:
        cfg["paths"]["based_root"] = args.based_root
    if args.work:
        cfg["paths"]["work"] = args.work
    for k in ("whisper", "yunet", "sface"):
        if getattr(args, k, None):
            cfg["models"][k] = getattr(args, k)
    print("wrote", config.save(cfg, args.out))


def cmd_install(args, cfg):
    from . import installer
    print(installer.install_resolve_script(cfg))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="autoedit", description=__doc__)
    ap.add_argument("--config", help="settings file (default: ./autoedit.json or ~/.config/autoedit/config.json)")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, plan=True, shorts=True):
        p = sub.add_parser(name, help=help_)
        if plan:
            p.add_argument("--plan", help="edit.json to work on (default: newest in the work folder)")
        if shorts:
            p.add_argument("--shorts", help="comma separated short names, e.g. G1291,G1292")
        p.set_defaults(fn=fn)
        return p

    add("doctor", cmd_doctor, "check this computer (ffmpeg, models, Resolve) and print the order of work",
        plan=False, shorts=False)
    p = add("init", cmd_init, "write a settings file", plan=False, shorts=False)
    p.add_argument("--out", default="autoedit.json")
    p.add_argument("--based-root")
    p.add_argument("--work")
    for k in ("whisper", "yunet", "sface"):
        p.add_argument(f"--{k}", help=f"models.{k}")
    p = add("ingest", cmd_ingest, "newest dated shoot folder -> edit plan", plan=False)
    p.add_argument("--folder", help="use this shoot folder instead of the newest one")
    for name, (_, help_) in COMMANDS.items():
        p = add(name, cmd_stage, help_)
        if name == "rough-cut":
            p.add_argument("--showing", choices=["off", "flag", "claude"], help="silent 'showing to camera' moments")
        if name == "fine-cut":
            p.add_argument("--llm", action="store_true", help="let Claude read the transcript and propose cuts (needs an API key)")
        if name == "broll":
            p.add_argument("--short", help="short to put the B-roll in (default: first)")
            p.add_argument("--product", required=True, help="product being explained, e.g. 'curl cream'")
            p.add_argument("--at", type=float, help="seconds on the short's timeline (default: after first mention)")
            p.add_argument("--model", help="only B-roll of this model (folder or file name contains it)")
        if name == "stamps-and-music":
            p.add_argument("--stamp", help="stamp PNG downloaded from the Editor Portal")
        if name == "audio-fix":
            p.add_argument("--reference", help="reference voice wav to match the tone to")
    p = add("apply", cmd_apply, "build the timelines in DaVinci Resolve")
    p.add_argument("--layout", choices=["per-short", "batch"])
    p.add_argument("--edl", action="store_true", help="also write EDL + SRT files next to the plan")
    p.add_argument("--no-resolve", action="store_true", help="only write EDL + SRT files")
    add("status", cmd_status, "per-short summary of the plan", shorts=False)
    p = add("run", cmd_run, "ingest + every stage + build in Resolve", plan=False)
    p.add_argument("--folder")
    p.add_argument("--stages", help=f"comma separated, default {','.join(DEFAULT_RUN)}")
    p.add_argument("--dry-run", action="store_true", help="plan only, do not touch Resolve")
    p.add_argument("--llm", action="store_true", help="Claude reads the transcripts too (needs an API key)")
    add("install-resolve-script", cmd_install, "put a launcher in Resolve's Workspace > Scripts menu",
        plan=False, shorts=False)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    cfg = config.load(args.config)
    if args.command == "install-resolve-script":
        return cmd_install(args, cfg)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
