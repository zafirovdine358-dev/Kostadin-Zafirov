"""DaVinci Resolve scripting API: connect, then build timelines from an edit plan.

Standard library only, so it runs from a terminal against Resolve Studio and from Workspace > Scripts. Python scripting
is a Studio feature from Resolve 21.1 on; the Free edition could run it from Workspace > Scripts up to 21.0. It needs
Resolve 18.5+ (AppendToTimeline with trackIndex/recordFrame). After building, every timeline is read back and compared
with the plan, because the API's frame conventions are easy to get wrong: look at the issues list.
"""
import builtins
import os
import re
import sys

from . import export, timebase
from .plan import EditPlan, Short

VIDEO, AUDIO = 1, 2                    # clipInfo mediaType
COLORS = {"Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia", "Rose", "Lavender", "Sky", "Mint",
          "Lemon", "Sand", "Cocoa", "Cream"}


class ResolveUnavailable(RuntimeError):
    pass


def _defaults() -> tuple[str, str]:
    if sys.platform.startswith("win"):
        api = os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
                           r"Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting")
        lib = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
    elif sys.platform == "darwin":
        api = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
        lib = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
    else:
        api = "/opt/resolve/Developer/Scripting"
        lib = "/opt/resolve/libs/Fusion/fusionscript.so"
    return api, lib


def connect():
    """The Resolve app object. Inside Resolve a global `resolve` already exists; outside, Resolve Studio must be
    running with Preferences > System > General > External scripting using: Local."""
    for holder in (builtins, sys.modules.get("__main__")):
        app = getattr(holder, "resolve", None)
        if app is not None:
            return app
    api, lib = _defaults()
    os.environ.setdefault("RESOLVE_SCRIPT_API", api)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", lib)
    mods = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
    if os.path.isdir(mods) and mods not in sys.path:
        sys.path.append(mods)
    try:
        import DaVinciResolveScript as dvr
    except ImportError as e:
        raise ResolveUnavailable(
            "Cannot import DaVinciResolveScript. Install DaVinci Resolve and set RESOLVE_SCRIPT_API "
            f"(looked in {mods}). Scripting from a terminal needs Resolve Studio; the Free edition can only run "
            "scripts from Workspace > Scripts, and only Lua ones from 21.1 on (autoedit install-resolve-script, "
            "or autoedit apply --no-resolve).") from e
    app = dvr.scriptapp("Resolve")
    if app is None:
        raise ResolveUnavailable("Resolve is not running, or Preferences > System > General > External scripting "
                                 "using is not set to Local (Studio only).")
    return app


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _fps_text(fps: float) -> str:
    return f"{fps:.3f}".rstrip("0").rstrip(".")


class Builder:
    def __init__(self, resolve, cfg: dict, log=print):
        self.resolve, self.cfg, self.log = resolve, cfg, log
        self.project = resolve.GetProjectManager().GetCurrentProject()
        if self.project is None:
            raise ResolveUnavailable("Open or create a project in Resolve first.")
        self.mp = self.project.GetMediaPool()
        self.items: dict[str, object] = {}
        self.issues: list[str] = []
        self.shift = 0                      # frames added to every recordFrame, learned from the first placement
        self.calibrated = False

    # ------------------------------------------------------------ small helpers
    def warn(self, msg: str) -> None:
        self.issues.append(msg)
        self.log(f"  ! {msg}")

    def check_version(self) -> None:
        try:
            v = [int(x) for x in self.resolve.GetVersion()[:2]]
            if v < [18, 5]:
                self.warn(f"Resolve {v[0]}.{v[1]}: 18.5 or newer is needed to place clips on tracks at exact frames")
        except Exception:
            pass

    def set_props(self, item, **props) -> None:
        for k, v in props.items():
            try:
                ok = item.SetProperty(k, v)
            except Exception:
                ok = False
            if not ok:
                self.warn(f"Resolve refused {k}={v} on {self._name(item)}")

    @staticmethod
    def _name(item) -> str:
        try:
            return item.GetName()
        except Exception:
            return "clip"

    # ------------------------------------------------------------ project / media pool
    def next_batch_name(self) -> str:
        best = 0
        for i in range(1, int(self.project.GetTimelineCount() or 0) + 1):
            m = re.match(r"^BATCH (\d+)", self.project.GetTimelineByIndex(i).GetName())
            if m:
                best = max(best, int(m.group(1)))
        return f"BATCH {best + 1}"

    def use_folder(self, name: str):
        root = self.mp.GetRootFolder()
        folder = next((f for f in (root.GetSubFolderList() or []) if f.GetName() == name), None)
        folder = folder or self.mp.AddSubFolder(root, name)
        if folder:
            self.mp.SetCurrentFolder(folder)
        return folder

    def import_files(self, paths: list[str]) -> None:
        paths = [p for p in dict.fromkeys(paths) if p]
        folder = self.mp.GetCurrentFolder()
        for clip in (folder.GetClipList() if folder else None) or []:
            fp = clip.GetClipProperty("File Path")
            if fp:
                self.items.setdefault(_key(fp), clip)
        missing = [p for p in paths if _key(p) not in self.items and os.path.isfile(p)]
        if missing:
            for it in self.mp.ImportMedia(missing) or []:
                fp = it.GetClipProperty("File Path")
                if fp:
                    self.items[_key(fp)] = it
        for p in paths:
            if _key(p) not in self.items:
                self.warn(f"Resolve did not import {p}")

    def clip_frames(self, item, fps: float) -> int:
        """Length of an audio/still clip in timeline frames, from its Duration timecode."""
        d = str(item.GetClipProperty("Duration") or "")
        m = re.match(r"^(\d+):(\d+):(\d+)[:;](\d+)$", d)
        if not m:
            return 0
        h, mi, s, f = (int(x) for x in m.groups())
        return ((h * 60 + mi) * 60 + s) * int(round(timebase.exact(fps))) + f

    def rec(self, frame: int) -> int:
        return frame + self.shift

    # ------------------------------------------------------------ timelines
    def new_timeline(self, name: str, t: dict):
        names = {self.project.GetTimelineByIndex(i).GetName() for i in range(1, int(self.project.GetTimelineCount() or 0) + 1)}
        unique, n = name, 1
        while unique in names:                      # never replace a timeline you may have been editing
            n += 1
            unique = f"{name} v{n}"
        if int(self.project.GetTimelineCount() or 0) == 0:
            for k, v in (("timelineResolutionWidth", str(int(t["width"]))), ("timelineResolutionHeight", str(int(t["height"]))),
                         ("timelineFrameRate", _fps_text(t["fps"]))):
                self.project.SetSetting(k, v)
        tl = self.mp.CreateEmptyTimeline(unique)
        if not tl:
            raise RuntimeError(f"Resolve would not create the timeline {unique}")
        self.project.SetCurrentTimeline(tl)
        tl.SetSetting("useCustomSettings", "1")
        want = {"timelineResolutionWidth": str(int(t["width"])), "timelineResolutionHeight": str(int(t["height"])),
                "timelineOutputResolutionWidth": str(int(t["width"])),
                "timelineOutputResolutionHeight": str(int(t["height"])), "timelineFrameRate": _fps_text(t["fps"])}
        for k, v in want.items():
            tl.SetSetting(k, v)
        for k, v in want.items():
            got = str(tl.GetSetting(k))
            try:
                same = abs(float(got) - float(v)) < 0.01
            except ValueError:
                same = got == v
            if not same:
                self.warn(f"{unique}: {k} is {got}, wanted {v}. Set the Project Settings (Timeline resolution and "
                          "frame rate) to this before building.")
        try:
            tl.SetStartTimecode("00:00:00:00")
        except Exception:
            pass
        return tl

    def ensure_tracks(self, tl, kind: str, n: int) -> None:
        while int(tl.GetTrackCount(kind) or 0) < n:
            if not tl.AddTrack(kind):
                self.warn(f"could not add a {kind} track")
                return

    def append(self, variants: list[list[dict]]) -> list:
        """Try each way of writing the same placement until Resolve accepts one."""
        for infos in variants:
            if not infos:
                continue
            res = self.mp.AppendToTimeline(infos)
            if res:
                return list(res)
        return []

    # ------------------------------------------------------------ one short
    def build_short(self, tl, short: Short, tl_cfg: dict, offset: int = 0) -> dict:
        fps = tl_cfg["fps"]
        item_of = lambda p: self.items.get(_key(p))
        bounds = short.frame_bounds(fps)
        has_audio = short.audio.get("mic", "single") != "none"
        use_voice = bool(short.voice) and item_of(short.voice) is not None

        # --- V1: the pieces back to back
        src = item_of(short.source)
        live = [(k, p) for k, p in enumerate(short.pieces) if bounds[k + 1] > bounds[k]]

        def infos_v1() -> tuple[list[dict], list[dict]]:
            video, sound = [], []
            for k, p in live:
                n = bounds[k + 1] - bounds[k]
                baked = bool(p.media) and item_of(p.media) is not None
                clip = item_of(p.media) if baked else src
                start = timebase.to_frame(p.media_in if baked else p.src_in, fps if baked else short.fps)
                count = n if baked else max(1, int(round(n * timebase.exact(short.fps) / timebase.exact(fps))))
                info = {"mediaPoolItem": clip, "startFrame": start, "endFrame": start + count - 1, "trackIndex": 1,
                        "recordFrame": self.rec(offset + bounds[k])}
                if use_voice or baked or not has_audio:
                    info["mediaType"] = VIDEO
                video.append(info)
                if baked and not use_voice and has_audio:      # baked picture is video only: keep the original sound
                    s2 = timebase.to_frame(p.src_in, short.fps)
                    sound.append({"mediaPoolItem": src, "startFrame": s2, "endFrame": s2 + count_src(n) - 1,
                                  "mediaType": AUDIO, "trackIndex": 1, "recordFrame": self.rec(offset + bounds[k])})
            return video, sound

        def count_src(n: int) -> int:
            return max(1, int(round(n * timebase.exact(short.fps) / timebase.exact(fps))))

        def place_v1():
            video, sound = infos_v1()
            placed = self.append([video, [{k: v for k, v in i.items() if k not in ("trackIndex", "recordFrame")}
                                          for i in video]])
            if placed and sound:
                self.append([sound])
            return placed

        if not place_v1():
            raise RuntimeError(f"{short.name}: Resolve did not place the pieces")
        if not self.calibrated and not offset:
            self.calibrated = True
            first = (tl.GetItemListInTrack("video", 1) or [None])[0]
            d = (int(first.GetStart()) - int(tl.GetStartFrame() or 0) - bounds[0]) if first is not None else 0
            if d:       # recordFrame is not what we assumed (absolute vs relative): put everything back, shifted
                every = list(tl.GetItemListInTrack("video", 1) or []) + list(tl.GetItemListInTrack("audio", 1) or [])
                if tl.DeleteClips(every, False):
                    self.shift = -d
                    self.log(f"  recordFrame is offset by {d} frames in this Resolve: compensating")
                    place_v1()
                else:
                    self.warn(f"the first clip landed {d} frames from where it was asked to: check the timeline "
                              "start timecode")
        v1 = self.track_items(tl, "video", 1, offset, bounds[-1])
        for item, (k, p) in zip(v1, live):
            if not p.media:
                self.set_props(item, Scaling=1, ZoomX=p.zoom, ZoomY=p.zoom, Pan=p.pan, Tilt=p.tilt)

        # --- dialogue and music
        if use_voice:
            self.audio_clip(tl, short.voice, 1, offset, bounds[-1], fps, short.name)
        for o in short.overlays:
            if o.kind == "music" and item_of(o.file) is not None:
                self.audio_clip(tl, o.file, o.track, offset + int(round(o.t_in * fps)),
                                int(round(o.t_out * fps)) - int(round(o.t_in * fps)), fps, o.label)

        # --- picture overlays: one call per track
        by_track: dict[int, list] = {}
        for o in short.overlays:
            if o.kind != "music" and item_of(o.file) is not None:
                by_track.setdefault(o.track, []).append(o)
        for track, ovs in sorted(by_track.items()):
            self.ensure_tracks(tl, "video", track)
            infos = []
            for o in ovs:
                f_in = int(round(o.t_in * fps))
                n = max(1, int(round(o.t_out * fps)) - f_in)
                start = timebase.to_frame(o.props.get("src_in", 0.0), o.props.get("fps", fps)) if o.kind == "broll" else 0
                count = n if o.kind != "broll" else max(1, int(round(n * timebase.exact(o.props.get("fps", fps)) / timebase.exact(fps))))
                infos.append({"mediaPoolItem": item_of(o.file), "startFrame": start, "endFrame": start + count - 1,
                              "mediaType": VIDEO, "trackIndex": track, "recordFrame": self.rec(offset + f_in)})
            self.append([infos])
            for item, o in zip(self.track_items(tl, "video", track, offset, bounds[-1]), sorted(ovs, key=lambda o: o.t_in)):
                if o.props.get("scaling") is not None and o.kind in ("broll",):
                    self.set_props(item, Scaling=o.props["scaling"])

        # --- captions and markers
        if short.subtitles and os.path.isfile(short.subtitles) and item_of(short.subtitles) is not None:
            self.subtitles(tl, short.subtitles, offset, bounds[-1], fps)
        for m in short.markers:
            color = m.color if m.color in COLORS else "Blue"
            if not tl.AddMarker(offset + int(round(m.t * fps)), color, m.name, m.note, 1, ""):
                self.warn(f"{short.name}: marker '{m.name}' at {m.t:.1f}s was not added")
        return {"v1": len(v1), "pieces": len(short.pieces)}

    def cuts_only(self, short: Short, tl_cfg: dict, name: str) -> str:
        """Last resort when clips cannot be appended: import the cut list as an EDL so there is at least a timeline."""
        folder = short.work or os.path.expanduser("~")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{short.name}.edl")
        with open(path, "w") as f:
            f.write(export.edl(short, tl_cfg["fps"], name))
        try:
            tl = self.mp.ImportTimelineFromFile(path, {"timelineName": f"{name} (cuts only)", "importSourceClips": False})
        except Exception:
            tl = None
        return tl.GetName() if tl else ""

    def track_items(self, tl, kind: str, track: int, offset: int, length: int) -> list:
        base = int(tl.GetStartFrame() or 0)
        out = []
        for it in tl.GetItemListInTrack(kind, track) or []:
            rel = int(it.GetStart()) - base
            if offset <= rel < offset + max(length, 1):
                out.append(it)
        return sorted(out, key=lambda i: i.GetStart())

    def audio_clip(self, tl, path: str, track: int, rec_frame: int, n: int, fps: float, label: str) -> None:
        item = self.items[_key(path)]
        total = self.clip_frames(item, fps)
        n = min(n, total) if total else n
        self.ensure_tracks(tl, "audio", track)
        base = {"mediaPoolItem": item, "mediaType": AUDIO, "trackIndex": track, "recordFrame": self.rec(rec_frame)}
        full = dict(base, startFrame=0, endFrame=max(n - 1, 0))
        res = self.append([[full], [base], [{"mediaPoolItem": item, "mediaType": AUDIO}]])
        if not res:
            self.warn(f"Resolve did not place audio '{label}' on A{track}: drag {path} onto the timeline")

    def subtitles(self, tl, path: str, offset: int, length: int, fps: float) -> None:
        item = self.items[_key(path)]
        if int(tl.GetTrackCount("subtitle") or 0) < 1:
            try:
                tl.AddTrack("subtitle")
            except Exception:
                pass
        base = {"mediaPoolItem": item, "trackIndex": 1, "recordFrame": self.rec(offset)}
        self.append([[dict(base, startFrame=0, endFrame=max(length - 1, 0))], [base], [{"mediaPoolItem": item}]])
        if not (tl.GetItemListInTrack("subtitle", 1) or []):
            self.warn(f"captions were not placed: drag {path} onto the timeline (or File > Import > Subtitle)")

    def verify(self, tl, short: Short, tl_cfg: dict, offset: int = 0) -> list[str]:
        """Read the timeline back and compare with the plan (port of every skill's 'verify' step)."""
        fps = tl_cfg["fps"]
        bounds = short.frame_bounds(fps)
        base = int(tl.GetStartFrame() or 0)
        problems = []
        v1 = self.track_items(tl, "video", 1, offset, bounds[-1])
        want = [(bounds[k], bounds[k + 1] - bounds[k]) for k in range(len(short.pieces)) if bounds[k + 1] > bounds[k]]
        if len(v1) != len(want):
            problems.append(f"{short.name}: V1 has {len(v1)} clips, the plan has {len(want)}")
        for it, (st, n) in zip(v1, want):
            rel = int(it.GetStart()) - base - offset
            dur = int(it.GetDuration())
            if rel != st or abs(dur - n) > 1:
                problems.append(f"{short.name}: V1 clip at {rel} (len {dur}), wanted {st} (len {n})")
                break
        live = [p for k, p in enumerate(short.pieces) if bounds[k + 1] > bounds[k]]
        for it, p in zip(v1, live):
            try:
                got = int(it.GetLeftOffset())
            except Exception:
                break
            if not p.media and abs(got - timebase.to_frame(p.src_in, short.fps)) > 1:
                problems.append(f"{short.name}: a clip starts {got} frames into the footage, the plan says "
                                f"{timebase.to_frame(p.src_in, short.fps)}: Resolve may count from the clip's timecode")
                break
        for o in short.overlays:
            if o.kind == "music":
                continue
            items = self.track_items(tl, "video", o.track, offset, bounds[-1])
            f_in = offset + int(round(o.t_in * fps))
            if not any(abs(int(i.GetStart()) - base - f_in) <= 1 for i in items):
                problems.append(f"{short.name}: {o.kind} '{o.label}' is not on V{o.track} at {o.t_in:.1f}s")
        return problems


# ------------------------------------------------------------------ whole plan
def media_paths(shorts: list[Short]) -> list[str]:
    out = []
    for s in shorts:
        out.append(s.source)
        out += [p.media for p in s.pieces if p.media]
        out += [o.file for o in s.overlays]
        out += [s.voice, s.subtitles]
    return [p for p in out if p]


def apply(plan: EditPlan, cfg: dict, shorts: list[str] | None = None, layout: str = "per-short", resolve=None,
          log=print) -> dict:
    resolve = resolve or connect()
    b = Builder(resolve, cfg, log)
    b.check_version()
    tl_cfg = dict(plan.timeline or cfg["timeline"])
    batch = plan.batch.get("name") or b.next_batch_name()
    plan.batch["name"] = batch
    sel = [s for s in plan.shorts if (not shorts or s.name in shorts) and s.enabled and s.pieces]
    if not sel:
        raise SystemExit("nothing to build: no enabled shorts with pieces")
    b.use_folder(batch)
    b.import_files(media_paths(sel))
    problems, made = [], []
    if layout == "batch":
        fps = tl_cfg["fps"]
        gap = int(round(tl_cfg.get("gap_s", 60.06) * timebase.exact(fps)))
        offsets, pos = [], 0
        for s in sel:
            offsets.append(pos)
            pos += s.frame_bounds(fps)[-1] + gap
        tl = b.new_timeline(batch, tl_cfg)
        for s, off in zip(sel, offsets):
            log(f"== {s.name}")
            b.build_short(tl, s, tl_cfg, off)
            problems += b.verify(tl, s, tl_cfg, off)
        made.append(batch)
    else:
        for s in sel:
            log(f"== {s.name}")
            tl = b.new_timeline(f"{batch} - {s.name}", tl_cfg)
            try:
                b.build_short(tl, s, tl_cfg, 0)
            except RuntimeError as ex:
                b.warn(str(ex))
                alt = b.cuts_only(s, tl_cfg, f"{batch} - {s.name}")
                b.warn(f"{s.name}: imported the cut list as '{alt}' instead (no framing, overlays or audio)" if alt else
                       f"{s.name}: could not import the cut list either; import {s.name}.edl by hand "
                       "(autoedit apply --no-resolve)")
                made.append(alt or tl.GetName())
                continue
            problems += b.verify(tl, s, tl_cfg, 0)
            made.append(tl.GetName())
    for p in problems:
        b.warn(p)
    log(f"built {len(made)} timeline(s): {', '.join(made)}" + (f"  ({len(b.issues)} issue(s) above)" if b.issues else "  (verified)"))
    return {"timelines": made, "issues": b.issues}


def run_from_menu(config_path: str = "") -> None:
    """Entry point of the launcher in Workspace > Scripts."""
    import glob
    from . import config
    cfg = config.load(config_path or None)
    found = glob.glob(os.path.join(os.path.expanduser(cfg["paths"]["work"]), "*", "edit.json"))
    if not found:
        print("No edit plan found. Run `autoedit run --dry-run` in a terminal first.")
        return
    path = max(found, key=os.path.getmtime)
    print("building from", path)
    plan = EditPlan.load(path)
    apply(plan, cfg, layout=cfg["timeline"]["layout"])
    plan.save(path)
