"""A stand-in for the DaVinci Resolve scripting API, following the documented behaviour of the calls autoedit uses.

It is only as right as that reading of the docs: it checks our logic (frames, tracks, properties, fallbacks), not
that real Resolve agrees. Knobs let tests simulate the API quirks the builder has to survive.
"""
import os

PROPS = {"Pan", "Tilt", "ZoomX", "ZoomY", "ZoomGang", "RotationAngle", "AnchorPointX", "AnchorPointY", "Pitch", "Yaw",
         "FlipX", "FlipY", "CropLeft", "CropRight", "CropTop", "CropBottom", "CropSoftness", "CropRetain",
         "DynamicZoomEase", "CompositeMode", "Opacity", "Distortion", "RetimeProcess", "MotionEstimation", "Scaling",
         "ResizeFilter"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mxf"}
AUDIO_EXT = {".wav", ".mp3", ".aac"}
STILL_EXT = {".png", ".jpg"}


class MPI:
    def __init__(self, path, frames):
        self.path, self.frames = path, frames
        ext = os.path.splitext(path)[1].lower()
        self.kind = ("video" if ext in VIDEO_EXT else "audio" if ext in AUDIO_EXT else
                     "subtitle" if ext == ".srt" else "still")

    def GetName(self):
        return os.path.basename(self.path)

    def GetClipProperty(self, key=None):
        if key == "File Path":
            return self.path
        if key == "Duration":
            f = self.frames
            return f"{f // 108000:02d}:{f // 1800 % 60:02d}:{f // 30 % 60:02d}:{f % 30:02d}"
        if key == "Type":
            return self.kind
        return ""


class Item:
    def __init__(self, mpi, kind, track, start, dur, src_start):
        self.mpi, self.kind, self.track, self.start, self.dur, self.src_start = mpi, kind, track, start, dur, src_start
        self.props = {}

    def GetName(self):
        return self.mpi.GetName()

    def GetStart(self):
        return self.start

    def GetDuration(self):
        return self.dur

    def GetEnd(self):
        return self.start + self.dur

    def GetMediaPoolItem(self):
        return self.mpi

    def GetLeftOffset(self):
        return self.src_start

    def SetProperty(self, key, value):
        if key not in PROPS:
            return False
        if key == "Scaling" and value not in (0, 1, 2, 3, 4):
            return False
        self.props[key] = value
        return True

    def GetProperty(self, key):
        return self.props.get(key)


class Timeline:
    def __init__(self, name, pool):
        self.name, self.pool = name, pool
        self.settings = {}
        self.start_frame = 108000
        self.tracks = {"video": {1: []}, "audio": {1: []}, "subtitle": {}}
        self.markers = {}

    def GetName(self):
        return self.name

    def SetSetting(self, k, v):
        if self.pool.ignore_timeline_settings and k != "useCustomSettings":
            return False
        self.settings[k] = v
        return True

    def GetSetting(self, k):
        return self.settings.get(k, "")

    def GetStartFrame(self):
        return self.start_frame

    def SetStartTimecode(self, tc):
        if not self.pool.ignore_start_tc:
            self.start_frame = 0
        return True

    def GetTrackCount(self, kind):
        return max(self.tracks[kind], default=0)

    def AddTrack(self, kind, *a):
        if kind == "subtitle" and not self.pool.support_subtitles:
            return False
        self.tracks[kind][self.GetTrackCount(kind) + 1] = []
        return True

    def GetItemListInTrack(self, kind, idx):
        return sorted(self.tracks.get(kind, {}).get(idx, []), key=lambda i: i.start)

    def AddMarker(self, frame, color, name, note, dur, custom):
        if frame in self.markers:
            return False
        self.markers[frame] = (color, name, note, dur)
        return True

    def GetMarkers(self):
        return self.markers

    def DeleteClips(self, items, ripple=False):
        for kind in self.tracks.values():
            for tr in kind.values():
                tr[:] = [i for i in tr if i not in items]
        return True


class Folder:
    def __init__(self, name):
        self.name, self.clips, self.subs = name, [], []

    def GetName(self):
        return self.name

    def GetClipList(self):
        return list(self.clips)

    def GetSubFolderList(self):
        return list(self.subs)


class Pool:
    def __init__(self, project, durations, **knobs):
        self.project, self.durations = project, durations
        self.root = Folder("Master")
        self.cur = self.root
        self.imports = 0
        self.calls = []
        self.ignore_record_frame = knobs.get("ignore_record_frame", False)
        self.record_absolute = knobs.get("record_absolute", False)
        self.ignore_start_tc = knobs.get("ignore_start_tc", False)
        self.support_subtitles = knobs.get("support_subtitles", True)
        self.audio_in_samples = knobs.get("audio_in_samples", False)
        self.ignore_timeline_settings = knobs.get("ignore_timeline_settings", False)
        self.fail_append = knobs.get("fail_append", False)
        self.allow_edl_import = knobs.get("allow_edl_import", True)

    def GetRootFolder(self):
        return self.root

    def GetCurrentFolder(self):
        return self.cur

    def SetCurrentFolder(self, f):
        self.cur = f
        return True

    def AddSubFolder(self, parent, name):
        f = Folder(name)
        parent.subs.append(f)
        return f

    def ImportMedia(self, paths):
        out = []
        for p in paths:
            if not os.path.isfile(p):
                continue
            self.imports += 1
            m = MPI(p, self.durations.get(os.path.basename(p), 150))
            self.cur.clips.append(m)
            out.append(m)
        return out

    def ImportTimelineFromFile(self, path, opts=None):
        self.imported = getattr(self, "imported", []) + [(path, opts)]
        if not self.allow_edl_import:
            return None
        tl = Timeline((opts or {}).get("timelineName", "imported"), self)
        self.project.timelines.append(tl)
        return tl

    def CreateEmptyTimeline(self, name):
        tl = Timeline(name, self)
        self.project.timelines.append(tl)
        return tl

    def AppendToTimeline(self, infos):
        self.calls.append(infos)
        if self.fail_append:
            return []
        tl = self.project.current
        placed = []
        for info in infos:
            if not isinstance(info, dict):
                info = {"mediaPoolItem": info}
            m = info["mediaPoolItem"]
            track = info.get("trackIndex", 1)
            kinds = {1: ["video"], 2: ["audio"]}.get(info.get("mediaType"), ["video", "audio"] if m.kind == "video" else
                                                      ["audio"] if m.kind == "audio" else [m.kind if m.kind != "still" else "video"])
            if m.kind == "subtitle":
                kinds = ["subtitle"]
                if not self.support_subtitles or track not in tl.tracks["subtitle"]:
                    return []
            if "startFrame" in info:
                a, b = info["startFrame"], info["endFrame"]
                limit = m.frames * (48 if (m.kind == "audio" and self.audio_in_samples) else 1)
                if a > b or b >= limit:
                    return []
                dur = b - a + 1
            else:
                a, dur = 0, m.frames
            for kind in kinds:
                tr = tl.tracks[kind].get(track)
                if tr is None:
                    return []
                if "recordFrame" in info and not self.ignore_record_frame:
                    rel = info["recordFrame"] - (tl.start_frame if self.record_absolute else 0)
                else:
                    rel = max((i.start - tl.start_frame + i.dur for i in tr), default=0)
                it = Item(m, kind, track, rel + tl.start_frame, dur, a)
                tr.append(it)
                placed.append(it)
        return placed


class Project:
    def __init__(self, durations, **knobs):
        self.timelines, self.current, self.settings = [], None, {}
        self.pool = Pool(self, durations, **knobs)

    def GetMediaPool(self):
        return self.pool

    def GetTimelineCount(self):
        return len(self.timelines)

    def GetTimelineByIndex(self, i):
        return self.timelines[i - 1]

    def SetCurrentTimeline(self, tl):
        self.current = tl
        return True

    def GetCurrentTimeline(self):
        return self.current

    def SetSetting(self, k, v):
        self.settings[k] = v
        return True

    def GetSetting(self, k):
        return self.settings.get(k, "")


class Manager:
    def __init__(self, project):
        self.project = project

    def GetCurrentProject(self):
        return self.project


class Resolve:
    def __init__(self, durations=None, version=(19, 0, 1, 7, ""), **knobs):
        self.project = Project(durations if durations is not None else {}, **knobs)
        self.version = list(version)

    def GetProjectManager(self):
        return Manager(self.project)

    def GetVersion(self):
        return self.version
