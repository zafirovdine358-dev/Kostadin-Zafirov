"""The edit plan: one JSON document every stage reads and writes, and `apply` turns into a timeline."""
import json
import os
from dataclasses import asdict, dataclass, field, fields

from . import timebase

PLAN_VERSION = 1

# Order the skills run in. The first three change what is kept; the rest add things on top.
STAGES = ["rough", "angles", "fine", "captions", "products", "behind", "broll", "stamps", "music", "audio"]
CUT_STAGES = STAGES[:3]
# what each stage leaves in Short.stats, so re-running a stage cannot leave stale numbers behind
STAT_KEYS = {"rough": ["rough", "protect"], "angles": ["angles", "turns", "roles"], "fine": ["fine"],
             "captions": ["captions", "caption_list", "mute"], "products": ["products"],
             "behind": ["behind", "behind_src"], "broll": ["broll"], "stamps": ["stamp"], "music": ["music"],
             "audio": ["audio"]}


@dataclass
class Piece:
    """A stretch of source footage on a short's timeline. Pieces play back to back."""
    src_in: float
    src_out: float
    spk: str = ""            # speaker talking at the start: "H" host, "G" guest
    who: str = ""            # whose face is framed
    zoom: float = 1.0        # 1.0 = native pixels (Premiere's 100%)
    pan: float = 0.0         # Resolve Pan, canvas px, positive moves the picture right
    tilt: float = 0.0        # Resolve Tilt, canvas px, positive moves the picture up
    flags: list = field(default_factory=list)   # bad_angle, stab, phone
    note: str = ""
    media: str = ""          # optional baked video (already framed) used instead of the source picture
    media_in: float = 0.0    # seconds into that baked file where this piece starts

    @property
    def dur(self) -> float:
        return self.src_out - self.src_in


@dataclass
class Overlay:
    """Clip placed over the footage. Times are seconds on the short's timeline."""
    kind: str                # product, behind_product, behind_person, broll, stamp, based, music
    file: str
    t_in: float
    t_out: float
    track: int = 0
    props: dict = field(default_factory=dict)   # zoom, pan, tilt, opacity, scaling, src_in, gain_db
    label: str = ""
    stage: str = ""


@dataclass
class Marker:
    t: float
    color: str = "Blue"
    name: str = ""
    note: str = ""
    stage: str = ""


@dataclass
class Short:
    name: str
    source: str
    fps: float = 29.97
    width: int = 3840
    height: int = 2160
    duration: float = 0.0
    audio: dict = field(default_factory=dict)   # mic: single|two, a1/a2: [stream, channel], empty: [...]
    enabled: bool = True
    pieces: list = field(default_factory=list)
    overlays: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    voice: str = ""          # processed dialogue wav, aligned to the timeline from t=0
    subtitles: str = ""
    notes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    snaps: dict = field(default_factory=dict)    # stage -> pieces after that stage (dicts)
    work: str = ""

    def tl_duration(self) -> float:
        return sum(p.dur for p in self.pieces)

    def kept(self) -> list[tuple[float, float]]:
        return [(p.src_in, p.src_out) for p in self.pieces]

    def src_to_tl(self, t: float) -> float | None:
        pos = 0.0
        for p in self.pieces:
            if p.src_in <= t < p.src_out:
                return pos + (t - p.src_in)
            pos += p.dur
        return None

    def tl_to_src(self, t: float) -> float | None:
        pos = 0.0
        for p in self.pieces:
            if pos <= t < pos + p.dur:
                return p.src_in + (t - pos)
            pos += p.dur
        return None

    def frame_bounds(self, fps: float) -> list[int]:
        """Timeline frame where each piece starts, plus the end: cumulative rounding so nothing drifts."""
        fps = timebase.exact(fps)
        out, acc = [0], 0.0
        for p in self.pieces:
            acc += p.dur
            out.append(int(round(acc * fps)))
        return out

    def overlays_have(self, kind: str) -> bool:
        return any(o.kind == kind for o in self.overlays)

    def piece_starts(self) -> list[float]:
        out, pos = [], 0.0
        for p in self.pieces:
            out.append(pos)
            pos += p.dur
        return out

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    # -- stage bookkeeping -------------------------------------------------
    def snapshot(self, stage: str) -> None:
        self.snaps[stage] = [asdict(p) for p in self.pieces]

    def pieces_before(self, stage: str) -> list:
        """Pieces as the previous cut stage left them (or the whole clip)."""
        i = STAGES.index(stage)
        for prev in reversed(CUT_STAGES[:i]):
            if prev in self.snaps:
                return [_build(Piece, d) for d in self.snaps[prev]]
        return [Piece(0.0, self.duration)]

    def clear_stage(self, stage: str) -> None:
        self.overlays = [o for o in self.overlays if o.stage != stage]
        self.markers = [m for m in self.markers if m.stage != stage]
        if stage == "captions":
            self.subtitles = ""
        if stage == "audio":
            self.voice = ""
        for k in STAT_KEYS.get(stage, []):
            self.stats.pop(k, None)

    def clear_from(self, stage: str) -> None:
        """Re-running a stage invalidates everything that came after it."""
        for st in STAGES[STAGES.index(stage):]:
            self.clear_stage(st)
            if st != stage:
                self.snaps.pop(st, None)
        self.notes = [n for n in self.notes if not n.startswith(tuple(f"[{s}]" for s in STAGES[STAGES.index(stage):]))]


@dataclass
class EditPlan:
    batch: dict = field(default_factory=dict)
    timeline: dict = field(default_factory=dict)
    shorts: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    version: int = PLAN_VERSION

    def short(self, name: str) -> Short:
        for s in self.shorts:
            if s.name == name:
                return s
        raise KeyError(name)

    def done(self, stage: str) -> None:
        if stage not in self.stages:
            self.stages.append(stage)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=1)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "EditPlan":
        with open(path) as f:
            d = json.load(f)
        shorts = []
        for s in d.get("shorts", []):
            s = dict(s)
            s["pieces"] = [_build(Piece, p) for p in s.get("pieces", [])]
            s["overlays"] = [_build(Overlay, o) for o in s.get("overlays", [])]
            s["markers"] = [_build(Marker, m) for m in s.get("markers", [])]
            shorts.append(_build(Short, s))
        d["shorts"] = shorts
        return _build(cls, d)


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})
