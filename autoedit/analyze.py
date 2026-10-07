"""Footage analysis via ffmpeg: duration/fps, silence, scene changes, motion."""
import json
import re
import subprocess
from dataclasses import dataclass

Interval = tuple[float, float]


@dataclass
class MediaInfo:
    path: str
    duration: float
    fps: float


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-500:]}")
    return p.stdout + p.stderr


def probe(path: str) -> MediaInfo:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate:format=duration",
                "-of", "json", path])
    data = json.loads(out)
    num, den = data["streams"][0]["r_frame_rate"].split("/")
    return MediaInfo(path, float(data["format"]["duration"]), float(num) / float(den))


def detect_silence(path: str, noise_db: float = -30.0, min_len: float = 0.6) -> list[Interval]:
    """Intervals where audio is below noise_db for at least min_len seconds."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
                "-af", f"silencedetect=noise={noise_db}dB:d={min_len}",
                "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    out = [(max(s, 0.0), e) for s, e in zip(starts, ends)]
    if len(starts) > len(ends):  # silence runs to end of file
        out.append((max(starts[-1], 0.0), float("inf")))
    return out


def detect_scenes(path: str, threshold: float = 0.3) -> list[float]:
    """Timestamps (s) of scene changes."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an",
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-f", "null", "-"])
    return [float(x) for x in re.findall(r"pts_time:([\d.]+)", log)]


def detect_dead_video(path: str, black_min: float = 1.0, freeze_min: float = 2.0) -> list[Interval]:
    """Black frames (lens cap / pocket shots) and frozen video."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an",
                "-vf", f"blackdetect=d={black_min}:pix_th=0.10,freezedetect=n=-60dB:d={freeze_min}",
                "-f", "null", "-"])
    black = [(float(a), float(b)) for a, b in
             re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    fs = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    fe = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    freeze = list(zip(fs, fe))
    return black + freeze


def loudness_profile(path: str, window: float = 1.0) -> list[float]:
    """Mean volume (dB) per `window` seconds - used to rank highlights."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
                "-af", f"asetnsamples=n=48000:p=0,aresample=48000,"
                       f"astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
                "-f", "null", "-"])
    vals = [float(x) for x in re.findall(r"RMS_level=(-?[\d.]+|-inf)", log)
            if x != "-inf"]
    return vals
