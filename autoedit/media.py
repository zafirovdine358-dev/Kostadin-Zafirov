"""ffmpeg/ffprobe wrappers: probing, audio decoding to numpy, frame grabbing."""
import json
import os
import subprocess
import wave
from dataclasses import dataclass, field

import numpy as np

FFMPEG = os.environ.get("AUTOEDIT_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("AUTOEDIT_FFPROBE", "ffprobe")


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(cmd, capture_output=True, **kw)
    except FileNotFoundError as e:
        raise MediaError(f"{cmd[0]} not found - install ffmpeg and put it on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{os.path.basename(cmd[0])} timed out after {kw.get('timeout')}s") from e
    if p.returncode != 0:
        err = p.stderr.decode(errors="replace") if isinstance(p.stderr, bytes) else p.stderr
        raise MediaError(f"{os.path.basename(cmd[0])} failed: {err[-600:]}")
    return p


@dataclass
class AudioStream:
    index: int          # position among audio streams
    channels: int
    rate: int
    codec: str = ""


@dataclass
class MediaInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str = ""
    rotation: int = 0
    audio: list[AudioStream] = field(default_factory=list)
    timecode: str = ""

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


def _rate(s: str) -> float:
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def probe(path: str) -> MediaInfo:
    p = run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    data = json.loads(p.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio = [AudioStream(i, int(s.get("channels", 1)), int(s.get("sample_rate", 48000)), s.get("codec_name", ""))
             for i, s in enumerate(x for x in data["streams"] if x["codec_type"] == "audio")]
    dur = float(data["format"].get("duration") or (v or {}).get("duration") or 0)
    if v is None:
        if not audio:
            raise MediaError(f"no video or audio stream in {path}")
        return MediaInfo(path, dur, 0.0, 0, 0, "", 0, audio)
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = rot or int(v.get("tags", {}).get("rotate", 0) or 0)
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = _rate(v.get("r_frame_rate", "")) or _rate(v.get("avg_frame_rate", ""))
    tc = v.get("tags", {}).get("timecode", "") or data["format"].get("tags", {}).get("timecode", "")
    return MediaInfo(path, dur, fps, w, h, v.get("codec_name", ""), rot, audio, tc)


def decode_audio(path: str, stream: int = 0, rate: int = 16000, start: float = 0.0,
                 dur: float | None = None, filters: str | None = None,
                 channels: int | None = None) -> np.ndarray:
    """Decode one audio stream to float32 [samples, channels]."""
    if channels is None:
        channels = next((a.channels for a in probe(path).audio if a.index == stream), 1)
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start:.3f}", "-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-map", f"0:a:{stream}", "-vn"]
    if filters:
        cmd += ["-af", filters]
        channels = 1 if "pan=" in filters else channels
    cmd += ["-ar", str(rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    x = np.frombuffer(run(cmd).stdout, dtype=np.float32)
    return x[: len(x) // channels * channels].reshape(-1, channels)


def extract_wav(path: str, out: str, stream: int = 0, channel: int | None = None,
                rate: int = 16000, filters: str | None = None) -> str:
    """Mono 16-bit wav; channel=None averages all channels of the stream."""
    af = ([f"pan=mono|c0=c{channel}"] if channel is not None else []) + ([filters] if filters else [])
    cmd = [FFMPEG, "-y", "-v", "error", "-i", path, "-map", f"0:a:{stream}", "-vn"]
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", out]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run(cmd)
    return out


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit pcm wav into float32 [-1, 1] (mono mixdown) and its rate."""
    with wave.open(path, "rb") as w:
        n, ch, rate, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise MediaError(f"{path}: expected 16-bit wav")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def write_wav(path: str, x: np.ndarray, rate: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def grab_frames(path: str, start: float, dur: float, fps: float, width: int, height: int):
    """Yield (t, BGR uint8 [h, w, 3]) at `fps` from [start, start+dur)."""
    cmd = [FFMPEG, "-v", "error", "-hwaccel", "auto", "-ss", f"{start:.3f}", "-i", path, "-t", f"{dur:.3f}", "-an",
           "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, k = width * height * 3, 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1
    finally:
        p.stdout.close()
        p.wait()


def save_frame(path: str, t: float, out: str, height: int = 360) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run([FFMPEG, "-y", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", path, "-frames:v", "1",
         "-vf", f"scale=-2:{height}", "-q:v", "4", out])
    return out


def frame_jpeg(path: str, t: float, height: int = 360) -> bytes:
    """One JPEG frame at time t, as bytes."""
    return run([FFMPEG, "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", path, "-frames:v", "1",
                "-vf", f"scale=-2:{height}", "-q:v", "4", "-f", "image2pipe", "-c:v", "mjpeg", "-"]).stdout


def has_encoder(name: str) -> bool:
    out = run([FFMPEG, "-hide_banner", "-encoders"]).stdout.decode(errors="replace")
    return f" {name} " in out


class FrameWriter:
    """Streaming raw-frame encoder: ProRes 4444 with alpha, or ProRes 422 HQ (QuickTime Animation / x264 fallbacks)."""

    def __init__(self, width: int, height: int, fps: float, out: str, alpha: bool = False, pix_in: str | None = None):
        prores = has_encoder("prores_ks")
        if alpha:
            codec = (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]
                     if prores else ["-c:v", "qtrle", "-pix_fmt", "argb"])
        else:
            codec = (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
                     if prores else ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"])
        pix_in = pix_in or ("rgba" if alpha else "bgr24")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{width}x{height}",
               "-r", f"{fps:.5f}", "-i", "-", *codec, out]
        self.out = out
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: bytes) -> None:
        try:
            self.p.stdin.write(frame)
        except BrokenPipeError:
            pass

    def close(self) -> str:
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        err = self.p.stderr.read().decode(errors="replace")
        if self.p.wait() != 0:
            raise MediaError(f"ffmpeg encode failed: {err[-400:]}")
        return self.out


def encode_frames(frames, width: int, height: int, fps: float, out: str, alpha: bool = False,
                  pix_in: str | None = None) -> str:
    w = FrameWriter(width, height, fps, out, alpha, pix_in)
    for fr in frames:
        w.write(fr)
    return w.close()


def encode_rgba(frames, width: int, height: int, fps: float, out: str) -> str:
    return encode_frames(frames, width, height, fps, out, alpha=True)


def framed_filter(src_w: int, src_h: int, W: int, H: int, zoom: float, pan: float, tilt: float) -> str:
    """ffmpeg filter that shows exactly what Resolve shows for a Crop-scaled clip with this zoom/pan/tilt."""
    cw, ch = min(int(round(W / zoom)), src_w), min(int(round(H / zoom)), src_h)
    x0 = (src_w * zoom / 2 - W / 2 - pan) / zoom
    y0 = (src_h * zoom / 2 - H / 2 + tilt) / zoom
    cx, cy = int(round(min(max(x0, 0), src_w - cw))), int(round(min(max(y0, 0), src_h - ch)))
    return f"crop={cw}:{ch}:{cx}:{cy},scale={W}:{H}:flags=lanczos"


def grab_framed(path: str, start: float, n: int, fps: float, src_w: int, src_h: int, W: int, H: int,
                zoom: float, pan: float, tilt: float):
    """Yield exactly n BGR frames [H, W, 3] of the framed picture starting at `start` (last frame repeats if short)."""
    vf = f"fps={fps:.5f},{framed_filter(src_w, src_h, W, H, zoom, pan, tilt)}"
    cmd = [FFMPEG, "-v", "error", "-hwaccel", "auto", "-ss", f"{start:.4f}", "-i", path, "-an", "-vf", vf,
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, k, last = W * H * 3, 0, None
    try:
        while k < n:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            k += 1
            yield last
    finally:
        p.stdout.close()
        p.wait()
    while k < n and last is not None:
        k += 1
        yield last


def filter_audio(x: np.ndarray, rate: int, af: str) -> np.ndarray:
    """Run a mono float32 signal through an ffmpeg audio filter chain, in memory."""
    p = run([FFMPEG, "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-", "-af", af,
             "-f", "f32le", "-ar", str(rate), "-ac", "1", "-"], input=np.ascontiguousarray(x, np.float32).tobytes())
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def still_to_video(img: str, dur: float, fps: float, width: int, height: int, out: str, fit: bool = True) -> str:
    """A transparent-background movie of a still at least `dur` long (Resolve stills have a fixed length)."""
    if os.path.isfile(out) and probe(out).duration >= dur - 0.05:
        return out
    dims = run([FFPROBE, "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", img]) \
        .stdout.decode().strip().split(",")
    if len(dims) < 2 or not dims[0].isdigit() or int(dims[0]) == 0 or int(dims[1]) == 0:
        raise MediaError(f"{img} is not a readable image")
    scale = (f"scale={width}:{height}:force_original_aspect_ratio=decrease" if fit
             else f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}")
    vf = f"{scale},format=rgba,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x00000000"
    codec = (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]
             if has_encoder("prores_ks") else ["-c:v", "qtrle", "-pix_fmt", "argb"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run([FFMPEG, "-y", "-v", "error", "-loop", "1", "-framerate", f"{fps:.5f}", "-i", img, "-t", f"{dur:.4f}",
         "-vf", vf, *codec, out], timeout=120 + 12 * dur)
    return out