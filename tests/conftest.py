import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_clip(path, dur=4.0, audio="single", w=320, h=240, fps=30):
    """Tiny test clip. audio: single (mono), identical (stereo L=R), two (stereo, different mics), quad, none."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=d={dur}:s={w}x{h}:r={fps}"]
    sine = lambda f: ["-f", "lavfi", "-i", f"sine=f={f}:d={dur}"]
    null = ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={dur}"]
    if audio == "single":
        cmd += sine(440) + ["-map", "0:v", "-map", "1:a"]
    elif audio == "identical":
        cmd += sine(440) + ["-filter_complex", "[1:a]asplit[a][b];[a][b]amerge=inputs=2[o]", "-map", "0:v", "-map", "[o]"]
    elif audio == "two":
        cmd += sine(440) + sine(990) + ["-filter_complex", "[1:a][2:a]amerge=inputs=2[o]", "-map", "0:v", "-map", "[o]"]
    elif audio == "quad":
        cmd += sine(440) + sine(990) + null + null + [
            "-filter_complex", "[1:a][2:a][3:a][4:a]amerge=inputs=4[o]", "-map", "0:v", "-map", "[o]"]
    else:
        cmd += ["-map", "0:v"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def clip_factory(tmp_path):
    def f(name, **kw):
        return make_clip(str(tmp_path / name), **kw)
    return f


def make_speechy_clip(path, bursts, dur, audio_hz=(220,), w=320, h=240, fps=30):
    """Clip whose audio is loud tone bursts on near-silence: stands in for speech.

    bursts: [(start, end), ...] for one channel, or a list of such lists (one per entry of audio_hz).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per = bursts if bursts and isinstance(bursts[0], list) else [bursts] * len(audio_hz)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=d={dur}:s={w}x{h}:r={fps}"]
    for hz, bs in zip(audio_hz, per):
        expr = "+".join(f"between(t,{a},{b})" for a, b in bs) or "0"
        cmd += ["-f", "lavfi", "-i", f"aevalsrc='0.4*sin(2*PI*{hz}*t)*({expr})+0.0005*sin(2*PI*50*t)':s=16000:d={dur}"]
    n = len(audio_hz)
    if n == 1:
        cmd += ["-map", "0:v", "-map", "1:a"]
    else:
        ins = "".join(f"[{i + 1}:a]" for i in range(n))
        cmd += ["-filter_complex", f"{ins}amerge=inputs={n}[o]", "-map", "0:v", "-map", "[o]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True)
    return path
