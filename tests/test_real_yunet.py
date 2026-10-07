"""Runs the real YuNet model (set AUTOEDIT_YUNET to face_detection_yunet_2023mar.onnx) on a real face photo.

Skipped unless the model file is given and scikit-image (for its bundled test photo) and OpenCV are installed.
It checks the whole reframing chain: detect the face in a 4K scene, plan the framing, render what Resolve will show
with that zoom/pan/tilt, find the face again in the rendered picture and see that it sits where it was meant to.
"""
import os
import subprocess

import numpy as np
import pytest

from autoedit import angles, config, faces as fc, media
from autoedit.plan import Piece

MODEL = os.environ.get("AUTOEDIT_YUNET", "")
cv2 = pytest.importorskip("cv2")
skdata = pytest.importorskip("skimage.data")
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL), reason="set AUTOEDIT_YUNET to the YuNet .onnx file")


def scene(tmp_path, face_left=2100, eye_y=700):
    """4K grey scene with the photo scaled so the face is ~210 px wide and the eyes sit at source y=eye_y."""
    big = cv2.resize(cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR), (1200, 1200), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((2160, 3840, 3), 120, np.uint8)
    top = eye_y - 239
    canvas[top:top + 1200, face_left:face_left + 1200] = big
    png = str(tmp_path / "scene.png")
    cv2.imwrite(png, canvas)
    mp4 = str(tmp_path / "scene.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30", "-i", png, "-f", "lavfi", "-i",
                    "sine=f=300:d=3", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-c:a", "aac",
                    "-shortest", mp4], check=True)
    return mp4


def test_the_face_ends_up_where_the_framing_says(tmp_path):
    mp4 = scene(tmp_path)
    info = media.probe(mp4)
    det = fc.YuNet(MODEL)
    samples = fc.sample_faces(mp4, [(0.0, 2.5)], 5.0, info.width, info.height, det)
    assert samples and all(len(s.faces) == 1 for s in samples)
    f = samples[0].faces[0]
    assert abs(f.ex - 2624) < 25 and abs(f.ey - 700) < 25 and 150 < f.w < 280
    fc.build_tracks(samples, info.width)
    fc.label_samples(samples, fc.assign_roles(samples, []))
    cfg = config.merge(config.DEFAULTS, {})
    pieces, marks = angles.frame_pieces([Piece(0, 2.5, "H")], samples, info.width, info.height, cfg, angles.canvas(cfg))
    p = pieces[0]
    assert p.who == "G" and not p.flags and not marks and p.zoom == 1.0
    frame = next(media.grab_framed(mp4, 1.0, 1, 30.0, info.width, info.height, 1080, 1920, p.zoom, p.pan, p.tilt))
    seen = fc.YuNet(MODEL).faces(frame)
    assert len(seen) == 1
    assert abs(seen[0].cx - 540) < 20            # head between the blue guides
    assert abs(seen[0].ey - 570) < 40            # eye line


def test_the_real_model_via_the_config_and_doctor(tmp_path):
    cfg = config.merge(config.DEFAULTS, {"models": {"yunet": MODEL}})
    assert angles.make_detector(cfg) is not None
