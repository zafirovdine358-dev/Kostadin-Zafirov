"""The Whisper and YuNet wrappers, run against stand-ins that return what the real libraries return."""
from types import SimpleNamespace

import numpy as np
import pytest

from autoedit import analysis, config, faces as fc

CFG = config.merge(config.DEFAULTS, {})


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, x, **kw):
        self.calls.append(kw)
        w = lambda t, s, e, p=0.9: SimpleNamespace(word=f" {t}", start=s, end=e, probability=p)
        if kw["vad_filter"]:
            segs = [SimpleNamespace(words=[w("Are", 1.0, 1.2), w("you", 1.25, 1.4)], no_speech_prob=0.1)]
        else:
            segs = [SimpleNamespace(words=[w("you", 1.25, 1.4), w("following?", 2.0, 2.5), w("noise", 5.0, 5.2, 0.2)],
                                    no_speech_prob=0.2),
                    SimpleNamespace(words=[w("hallucinated", 6.0, 6.5)], no_speech_prob=0.95)]
        return segs, SimpleNamespace()


def test_whisper_words_and_the_second_pass_merge(monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(analysis, "_model", lambda cfg: fake)
    x = np.zeros(16000 * 8, np.float32)
    a = analysis.whisper_words(x, CFG, vad=True)
    b = analysis.whisper_words(x, CFG, vad=False)
    assert [w.t for w in a] == ["Are", "you"] and a[0].src == "a" and a[0].p == pytest.approx(0.9)
    assert fake.calls[0]["vad_filter"] is True and fake.calls[1]["vad_filter"] is False
    assert fake.calls[1]["condition_on_previous_text"] is False and fake.calls[0]["word_timestamps"] is True
    assert [w.t for w in b] == ["you", "following?", "noise"]            # the no-speech segment is dropped
    e = np.full(400, -45.0)
    e[100:130] = 0.0                                                      # real sound only under 'following?' (2.0-2.6 s)
    merged = analysis.merge_passes(a, b, e)
    assert [w.t for w in merged] == ["Are", "you", "following?"]         # 'you' overlaps, 'noise' is low-prob and silent


class FakeDet:
    def __init__(self, rows):
        self.rows, self.size = rows, None

    def setInputSize(self, s):
        self.size = s

    def detect(self, img):
        assert img.shape[1] == 960
        return None, self.rows


class FakeRec:
    def alignCrop(self, img, row):
        self.row = row
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, aligned):
        return np.array([[1.0, 0.0, 0.0]], np.float32)


def test_yunet_rows_become_source_pixel_faces():
    cv2 = pytest.importorskip("cv2")
    rows = np.array([[100, 50, 60, 80, 115, 70, 145, 70, 130, 90, 120, 110, 140, 110, 0.9],
                     [10, 10, 10, 12, 12, 14, 16, 14, 14, 16, 13, 18, 16, 18, 0.95]], np.float32)
    d = fc.YuNet.__new__(fc.YuNet)
    d.cv2, d.det, d.rec, d.min_w = cv2, FakeDet(rows), FakeRec(), 35.0
    d.host = np.array([1.0, 0.0, 0.0])
    img = np.zeros((1080, 1920, 3), np.uint8)
    faces = d.faces(img, to_src=2.0)                 # frame is half the width of a 4K source
    assert len(faces) == 1                           # the 20 px face is ignored
    f = faces[0]
    assert (f.x, f.y, f.w, f.h) == (400, 200, 240, 320)
    assert (f.ex, f.ey) == (520, 280) and f.score == pytest.approx(0.9)
    assert f.sim == pytest.approx(1.0) and f.emb is not None
    assert d.det.size == (960, 540)
    assert d.rec.row[0] == 200                       # SFace is shown the face in full-frame pixels
