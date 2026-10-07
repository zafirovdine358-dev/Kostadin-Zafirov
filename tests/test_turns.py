import numpy as np

from autoedit import turns
from autoedit.analysis import Word

SR = 16000


def W(s, e, t):
    return Word(s, e, t)


def test_sentences_split_on_punctuation_and_gaps():
    ws = [W(0, .3, "Are"), W(.3, .6, "you"), W(.6, 1.0, "following?"), W(1.2, 1.5, "Yes"),
          W(1.5, 1.9, "I"), W(3.0, 3.4, "do.")]
    ss = turns.sentences(ws)
    assert [(s.first, s.last) for s in ss] == [(0, 2), (3, 4), (5, 5)]
    assert ss[0].text == "Are you following?"


def voiced(f0, dur, formant):
    """Crude 'voice': harmonic stack at f0 shaped around a formant."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for h in range(1, int(4000 / f0)):
        y += np.sin(2 * np.pi * f0 * h * t) * np.exp(-((f0 * h - formant) / 600.0) ** 2)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def test_voice_labels_separate_two_voices_and_pick_host_by_cues():
    low, high = voiced(110, 2.0, 500), voiced(210, 2.0, 2200)
    audio = np.concatenate([low, high, low, high, low])
    sents = [turns.Sentence(i * 2.0, (i + 1) * 2.0, i, i, t) for i, t in enumerate(
        ["Are you following based on tiktok?", "Yes I do.", "Let me recommend the sea salt spray.",
         "Okay thanks.", "Try the texture powder too."])]
    labels = turns.voice_labels(sents, audio)
    assert labels[0] == labels[2] == labels[4]
    assert labels[1] == labels[3] and labels[0] != labels[1]
    assert labels[0] == "H"       # the low voice carries the host cues


def test_mic_labels_use_louder_channel():
    t = np.arange(SR * 4) / SR
    a1 = np.where((t < 2), 0.3, 0.01) * np.sin(2 * np.pi * 200 * t)
    a2 = np.where((t >= 2), 0.3, 0.01) * np.sin(2 * np.pi * 300 * t)
    sents = [turns.Sentence(0.0, 1.9, 0, 0), turns.Sentence(2.1, 3.9, 1, 1)]
    out = turns.mic_labels(sents, {"a1": a1.astype(np.float32), "a2": a2.astype(np.float32)}, {"A1": "G", "A2": "H"})
    assert [l for l, _ in out] == ["G", "H"] and all(m > 10 for _, m in out)


def test_build_turns_absorbs_backchannels():
    ss = [turns.Sentence(0, 4, 0, 0), turns.Sentence(4.2, 4.5, 1, 1), turns.Sentence(4.6, 8, 2, 2),
          turns.Sentence(8.2, 10, 3, 3)]
    T = turns.build_turns(ss, ["H", "G", "H", "G"])
    assert [(t.spk, round(t.s, 1), round(t.e, 1)) for t in T] == [("H", 0, 8), ("G", 8.2, 10)]


def test_kmeans2_and_speaker_at():
    X = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5], [0, 0.1]])
    lab = turns.kmeans2(X)
    assert lab[0] == lab[1] == lab[4] != lab[2] == lab[3]
    T = [turns.Turn(0, 4, "H"), turns.Turn(4.2, 9, "G")]
    assert turns.speaker_at(T, 1) == "H" and turns.speaker_at(T, 4.15) == "G" and turns.speaker_at(T, 9) == "G"


def test_one_voice_is_not_split_into_two_speakers():
    rng = np.random.default_rng(0)
    base = voiced(130, 2.0, 700)
    audio = np.concatenate([base * (0.8 + 0.2 * rng.random()) + 0.003 * rng.standard_normal(len(base)).astype(np.float32)
                            for _ in range(6)])
    sents = [turns.Sentence(i * 2.0, (i + 1) * 2.0, i, i, "Try the sea salt spray." if i % 2 else "Okay thanks.")
             for i in range(6)]
    assert turns.voice_labels(sents, audio.astype(np.float32)) == ["H"] * 6


def test_mics_that_hear_everyone_keep_the_previous_speaker():
    t = np.arange(SR * 6) / SR
    loud = np.where(t < 2, 0.3, 0.01) * np.sin(2 * np.pi * 200 * t)
    bleed = np.where(t < 2, 0.01, 0.3) * np.sin(2 * np.pi * 300 * t)
    near = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.18 * np.sin(2 * np.pi * 300 * t)     # both mics equally loud here
    a1 = np.where(t < 4, loud, near).astype(np.float32)
    a2 = np.where(t < 4, bleed, near * 0.95).astype(np.float32)
    sents = [turns.Sentence(0.0, 1.9, 0, 0), turns.Sentence(2.1, 3.9, 1, 1), turns.Sentence(4.1, 5.9, 2, 2)]
    out = [l for l, _ in turns.mic_labels(sents, {"a1": a1, "a2": a2}, {"A1": "G", "A2": "H"})]
    assert out == ["G", "H", "H"]


def test_two_voices_decision_uses_pitch_or_tone_not_noise():
    def feat(shape, f0):
        return np.concatenate([shape, [np.log(f0 + 1) * 3]])
    flat = np.zeros(20)
    bright = np.linspace(-4, 4, 20)
    same = np.array([feat(flat + 0.05 * i, 120) for i in range(6)])
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert not turns.two_voices(same, lab)                                       # tiny differences are noise
    pitch = np.array([feat(flat, 120)] * 3 + [feat(flat, 200)] * 3)
    assert turns.two_voices(pitch, lab)                                          # a clearly higher voice
    tone = np.array([feat(flat, 120)] * 3 + [feat(bright, 125)] * 3)
    assert turns.two_voices(tone, lab)                                           # same pitch, very different tone
    assert not turns.two_voices(pitch, np.array([0, 0, 0, 0, 0, 1]))             # one stray sentence is not a speaker
