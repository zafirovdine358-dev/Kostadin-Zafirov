import numpy as np

from autoedit import faces as fc
from autoedit import turns


def F(cx, w=200, y=600, sim=None, mouth=0.0):
    return fc.Face(cx - w / 2, y, w, w * 1.2, cx, y + 0.4 * w, 1.0, sim, mouth)


def test_tracks_follow_two_people_and_keep_ids():
    samples = [fc.Sample(i * 0.2, [F(1200 + i * 3), F(2600 - i * 2)]) for i in range(20)]
    assert fc.build_tracks(samples, 3840) == 2
    assert {f.track for s in samples for f in s.faces if f.cx < 2000} == {0}
    assert {f.track for s in samples for f in s.faces if f.cx > 2000} == {1}


def test_track_survives_short_dropout_but_not_a_teleport():
    samples = [fc.Sample(0.0, [F(1000)]), fc.Sample(0.2, []), fc.Sample(0.4, [F(1010)]), fc.Sample(0.6, [F(3000)])]
    assert fc.build_tracks(samples, 3840) == 2
    assert samples[0].faces[0].track == samples[2].faces[0].track != samples[3].faces[0].track


def test_roles_by_host_similarity():
    samples = [fc.Sample(i * 0.2, [F(1200, sim=0.7), F(2600, sim=0.1)]) for i in range(20)]
    fc.build_tracks(samples, 3840)
    roles = fc.assign_roles(samples, [])
    fc.label_samples(samples, roles)
    assert samples[0].H.cx == 1200 and samples[0].G.cx == 2600


def test_roles_by_mouth_motion_when_no_embeddings():
    T = [turns.Turn(0, 2, "H"), turns.Turn(2, 4, "G")]
    samples = []
    for i in range(20):
        t = i * 0.2
        left_talks = t < 2
        samples.append(fc.Sample(t, [F(1200, mouth=0.3 if left_talks else 0.01), F(2600, mouth=0.01 if left_talks else 0.3)]))
    fc.build_tracks(samples, 3840)
    roles = fc.assign_roles(samples, T)
    fc.label_samples(samples, roles)
    assert samples[0].H.cx == 1200 and samples[0].G.cx == 2600


def test_mouth_motion_ignores_global_shake_and_sees_the_mouth():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (400, 400)).astype(np.uint8)
    f = fc.Face(100, 100, 200, 240, 200, 160)
    still = fc.mouth_motion(base, base, f)
    shake = fc.mouth_motion(base, np.clip(base.astype(int) + 20, 0, 255).astype(np.uint8), f)
    talk = base.copy()
    talk[100 + int(0.7 * 240): 100 + 240, 140:260] = 255 - talk[100 + int(0.7 * 240): 100 + 240, 140:260]
    talking = fc.mouth_motion(base, talk, f)
    assert still == 0.0 and shake < 0.02 and talking > 0.2


def test_host_centroid_needs_a_recurring_face():
    rng = np.random.default_rng(0)
    host = rng.normal(size=128)
    emb = {}
    for k in range(5):
        guest = rng.normal(size=128)
        emb[f"S{k}"] = np.vstack([host + rng.normal(0, 0.1, 128), guest])
    c = fc.host_centroid(emb, min_shorts=3)
    assert c is not None and abs(c @ (host / np.linalg.norm(host))) > 0.95
    assert fc.host_centroid({"only": emb["S0"]}) is None
