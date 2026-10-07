from types import SimpleNamespace

from autoedit import llm, showing

CFG = {"llm": {"max_frames": 10, "frame_height": 120, "model": "m", "max_tokens": 100, "effort": "low", "fallbacks": True}}


def test_runs_merge_and_pad():
    times = [1.0, 1.75, 2.5, 3.25, 4.0]
    out = showing.runs(times, [False, True, True, False, False], 0.0, 6.0)
    assert len(out) == 1 and abs(out[0][0] - (1.75 - 0.375 - 0.15)) < 1e-9 and abs(out[0][1] - (2.5 + 0.375 + 0.15)) < 1e-9


def test_judge_samples_and_budget():
    seen = []

    def ask(jpegs):
        seen.append(len(jpegs))
        return [i == 1 for i in range(len(jpegs))]

    short = SimpleNamespace(name="G1", source="x")
    out = showing.judge(short, [(10.0, 14.0)], CFG, log=lambda *_: None, ask=ask, grab=lambda t: b"jpg")
    assert seen == [5] and len(out) == 1 and 10.0 <= out[0][0] < out[0][1] <= 14.0
    # budget: 10 frames total, second range gets what is left
    seen.clear()
    showing.judge(short, [(0, 6), (10, 18)], CFG, log=lambda *_: None, ask=ask, grab=lambda t: b"jpg")
    assert sum(seen) <= 10


class FakeBeta:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.calls = text, stop, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="text", text=self.text)])


class FakeClient:
    def __init__(self, text, stop="end_turn"):
        self.beta = FakeBeta(text, stop)


def test_showing_frames_parses_and_sends_images():
    c = FakeClient('{"frames":[{"i":0,"showing":false},{"i":1,"showing":true}]}')
    assert llm.showing_frames(CFG, [b"a", b"b"], client=c) == [False, True]
    kw = c.beta.calls[0]
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["extra_body"] == {"fallbacks": "default"}
    assert kw["output_config"]["format"]["type"] == "json_schema" and "thinking" not in kw
    assert sum(1 for b in kw["messages"][0]["content"] if b["type"] == "image") == 2


def test_refusal_and_bad_json_raise():
    for c in (FakeClient("", "refusal"), FakeClient("not json")):
        try:
            llm.showing_frames(CFG, [b"a"], client=c)
        except llm.LLMError:
            continue
        raise AssertionError("expected LLMError")


def test_propose_cuts_validates_indices():
    c = FakeClient('{"hook_word": 3, "end_word": 99, "cuts": [{"start": 4, "end": 6, "reason": "echo"},'
                   '{"start": 8, "end": 50, "reason": "repeat"}], "flags": []}')
    out = llm.propose_cuts(CFG, "[0] a", 10, client=c)
    assert out["hook_word"] == 3 and out["end_word"] is None and out["cuts"] == [{"start": 4, "end": 6, "reason": "echo"}]
