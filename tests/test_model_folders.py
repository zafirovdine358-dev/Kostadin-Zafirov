"""The local model folders the BASED Premiere setup keeps in Documents/Claude Tools."""
import json

import pytest

from autoedit import cli, config, doctor

EN_TOKENIZER = '{"version": "1.0", "added_tokens": [\n {"id": 50256,\n  "content": "<|endoftext|>", "special": true}]}'
MULTI_TOKENIZER = EN_TOKENIZER.replace("50256", "50257")
ALL = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")


def tools(home, whisper_files=ALL, tokenizer=EN_TOKENIZER):
    root = home / "Documents" / "Claude Tools"
    w = root / "whisper-base"
    w.mkdir(parents=True)
    for n in whisper_files:
        (w / n).write_text(tokenizer if n == "tokenizer.json" else "x")
    (root / "models").mkdir()
    for n in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"):
        (root / "models" / n).write_bytes(b"onnx")
    return root


def line(out, name):
    return next(ln for ln in out.splitlines() if f"] {name}" in ln)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_finds_the_models_the_premiere_setup_left_in_claude_tools(home):
    assert config.find_models() == {}
    root = tools(home)
    assert config.find_models() == {"whisper": str(root / "whisper-base"),
                                    "yunet": str(root / "models" / "face_detection_yunet_2023mar.onnx"),
                                    "sface": str(root / "models" / "face_recognition_sface_2021dec.onnx")}


def test_a_whisper_folder_without_model_bin_is_not_found(home):
    tools(home, whisper_files=("config.json", "tokenizer.json", "vocabulary.txt"))      # only the small files
    found = config.find_models()
    assert "whisper" not in found and "yunet" in found


def test_init_fills_in_what_it_finds_but_flags_win(home, capsys):
    root = tools(home)
    out = home / "autoedit.json"
    cli.main(["init", "--out", str(out)])
    models = json.loads(out.read_text())["models"]
    assert models["whisper"] == str(root / "whisper-base") and models["yunet"].endswith("yunet_2023mar.onnx")
    assert models["sface"].endswith("sface_2021dec.onnx") and "found whisper" in capsys.readouterr().out
    out.unlink()
    cli.main(["init", "--out", str(out), "--whisper", "base", "--yunet", "/mine/yunet.onnx"])
    models = json.loads(out.read_text())["models"]
    assert models["whisper"] == "base" and models["yunet"] == "/mine/yunet.onnx"
    assert models["sface"].endswith("sface_2021dec.onnx")


def test_init_leaves_the_defaults_when_nothing_is_there(home, capsys):
    out = home / "autoedit.json"
    cli.main(["init", "--out", str(out)])
    assert json.loads(out.read_text())["models"] == {"whisper": "base", "yunet": "", "sface": "", "rnnoise": ""}
    assert "found" not in capsys.readouterr().out


def test_doctor_names_the_missing_whisper_files(home, capsys):
    w = tools(home, whisper_files=("config.json", "tokenizer.json", "vocabulary.txt")) / "whisper-base"
    doctor.run(config.merge(config.DEFAULTS, {"models": {"whisper": str(w)}}))
    ln = line(capsys.readouterr().out, "whisper model")
    assert ln.startswith("[FAIL]") and "is missing model.bin" in ln and "copy the whole folder" in ln


def test_doctor_passes_a_complete_folder_and_says_when_it_is_english_only(home, capsys):
    w = tools(home) / "whisper-base"
    cfg = config.merge(config.DEFAULTS, {"models": {"whisper": str(w)}})
    doctor.run(cfg)
    ln = line(capsys.readouterr().out, "whisper model")
    assert ln.startswith("[PASS]") and ln.endswith("(English only)")
    (w / "tokenizer.json").write_text(MULTI_TOKENIZER)
    doctor.run(cfg)
    ln = line(capsys.readouterr().out, "whisper model")
    assert ln.startswith("[PASS]") and "English only" not in ln


def test_doctor_points_at_models_it_found_but_that_are_not_set(home, capsys):
    tools(home)
    doctor.run(config.merge(config.DEFAULTS, {}))
    out = capsys.readouterr().out
    for name in ("whisper model", "YuNet", "SFace"):
        ln = line(out, name)
        assert ln.startswith("[WARN]") and "found" in ln.lower() and "autoedit init" in ln
