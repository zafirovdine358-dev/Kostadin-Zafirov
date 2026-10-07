"""Machine check (port of based-setup): what is installed, what is missing, what to do next."""
import importlib
import os
import re
import shutil
import subprocess
import sys

from . import config, media

NEXT_STEPS = [
    ("autoedit ingest", "newest shoot folder -> edit plan (import BASED!)"),
    ("autoedit rough-cut", "cut every stretch where nobody is speaking"),
    ("autoedit camera-angles", "cut on speaker changes, reframe on the speaker"),
    ("autoedit fine-cut", "tighten each short start to finish"),
    ("autoedit subtitles", "word-for-word captions, BASED graphic, product slide-ins"),
    ("autoedit products-behind", "products that slide in from behind the person"),
    ("autoedit broll --short NAME --product NAME", "model B-roll over a product explanation (when asked)"),
    ("autoedit stamps-and-music", "batch stamp and one upbeat track per short"),
    ("autoedit audio-fix", "clean and level the voices"),
    ("autoedit apply", "build the timelines in DaVinci Resolve"),
]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _mod(name: str) -> str | None:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "ok")
    except Exception:
        return None


def _english_only(folder: str) -> bool:
    """Whisper's English-only models have <|endoftext|> at id 50256 (the multilingual ones at 50257)."""
    try:
        with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
            m = re.search(r'"id":\s*(\d+),\s*"content":\s*"<\|endoftext\|>"', f.read(4096))
    except OSError:
        return False
    return bool(m) and int(m.group(1)) == 50256


def _whisper_row(setting: str, found: str | None) -> tuple[str, str, str]:
    w = os.path.expanduser(setting)
    if not os.path.isdir(w):
        hint = (f" Found {found}: run `autoedit init`, or set models.whisper to it." if found else
                " For offline machines set models.whisper to a faster-whisper folder.")
        return WARN, "whisper model", f"'{setting}' is downloaded on first use.{hint}"
    missing = [n for n in ("model.bin", "config.json", "tokenizer.json") if not os.path.isfile(os.path.join(w, n))]
    if not any(os.path.isfile(os.path.join(w, v)) for v in ("vocabulary.txt", "vocabulary.json")):
        missing.append("vocabulary.txt")
    if missing:
        big = " (model.bin is the big file, about 150 MB: copy the whole folder)" if "model.bin" in missing else ""
        return FAIL, "whisper model", f"{w} is missing {', '.join(missing)}{big}"
    return PASS, "whisper model", w + (" (English only)" if _english_only(w) else "")


def _ffmpeg_checks() -> list[tuple[str, str, str]]:
    rows = []
    for tool in (media.FFMPEG, media.FFPROBE):
        path = shutil.which(tool)
        rows.append((PASS if path else FAIL, tool, path or "not on PATH: install ffmpeg (https://ffmpeg.org)"))
    if shutil.which(media.FFMPEG):
        enc = subprocess.run([media.FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        flt = subprocess.run([media.FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        for need, hay, label in (("prores_ks", enc, "ProRes encoder (alpha overlays)"),
                                 ("libx264", enc, "H.264 encoder"),
                                 ("afftdn", flt, "denoise filter"), ("equalizer", flt, "EQ filter"),
                                 ("compand", flt, "expander filter"), ("alimiter", flt, "limiter filter")):
            ok = f" {need} " in hay
            rows.append((PASS if ok else WARN, f"ffmpeg {need}", label if ok else f"missing: {label}"))
    return rows


def run(cfg: dict) -> int:
    rows: list[tuple[str, str, str]] = []
    v = sys.version_info
    rows.append((PASS if v >= (3, 10) else FAIL, "python", f"{v.major}.{v.minor}.{v.micro} (need 3.10+)"))
    rows += _ffmpeg_checks()
    for mod, need, why in (("numpy", FAIL, "required"),
                           ("faster_whisper", WARN, "word timings: captions, fine cut, products"),
                           ("cv2", WARN, "faces: camera angles, products behind"),
                           ("PIL", WARN, "baked product overlays"),
                           ("mediapipe", WARN, "person matte for products behind"),
                           ("anthropic", WARN, "optional Claude passes (llm.enabled)")):
        ver = _mod(mod)
        rows.append((PASS if ver else need, f"python: {mod}", f"{ver}" if ver else f"not installed ({why})"))
    models = cfg["models"]
    found = config.find_models()
    rows.append(_whisper_row(models["whisper"], found.get("whisper")))
    for key, label in (("yunet", "YuNet face detector"), ("sface", "SFace face recognizer (host matching)")):
        p = os.path.expanduser(models[key])
        if p and os.path.isfile(p):
            rows.append((PASS, label, p))
        else:
            hint = f" (found {found[key]}: run `autoedit init`)" if key in found else ""
            rows.append((WARN, label, f"set models.{key} to the .onnx file{hint}"))
    for key in ("based_root", "work"):
        p = os.path.expanduser(cfg["paths"][key])
        if key == "work":
            try:
                os.makedirs(p, exist_ok=True)
                rows.append((PASS, "work folder", p))
            except OSError as e:
                rows.append((FAIL, "work folder", f"{p}: {e}"))
        else:
            rows.append((PASS if os.path.isdir(p) else WARN, "BASED folder", p if os.path.isdir(p) else
                         f"{p} not found (set paths.based_root, or pass --folder to ingest)"))
    for key in ("footage", "music", "products", "stamps", "broll"):
        p = config.path(cfg, key)
        rows.append((PASS if os.path.isdir(p) else WARN, f"folder: {key}", p))
    rows.append((PASS if os.environ.get("ANTHROPIC_API_KEY") else WARN, "ANTHROPIC_API_KEY",
                 "set" if os.environ.get("ANTHROPIC_API_KEY") else
                 "not set (only for --llm and --showing claude; `ant auth login` also works)"))
    try:
        from . import resolve_api
        app = resolve_api.connect()
        rows.append((PASS, "DaVinci Resolve", f"connected, version {'.'.join(str(x) for x in app.GetVersion()[:3])}"))
    except Exception as e:
        rows.append((WARN, "DaVinci Resolve", f"{e}"))
    width = max(len(r[1]) for r in rows)
    for status, name, msg in rows:
        print(f"[{status}] {name.ljust(width)}  {msg}")
    print("\nOrder of work:")
    for cmd, what in NEXT_STEPS:
        print(f"  {cmd.ljust(42)} {what}")
    return 1 if any(r[0] == FAIL for r in rows) else 0
