"""Per-short audio analysis cache: voice wav, energy track and word timings (two Whisper passes)."""
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from . import media, speech
from .plan import Short

SR = 16000
_MODELS: dict = {}


@dataclass
class Word:
    s: float
    e: float
    t: str
    p: float = 1.0
    src: str = "a"          # a = VAD pass, b = no-VAD pass, e = energy island (no text)
    spk: str = ""


@dataclass
class Analysis:
    words: list = field(default_factory=list)
    e: np.ndarray = field(default_factory=lambda: np.zeros(0))
    engine: str = "energy"      # whisper | energy
    duration: float = 0.0
    chan: dict = field(default_factory=dict)     # "a1"/"a2" -> mono float32 at SR (two-mic clips)
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))   # mono mix at SR

    @property
    def has_text(self) -> bool:
        return self.engine == "whisper"


def _channel(short: Short, key: str) -> np.ndarray | None:
    sel = short.audio.get(key)
    if not sel:
        return None
    stream, ch = sel
    n = next((a.channels for a in media.probe(short.source).audio if a.index == stream), 1)
    return media.decode_audio(short.source, stream, rate=SR, channels=n)[:, ch].copy()


def voice_audio(short: Short) -> tuple[np.ndarray, dict]:
    """Mono mix of the live mics (what Whisper and the energy track see) plus each mic on its own.

    Decoding means reading the whole camera file, so the result is cached next to the short's other work files.
    """
    cache = os.path.join(short.work, "audio16k.npz") if short.work else ""
    if cache and os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(short.source):
        z = np.load(cache)
        return z["mix"], {k: z[k] for k in ("a1", "a2") if k in z.files}
    chans = {k: c for k in ("a1", "a2") if (c := _channel(short, k)) is not None}
    if not chans:
        raise media.MediaError(f"{short.name}: no audible audio channel")
    n = min(len(c) for c in chans.values())
    mix = (sum(c[:n] for c in chans.values()) / len(chans)).astype(np.float32)
    chans = {k: c[:n] for k, c in chans.items()}
    if cache:
        os.makedirs(short.work, exist_ok=True)
        np.savez_compressed(cache, mix=mix, **chans)
    return mix, chans


def _model(cfg: dict):
    from faster_whisper import WhisperModel
    name = os.path.expanduser(cfg["models"]["whisper"])
    key = (name, cfg["asr"]["compute_type"])
    if key not in _MODELS:
        _MODELS[key] = WhisperModel(name, device="auto", compute_type=cfg["asr"]["compute_type"])
    return _MODELS[key]


def whisper_words(x: np.ndarray, cfg: dict, vad: bool) -> list[Word]:
    a = cfg["asr"]
    segs, _ = _model(cfg).transcribe(x, language=a["language"], word_timestamps=True, vad_filter=vad,
                                     beam_size=a["beam"], condition_on_previous_text=vad)
    out = []
    for s in segs:
        if not vad and getattr(s, "no_speech_prob", 0.0) >= a["pass2_max_no_speech"]:
            continue
        for w in s.words or []:
            out.append(Word(float(w.start), float(w.end), w.word.strip(), float(w.probability), "a" if vad else "b"))
    return out


def merge_passes(a: list[Word], b: list[Word], e: np.ndarray, min_prob: float = 0.5) -> list[Word]:
    """Add words only the no-VAD pass heard, if there is real sound under them (rough-cut SOP step 2)."""
    out = list(a)
    floor = float(np.percentile(e, 20)) if len(e) else 0.0
    for w in b:
        if w.p < min_prob or w.e - w.s <= 0:
            continue
        seg = e[int(w.s * speech.FPS_E): max(int(w.s * speech.FPS_E) + 1, int(w.e * speech.FPS_E))]
        if not len(seg) or seg.max() < floor + 8:
            continue
        if any(x.s < w.e and x.e > w.s for x in a):
            continue
        out.append(w)
    return sorted(out, key=lambda w: (w.s, w.e))


def have_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401  (only checking that it is installed)
        return True
    except ImportError:
        return False


def load(short: Short, cfg: dict, log=print, need_text: bool = False) -> Analysis:
    """Voice audio, energy and words for a short, cached in its work folder."""
    os.makedirs(short.work, exist_ok=True)
    x, chans = voice_audio(short)
    e = speech.energy_db(x, SR)
    an = Analysis(e=e, duration=len(x) / SR, chan=chans if len(chans) > 1 else {}, audio=x)
    cache = os.path.join(short.work, "words.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            d = json.load(f)
        # an energy-only cache is stale as soon as Whisper is available
        if d["engine"] == "whisper" or not have_whisper():
            an.engine, an.words = d["engine"], [Word(**w) for w in d["words"]]
            short.stats["asr"] = an.engine
            if need_text and not an.has_text:
                raise SystemExit("this stage needs word timings: pip install faster-whisper")
            return an
    words = None
    if have_whisper():
        log(f"  {short.name}: transcribing (two Whisper passes)")
        try:
            a = whisper_words(x, cfg, vad=True)
            b = whisper_words(x, cfg, vad=False) if cfg["asr"]["second_pass"] else []
            words = merge_passes(a, b, e, cfg["asr"]["pass2_min_prob"])
        except Exception as ex:      # no model on this machine, no network, ...
            if need_text:
                raise SystemExit(f"Whisper failed: {ex}\nSet models.whisper to a local faster-whisper folder.") from ex
            log(f"  {short.name}: Whisper unavailable ({type(ex).__name__}), using energy-only speech detection")
    elif need_text:
        raise SystemExit("this stage needs word timings: pip install faster-whisper "
                         "(and set models.whisper for offline machines)")
    else:
        log(f"  {short.name}: faster-whisper not installed, using energy-only speech detection")
    if words is not None:
        an.engine, an.words = "whisper", words
        with open(cache, "w") as f:
            json.dump({"engine": an.engine, "words": [asdict(w) for w in an.words]}, f)
    else:
        an.engine = "energy"
        an.words = [Word(s, t, "", 1.0, "e") for s, t in speech.island_words(e)]
        if not have_whisper():
            with open(cache, "w") as f:
                json.dump({"engine": an.engine, "words": [asdict(w) for w in an.words]}, f)
    short.stats["asr"] = an.engine
    return an
