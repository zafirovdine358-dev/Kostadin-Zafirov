"""Settings. Everything the BASED skills hard-coded lives here so another editor can change it."""
import copy
import json
import os

DEFAULTS = {
    "paths": {
        "based_root": "~/Documents/CLIENT WORK/BASED",
        "footage": "IRL/FOOTAGE/New Vids",
        "music": "IRL/ASSETS/Music",
        "products": "IRL/ASSETS/BASED Products",
        "stamps": "IRL/ASSETS/Stamps",
        "broll": "IRL/ASSETS/B-roll",
        "work": "~/Documents/BASED Auto Edit",
        "based_graphic": "",
    },
    "models": {"whisper": "base", "yunet": "", "sface": "", "rnnoise": ""},
    "timeline": {"width": 1080, "height": 1920, "fps": 29.97, "layout": "per-short", "gap_s": 60.06},
    "ingest": {"date_order": "DMY", "skip_words": ["DONE", "Multi-Interview"],
               "media_ext": [".mp4", ".mov", ".mxf", ".mkv", ".m4v", ".avi"], "empty_db": -60.0},
    "asr": {"language": "en", "compute_type": "int8", "beam": 1, "second_pass": True,
            "pass2_min_prob": 0.5, "pass2_max_no_speech": 0.7},
    "rough": {"pre": 0.15, "post": 0.25, "min_gap": 0.9, "head_min": 0.3, "guard_db": 8.0,
              "guard_quiet": 0.25, "min_cut_frames": 4, "showing": "flag", "review_min": 2.5},
    "angles": {"eye_y": 570.0, "center_x": 540.0, "margin": 3.0,
               "scale_steps": [1.0, 1.2, 1.37, 1.5], "face_px": [190.0, 150.0],
               "static_dx": 90.0, "static_dy": 70.0, "dp_tol": [45, 60, 80, 110],
               "min_turn": 0.5, "cut_lead_frames": 2, "min_cut_gap_frames": 3,
               "sample_fps": 5.0, "host_min_shorts": 10, "chunk_min_s": 1.5,
               "mics": {"A1": "G", "A2": "H"}, "min_cover": 0.35, "tiny_s": 0.2},
    "fine": {"hook_max_lead": 30.0, "min_gap": 0.15, "long_gap": 0.6,
             "tail": 0.15, "tail_q": 0.28, "head_pad": 0.05, "min_remove": 0.12, "veto_s": 0.24, "end_pad": 0.65,
             "hook_patterns": [r"\b(are|do) you (follow|following)\b", r"\bis live\b",
                               r"\bfollow(ing)? (us|based)\b", r"\bbased\b.*\btiktok\b|\btiktok\b.*\bbased\b"],
             "fillers": ["um", "uh", "umm", "uhh", "erm", "hmm", "mm", "mhm"],
             "echo_words": ["really", "okay", "right", "nice", "wow", "cool"],
             "closers": r"^(thank you|thanks|thank you so much|bye|goodbye|see you|take care|have a good \w+|appreciate it|there you go)\b",
             "keep_thanks": 1},
    "captions": {"max_words": 3, "long_chars": 18, "long_word": 8, "min_dur": 0.2, "end_pad": 0.4,
                 "style": {"font": "Europa Grotesk SH Medium", "color": "#FFFFFF", "y": 0.43},
                 "other_brands": ["camila rose", "head & shoulders", "head and shoulders", "old spice"],
                 "asr_fixes": [],
                 "groups": [["sea", "salt", "spray"], ["curl", "refresh", "spray"], ["leave-in", "conditioner"],
                            ["texture", "powder"], ["curl", "cream"], ["curl", "mousse"], ["hair", "clay"],
                            ["body", "wash"], ["body", "lotion"], ["face", "spray"], ["skin", "revival", "spray"],
                            ["skin", "revival"], ["curly", "hair"], ["your", "hair"], ["my", "hair"],
                            ["curl", "gel"], ["under", "eye"]]},
    "products": {"enabled": True, "hold_s": 3.0, "y": 0.7716, "enter_s": 0.5, "exit_s": 0.4,
                 "min_width": 0.15, "repeat_gap_s": 30.0,
                 "behind_min_s": 2.0, "behind_enter_f": 16, "behind_exit_f": 14, "behind_dx": 0.32,
                 "behind_rot": 14.0, "behind_scale": 1.25, "feather": 6.0,
                 "items": {
                     "sea salt spray": {"phrases": ["sea salt spray", "sss"], "image": "SSS SANTAL TILTED V03 1.png", "scale": 47},
                     "curl cream": {"phrases": ["curl cream"], "image": "CC Tilted 1.png", "scale": 32},
                     "skin revival spray": {"phrases": ["skin revival spray", "skin revival", "face spray", "hypochlorous"],
                                            "image": "NEW_Prime_HOCL_Tilted.png", "scale": 60},
                     "pomade": {"phrases": ["pomade"], "image": "Pomade.png", "scale": 12},
                     "hair clay": {"phrases": ["hair clay", "clay"], "image": "Hair Clay.png", "scale": 12},
                     "leave-in conditioner": {"phrases": ["leave-in conditioner", "leave-in"],
                                              "image": "Side_275ml Leave-In Conditioner 1.png", "scale": 43},
                     "texture powder": {"phrases": ["texture powder"], "image": "CopyPasta_1783543028601.png", "scale": 38},
                     "curl mousse": {"phrases": ["curl mousse"], "image": "Curl Mousse (from webp).png", "scale": 29},
                     "curl refresh spray": {"phrases": ["curl refresh spray", "curl refresh"],
                                            "image": "Curl Refresh Spray (from webp).png", "scale": 29},
                     "shampoo": {"phrases": ["shampoo"], "image": "Shampoo_Tilted_Brighter.png", "scale": 53},
                     "conditioner": {"phrases": ["conditioner"], "image": "Condtioner_Tilted.png", "scale": 53},
                     "body lotion": {"phrases": ["body lotion"], "image": "CopyPasta_1774873645954.png", "scale": 60},
                     "body wash": {"phrases": ["body wash"], "image": "CopyPasta_1777900450965.png", "scale": 55},
                 }},
    "broll": {"cut_min_s": 1.0, "cut_max_s": 2.0, "max_s": 8.0},
    "music": {"enabled": True, "prefix": "ES_", "no_repeat": 10, "gain_db": -18.0, "first5_db": -13.0,
              "ban": ["nintendo", "mario", "delfino", "bubblaine", "donk city", "steam gardens",
                      "trailer", "cinematic", "tension", "suspense"],
              "log": "music_log.json"},
    "stamps": {"enabled": True, "file": "", "tail_s": 30.0, "long_short_s": 60.0},
    "audio": {"enabled": True, "target_rms": -18.0, "ceiling_db": -3.0, "hp_hz": 90.0,
              "denoise_nr": 10.0, "expander_db": 9.0, "expander_ratio": 2.0, "eq_lo": -12.0,
              "eq_hi": 4.0, "crossfade_frames": 3, "reference": "", "mode": "render"},
    "llm": {"enabled": False, "model": "claude-opus-5-5", "max_tokens": 16000, "effort": "medium",
            "max_frames": 48, "frame_height": 360, "fallbacks": True},
    "tracks": {"footage": 1, "behind_product": 2, "behind_person": 3, "product": 4,
               "broll": 5, "stamp": 6, "based": 7, "voice": 1, "music": 3},   # video tracks V1.., audio tracks A1..
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def search_paths(explicit: str | None = None) -> list[str]:
    paths = [explicit, os.environ.get("AUTOEDIT_CONFIG"), "autoedit.json",
             os.path.expanduser("~/.config/autoedit/config.json")]
    return [os.path.expanduser(p) for p in paths if p]


def load(explicit: str | None = None) -> dict:
    for p in search_paths(explicit):
        if os.path.isfile(p):
            with open(p) as f:
                cfg = merge(DEFAULTS, json.load(f))
            cfg["_path"] = os.path.abspath(p)
            return cfg
    if explicit:
        raise SystemExit(f"config not found: {explicit}")
    return merge(DEFAULTS, {})


def save(cfg: dict, path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def path(cfg: dict, key: str) -> str:
    """Resolve a configured folder: absolute as given, otherwise under based_root."""
    p = os.path.expanduser(cfg["paths"][key])
    if key in ("based_root", "work") or os.path.isabs(p) or not p:
        return p
    return os.path.join(os.path.expanduser(cfg["paths"]["based_root"]), p)
