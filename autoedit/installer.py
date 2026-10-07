"""Put a launcher in Resolve's Workspace > Scripts menu so the plan can be built from inside the app (Free edition)."""
import os
import sys

LAUNCHER = '''# BASED Auto Edit: builds the newest edit plan as DaVinci Resolve timelines.
# Make the plan first with `autoedit run --dry-run` (or the stage commands) in a terminal.
import sys
sys.path.insert(0, {repo!r})
from autoedit import resolve_api
resolve_api.run_from_menu({config!r})
'''


def scripts_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("APPDATA", home), "Blackmagic Design", "DaVinci Resolve", "Support",
                            "Fusion", "Scripts", "Utility")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "Blackmagic Design", "DaVinci Resolve",
                            "Fusion", "Scripts", "Utility")
    return os.path.join(home, ".local", "share", "DaVinciResolve", "Fusion", "Scripts", "Utility")


def install_resolve_script(cfg: dict, folder: str | None = None) -> str:
    folder = folder or scripts_dir()
    os.makedirs(folder, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(folder, "BASED Auto Edit.py")
    with open(path, "w") as f:
        f.write(LAUNCHER.format(repo=repo, config=cfg.get("_path", "")))
    return f"installed {path}\nIn Resolve: Workspace > Scripts > BASED Auto Edit"
