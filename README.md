# autoedit: BASED IRL editing for DaVinci Resolve

The ten skills of the *BASED IRL editing* plugin (import, rough cut, camera angles, fine cut, subtitles, products,
products behind, B-roll, stamps and music, audio fix) ported from Premiere Pro to DaVinci Resolve, as a command-line
tool instead of a Claude session driving Premiere.

It watches your raw street-interview clips, decides the edit, and builds one Resolve timeline per short (or one
`BATCH n` timeline). You open it and finish by eye, the way the SOPs describe.

> **Status.** Everything that can be tested without Resolve is tested (over a hundred tests: the cutting logic, captions, audio
> chain, baked overlays, the Resolve builder against a stand-in of the scripting API). It has **not** been run against a
> real DaVinci Resolve, and the Whisper, face and Claude steps were tested with stand-ins because those models and keys
> are not available where this was written. See [Limits](#limits-read-this-before-trusting-it).

## What it does

| Skill in the plugin | Command | What happens |
|---|---|---|
| `based-setup` | `autoedit doctor`, `autoedit init` | checks ffmpeg, models, folders, Resolve; writes `autoedit.json` |
| `import-based` | `autoedit ingest` | finds the newest `DD/MM/YY Host` shoot folder, probes every clip, finds the live mic(s) and the empty audio channels |
| `based-rough-cut` | `autoedit rough-cut` | two Whisper passes plus an energy guard; cuts the head, every gap of 0.9 s or more and the tail; long silent stretches where someone may be showing a phone are kept and marked |
| `based-camera-angles` | `autoedit camera-angles` | cuts on every speaker change, centres the speaker's head (YuNet/SFace), one zoom per short, `BAD ANGLE` and `STABILISED` markers |
| `based-fine-cut` | `autoedit fine-cut [--llm]` | hook on the first clear question, fillers, stutters, echoes, repeats, trailing goodbyes, shaved heads and natural tails; `--llm` lets Claude read the transcript |
| `based-subtitles` | `autoedit subtitles` | word-for-word lowercase captions, 3 words max, `based` on its own, ASR repairs, other brands masked and muted, SRT; product slide-ins at first mention |
| `based-products-behind` | `autoedit products-behind` | slide-ins that stay on one person 2 s or more come out from behind them |
| `based-broll` | `autoedit broll --product "curl cream"` | 1-2 s cuts of a model using the product over the explanation, from your local B-roll folder |
| `based-stamps-and-music` | `autoedit stamps-and-music --stamp WM-XXXX.png` | stamp from the last 30 s on a cut; one `ES_` track per short, never repeated within 10 shorts, -18 dB |
| `based-audio-fix` | `autoedit audio-fix [--reference voice.wav]` | rumble cut, light denoise, tone matched to a reference, levelled to -18 dB, pauses pulled down 2:1, limited at -3 dB, 3-frame crossfades |
| (build) | `autoedit apply` | creates the timelines in Resolve and reads them back to check them |
| (all) | `autoedit run` | ingest, every stage, then apply |

Run the stages one at a time, like the plugin (`autoedit rough-cut`, look, `autoedit camera-angles`, ...), or all at
once with `autoedit run`. Re-running a stage resets everything after it; the plan stays in one file, `edit.json`.

## Install

```
pip install -e ".[all]"        # or just: pip install -e .   (numpy only, see tiers below)
autoedit doctor                # what is installed, what is missing, what to do next
autoedit init --based-root "~/Documents/CLIENT WORK/BASED" --work "~/Documents/BASED Auto Edit"
```

You need `ffmpeg`/`ffprobe` on the PATH and Python 3.10+. Everything is optional beyond that:

| You have | You get |
|---|---|
| ffmpeg + numpy | energy-based rough cut, stamps, music, audio fix, EDL |
| + `faster-whisper` | word timings: precise cuts, captions, fine cut, products, B-roll |
| + `opencv-python-headless` and the YuNet (and SFace) `.onnx` files | camera angles and reframing |
| + `pillow` | product slide-ins, stamp and BASED graphic as clips |
| + `mediapipe` | products behind the person |
| + `anthropic` and an API key (or `ant auth login`) | `--llm` fine cut, `--showing claude` |

Models are not downloaded for you: set `models.whisper` (a name or a local faster-whisper folder), `models.yunet`
and `models.sface` in `autoedit.json`:

```json
{"models": {"whisper": "base",
            "yunet": "~/Documents/Claude Tools/models/face_detection_yunet_2023mar.onnx",
            "sface": "~/Documents/Claude Tools/models/face_recognition_sface_2021dec.onnx"}}
```

`autoedit doctor` shows PASS for each one it finds. YuNet finds the faces (camera angles); SFace is only needed to
recognise the host across shorts: without it the host is the face whose mouth moves when the host's voice is on.
To check the YuNet file on your machine: `AUTOEDIT_YUNET=/path/to/face_detection_yunet_2023mar.onnx pytest tests/test_real_yunet.py`
(it frames a real face photo and checks the face lands between the guides).

## Use

```
autoedit run --dry-run                       # newest shoot -> plan, report.md, nothing touches Resolve
autoedit status                              # per-short table
autoedit apply                               # build the timelines in Resolve
autoedit apply --no-resolve                  # only write G1291.edl / G1291.srt (import by hand)
```

`apply` needs **Resolve Studio** running with *DaVinci Resolve > Preferences > System > General > External scripting
using: Local*. Resolve 21.1 moved Python scripting to Studio, so on the **Free** edition:

- up to 21.0: run `autoedit install-resolve-script` once, then use *Workspace > Scripts > BASED Auto Edit* inside
  Resolve; it builds the newest plan from there (that script needs nothing but the standard library);
- 21.1 and newer: the Scripts menu only runs Lua. Use `autoedit apply --no-resolve` and import the `.edl` (*File >
  Import > Timeline*) and the `.srt` (*File > Import > Subtitle*) by hand: cuts and captions only, none of the framing,
  overlays or cleaned audio.

When Resolve cannot be reached, `apply` and `run` write those two files themselves and exit with code 3.

Your footage layout is the plugin's: `BASED/IRL/FOOTAGE/New Vids/<DD-MM-YY Host>/`, `BASED/IRL/ASSETS/{Music,BASED
Products,Stamps,B-roll}`. Change any of it under `paths` in `autoedit.json`. The stamp comes from the Editor Portal:
download the 9:16 PNG yourself. This tool never opens the portal and never asks for, stores or writes down its link.

## What you get in Resolve

One timeline per short, named `BATCH n - G1291` (or `--layout batch`: one `BATCH n` timeline, shorts a minute apart).
1080x1920 at 29.97, clips imported into a `BATCH n` bin. A timeline with the same name is never replaced: you get `v2`.

| Track | Content |
|---|---|
| V1 | the footage pieces, back to back, picture only; Crop scaling with the planned zoom/pan/tilt |
| V2 / V3 | product behind the person / the person cut out (baked) |
| V4 | product slide-ins (baked) |
| V5 | B-roll, scaled to fill |
| V6 | stamp |
| V7 | BASED graphic |
| A1 | the cleaned dialogue (the live mic of each piece), or the camera audio if you skip `audio-fix` |
| A3 | music |
| subtitle track | captions |
| markers | green `STABILISED`, red `BAD ANGLE - fix`, yellow `CHECK showing?`, blue `B-ROLL` |

## Why it is not a line-for-line port

Resolve's scripting API cannot add keyframes, effects, masks or transitions. So:

- **Camera moves** are steady shots: a moving speaker becomes several pieces (`STABILISED` marker) that you can refine
  with keyframes.
- **Slide-ins, products behind, the stamp and the BASED graphic** are rendered to transparent ProRes clips with ffmpeg
  and placed on tracks. For products behind, the footage underneath is rendered too, so the three layers match to the
  pixel.
- **Voice clean-up, music gain and crossfades** are baked into one dialogue file and one music file per short.
- **Looking at frames** (is a phone shown?) needs a person or Claude: by default long silent stretches are kept and
  marked for you.

## Limits: read this before trusting it

- Not run against real Resolve. The API calls follow the 18.5+ documentation, with fallbacks, and the builder reads
  every timeline back and prints what is not where the plan says. Try the first run on a scratch project and look at one
  framed shot: the Pan/Tilt/Zoom/Crop conventions are the part to eyeball.
- Nobody here can hear: the audio stage is measured (speech level, peak, pause depth), not listened to. Your ears win.
- Speaker labelling is the louder mic when there are two mics, a voice-clustering guess (pitch and tone, plus who says
  the host phrases) when there is one. Put H/G per sentence in `<work>/<short>/labels.txt` to override it.
- Fine cut applies the editor's rules mechanically (hook, echoes, fillers, goodbyes). It never rearranges the
  conversation. `--llm` adds Claude's reading of the transcript, with every proposal checked and snapped to speech.
- Disk: baked clips and the cleaned voices go in the work folder (ProRes: large).

## Files

```
autoedit/        cli.py plan.py config.py doctor.py ingest.py analysis.py speech.py rough.py turns.py faces.py
                 framing.py angles.py finecut.py captions.py products.py sprites.py behind.py broll.py stamps.py
                 music.py audiofix.py export.py report.py resolve_api.py llm.py showing.py installer.py
skills/          a Claude skill that runs these commands for an editor
tests/           pytest, with a stand-in for the Resolve API (tests/fake_resolve.py)
```

`pip install -e ".[test,all]" && pytest`

README in Macedonian: [README.mk.md](README.mk.md).
