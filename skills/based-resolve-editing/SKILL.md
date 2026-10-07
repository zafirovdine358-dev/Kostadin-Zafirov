---
name: based-resolve-editing
description: Use when the editor wants to edit a BASED IRL batch in DaVinci Resolve (types "import BASED!", /based-resolve-editing, or asks for the rough cut, camera angles, fine cut, subtitles, products, B-roll, stamps and music or audio fix for Resolve). Runs the autoedit command-line tool stage by stage and reports what it did.
---

# BASED IRL editing in DaVinci Resolve (autoedit)

The `autoedit` tool in this repository is the Resolve version of the BASED IRL editing skills. It plans the edit in
Python (`edit.json`) and builds Resolve timelines through the scripting API. You run its commands, read its output and
report to the editor; you do not drive Resolve's UI.

Speak plainly to the editor: no file paths or tool names unless they ask.

## First time

1. `autoedit doctor`. Fix what it marks FAIL (ffmpeg, Python). Tell the editor which WARNs cost them features
   (no Whisper = no captions or fine cut; no face model = no camera angles; no `pillow` = no product overlays).
2. If there is no `autoedit.json`, `autoedit init --based-root "<their BASED folder>" --work "<scratch folder>"` and
   set `models.whisper`, `models.yunet`, `models.sface` to local files if they have them. Never download models
   without asking.
3. Resolve Studio: needs *External scripting using: Local*. Free edition: `autoedit install-resolve-script`, then they
   run *Workspace > Scripts > BASED Auto Edit* themselves.

## The order of work (what to type)

1. `autoedit ingest` (newest `DD/MM/YY Host` folder) - say which folder, how many clips, which have two mics.
2. `autoedit rough-cut` - cut every stretch where nobody speaks. Yellow `CHECK showing?` markers are silent stretches
   kept in case a phone or product is shown: the editor decides. `--showing claude` asks Claude to look at frames
   (needs an API key; say so before using it).
3. `autoedit camera-angles` - cut on speaker changes, centre the speaker.
4. `autoedit fine-cut` - tighten. Add `--llm` only if the editor wants Claude to read transcripts.
5. `autoedit subtitles` - captions, the BASED graphic, product slide-ins.
6. `autoedit products-behind` - needs `mediapipe`.
7. `autoedit broll --short G1291 --product "curl cream"` - only when asked.
8. `autoedit stamps-and-music --stamp <WM-XXXX.png>` - the editor downloads the stamp from the Editor Portal. Never ask
   for, store or write down the portal link.
9. `autoedit audio-fix` - optionally `--reference <voice.wav>` (an Adobe Podcast short they like).
10. `autoedit status`, then `autoedit apply` (add `--no-resolve` for EDL + SRT files only).

`autoedit run --dry-run` does 1-9 in one go. Re-running a stage resets everything after it; say so before you do it
over work the editor has already looked at. Use `--shorts G1291,G1292` to limit a stage.

## Rules that came with the editor's SOPs

- Keep the conversation in order, start to finish. Never rearrange it into a "best of".
- A short the editor already edited is theirs: do not rebuild it. Build into a new timeline (`apply` never replaces
  one) and let them compare.
- A guest who does not follow BASED cannot be a short: the tool notes it, tell the editor.
- Nobody here can hear. Audio numbers are measurements; say that and let their ears overrule them.
- The tool picks music by analysis, not by ear: list the picks and offer swaps.

## Report (end of every stage, and after `apply`)

Per short: length before and after, the hook line, what was cut, phone/product shots kept, the zoom and any BAD ANGLE or
STABILISED markers, captions/products/stamp/music, and anything for the editor to check. `report.md` in the work
folder has most of it; `autoedit status` prints the table and the notes.

`apply` prints every difference between the plan and what Resolve actually built. Read them out. If it says the first
clip landed in the wrong place, or that Resolve refused a property, stop and tell the editor what to check rather than
carrying on.
