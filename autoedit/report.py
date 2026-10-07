"""What the plan did, per short (the 'report' step every skill ends with). Standard library only."""
from .plan import EditPlan, Short


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}" if t >= 60 else f"{t:.1f}s"


def flags(s: Short) -> list[str]:
    out = []
    bad = sum("bad_angle" in p.flags for p in s.pieces)
    if bad:
        out.append(f"{bad} bad angle")
    review = [m for m in s.markers if m.name.startswith("CHECK")]
    if review:
        out.append(f"{len(review)} silent stretch(es) to review")
    if any("guest may not follow" in n for n in s.notes):
        out.append("guest may not follow BASED")
    if s.stats.get("asr") == "energy":
        out.append("no word timings")
    return out


def table(plan: EditPlan) -> str:
    head = ["short", "length", "pieces", "captions", "products", "stamp", "music", "voice", "check"]
    rows = []
    for s in plan.shorts:
        if not s.enabled:
            continue
        st = s.stats
        prods = st.get("products", [])
        behind = st.get("behind", 0)
        aud = st.get("audio")
        rows.append([
            s.name, f"{_fmt(s.duration)} -> {_fmt(s.tl_duration())}", str(len(s.pieces)),
            str(st["captions"]["count"]) if "captions" in st else "-",
            (f"{len(prods)}" + (f" ({behind} behind)" if behind else "")) if prods or "products" in st else "-",
            "yes" if "stamp" in st else "-", st.get("music", "-") if isinstance(st.get("music"), str) else "-",
            f"{aud['speech_after']:.0f} dB" if aud else "-", ", ".join(flags(s)) or "ok"])
    w = [max(len(head[i]), *(len(r[i]) for r in rows)) if rows else len(head[i]) for i in range(len(head))]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip()
    return "\n".join([line(head), line(["-" * x for x in w])] + [line(r) for r in rows])


def markdown(plan: EditPlan) -> str:
    b = plan.batch
    out = [f"# {b.get('name') or 'BATCH'}  ({b.get('date', '')} {b.get('host', '')})", "",
           f"Stages done: {', '.join(plan.stages) or 'none'}", ""]
    for s in plan.shorts:
        if not s.enabled:
            continue
        st = s.stats
        out.append(f"## {s.name}")
        out.append(f"- Length: {_fmt(s.duration)} -> {_fmt(s.tl_duration())}, {len(s.pieces)} pieces")
        fine = st.get("fine")
        if fine:
            out.append(f"- Hook: {fine['hook'] or 'none found (kept the start)'}")
            if fine["cuts"]:
                out.append("- Cut: " + ", ".join(f"{n} {r}" for r, n in sorted(fine["cuts"].items())))
        zooms = st.get("angles", {}).get("zoom")
        if zooms:
            out.append(f"- Framing: zoom {', '.join(f'{z:.2f}' for z in zooms)}, {st['angles']['turns']} speaker turns")
        phone = [p for p in s.pieces if "phone" in p.flags]
        if phone:
            out.append(f"- Kept silent 'showing' shots: {len(phone)}")
        if "captions" in st:
            c = st["captions"]
            out.append(f"- Captions: {c['count']} ({c['based']} BASED, {c['brands']} other brand)")
        if "products" in st:
            out.append("- Products: " + (", ".join(f"{n} at {a:.1f}s" for n, a, _ in st["products"]) or "none")
                       + (f" ({st['behind']} behind the person)" if st.get("behind") else ""))
        if "stamp" in st:
            out.append(f"- Stamp: {st['stamp']['start']:.1f}-{st['stamp']['end']:.1f}s"
                       + ("" if st["stamp"]["on_cut"] else " (not on a cut)"))
        if isinstance(st.get("music"), str):
            out.append(f"- Music: {st['music']} (picked by analysis, offer swaps)")
        if "audio" in st:
            a = st["audio"]
            out.append(f"- Voice: speech {a['speech_before']:.1f} -> {a['speech_after']:.1f} dB, peak {a['peak_after']:.1f}, "
                       f"pauses {a['gap_after']:.0f} dB under speech (measured, not heard: your ears decide)")
        if s.notes or flags(s):
            out.append("- To check:")
            for f in flags(s):
                out.append(f"  - {f}")
            for n in s.notes:
                out.append(f"  - {n}")
        out.append("")
    return "\n".join(out)
