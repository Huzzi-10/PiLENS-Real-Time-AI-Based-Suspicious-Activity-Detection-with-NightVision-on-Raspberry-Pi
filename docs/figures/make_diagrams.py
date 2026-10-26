"""Builds the README banner and the pipeline diagram as SVG (light and dark).

    python docs/figures/make_diagrams.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"

T = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", box="#f6f8fa", line="#d1d9e0", accent="#2a78d6",
                  accentbg="#e8f1fc", aqua="#14946a", aquabg="#e5f6ef", red="#d03b3b", band="#f6f8fa"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", box="#151b23", line="#3d444d", accent="#4493f8",
                 accentbg="#0f2440", aqua="#2fbf8a", aquabg="#0d2b22", red="#f47067", band="#151b23"),
}


def box(x, y, w, h, title, sub, t, kind="plain"):
    fill, stroke, tc = t["box"], t["line"], t["ink"]
    if kind == "accent":
        fill, stroke, tc = t["accentbg"], t["accent"], t["ink"]
    if kind == "aqua":
        fill, stroke, tc = t["aquabg"], t["aqua"], t["ink"]
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 - 6}" text-anchor="middle" font-size="15" font-weight="700" fill="{tc}">{title}</text>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 16}" text-anchor="middle" font-size="12" fill="{t["ink2"]}">{sub}</text>')


def arrow(x1, y1, x2, y2, t, color=None, dash=False):
    c = color or t["ink2"]
    d = ' stroke-dasharray="5 4"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.8"{d} marker-end="url(#ah-{c[1:]})"/>'


def marker(c):
    return (f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')


def pipeline(mode):
    t = T[mode]
    W, H = 1100, 430
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<defs>{marker(t["ink2"])}{marker(t["aqua"])}{marker(t["red"])}{marker(t["accent"])}</defs>',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>']
    # band 1
    s.append(f'<rect x="20" y="20" width="{W - 40}" height="190" rx="16" fill="{t["band"]}"/>')
    s.append(f'<text x="44" y="52" font-size="13" font-weight="700" fill="{t["accent"]}" letter-spacing="1">EVERY CLIP  ·  ON THE RASPBERRY PI 5 CPU  ·  ABOUT 0.14 s</text>')
    bw, bh, y = 178, 90, 82
    xs = [44, 252, 460, 668, 876]
    items = [("Camera", "NoIR, 30 FPS", "plain"), ("Motion gate", "skips still scenes", "plain"),
             ("MoViNet-A0", "2.4 s clip every 0.6 s", "accent"), ("Score head", "suspicious score 0-1", "accent"),
             ("2-of-3 rule", "2 of last 3 clips high", "plain")]
    for x, (a, b, k) in zip(xs, items):
        s.append(box(x, y, bw, bh, a, b, t, k))
    for i in range(4):
        s.append(arrow(xs[i] + bw + 4, y + bh / 2, xs[i + 1] - 6, y + bh / 2, t))
    # band 2
    s.append(f'<rect x="20" y="230" width="{W - 40}" height="180" rx="16" fill="{t["band"]}"/>')
    s.append(f'<text x="44" y="262" font-size="13" font-weight="700" fill="{t["aqua"]}" letter-spacing="1">ONLY WHEN AN ALERT IS RAISED  ·  ABOUT 2 s</text>')
    y2 = 292
    s.append(box(668, y2, bw, bh, "CLIP check", "looks at 4 frames", t, "aqua"))
    s.append(box(876, y2, bw, bh, "Alert", "snapshot, type hint, buzzer", t))
    s.append(arrow(876 + bw / 2, y + bh + 4, 668 + bw / 2 + 30, y2 - 6, t, t["aqua"]))
    s.append(f'<text x="{876 + bw / 2 - 28}" y="{(y + bh + y2) / 2 + 4}" text-anchor="end" font-size="12" fill="{t["aqua"]}">alert</text>')
    s.append(arrow(668 + bw + 4, y2 + bh / 2, 876 - 6, y2 + bh / 2, t, t["aqua"]))
    s.append(f'<text x="{668 + bw + 15}" y="{y2 + bh / 2 - 12}" text-anchor="middle" font-size="11" fill="{t["aqua"]}">real</text>')
    # reject path back to scoring
    s.append(f'<path d="M668,{y2 + bh / 2} H560 V{y + bh + 6}" fill="none" stroke="{t["red"]}" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#ah-{t["red"][1:]})"/>')
    s.append(f'<text x="660" y="{y2 + bh / 2 + 22}" text-anchor="end" font-size="12" fill="{t["red"]}">false alarm: dropped,</text>')
    s.append(f'<text x="660" y="{y2 + bh / 2 + 38}" text-anchor="end" font-size="12" fill="{t["red"]}">keep watching</text>')
    s.append(f'<text x="44" y="320" font-size="13" fill="{t["ink2"]}">The small model watches all the time.</text>')
    s.append(f'<text x="44" y="342" font-size="13" fill="{t["ink2"]}">The large model is asked only when something</text>')
    s.append(f'<text x="44" y="364" font-size="13" fill="{t["ink2"]}">looks wrong, and a 10 s cooldown follows each alert.</text>')
    s.append('</svg>')
    open(os.path.join(HERE, f"pipeline_{mode}.svg"), "w").write("".join(s))


def banner():
    t = T["dark"]
    W, H = 1280, 360
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1220"/><stop offset="1" stop-color="#0f2a4a"/></linearGradient>'
         '<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0 H0 V32" fill="none" stroke="#ffffff" stroke-opacity="0.05" stroke-width="1"/></pattern></defs>',
         f'<rect width="{W}" height="{H}" rx="20" fill="url(#g)"/>', f'<rect width="{W}" height="{H}" rx="20" fill="url(#grid)"/>']
    # detection box motif on the right
    s.append('<g transform="translate(905,58)">'
             '<rect x="0" y="0" width="300" height="200" rx="10" fill="#0d1117" fill-opacity="0.6" stroke="#3d444d"/>'
             '<rect x="70" y="44" width="92" height="128" fill="none" stroke="#4493f8" stroke-width="2.5"/>'
             '<rect x="70" y="26" width="132" height="18" fill="#4493f8"/>'
             '<text x="76" y="40" font-size="12" font-weight="700" fill="#0d1117">SUSPICIOUS 0.94</text>'
             '<rect x="178" y="70" width="80" height="100" fill="none" stroke="#9198a1" stroke-width="1.5" stroke-dasharray="4 3"/>'
             '<circle cx="18" cy="18" r="5" fill="#f47067"/><text x="30" y="22" font-size="11" fill="#9198a1">LIVE · 30 FPS</text>'
             '<text x="12" y="190" font-size="11" fill="#2fbf8a">CLIP check: confirmed</text>'
             '</g>')
    s.append('<text x="64" y="128" font-size="72" font-weight="800" fill="#e6edf3" letter-spacing="-1">PiLENS</text>')
    s.append('<text x="66" y="170" font-size="24" fill="#c9d1d9">Real-time crime and violence detection</text>')
    s.append('<text x="66" y="202" font-size="24" fill="#c9d1d9">on a Raspberry Pi 5. No GPU, no cloud.</text>')
    chips = [("0.919", "video AUC"), ("137 ms", "per clip on the Pi"), ("16.4% → 3.6%", "false alerts with CLIP"), ("2 h live", "0 dropped frames")]
    x = 66
    for v, l in chips:
        w = max(len(v) * 13 + 24, len(l) * 7.4 + 24)
        s.append(f'<rect x="{x}" y="240" width="{w}" height="70" rx="12" fill="#ffffff" fill-opacity="0.06" stroke="#ffffff" stroke-opacity="0.12"/>')
        s.append(f'<text x="{x + 12}" y="270" font-size="22" font-weight="700" fill="#4493f8">{v}</text>')
        s.append(f'<text x="{x + 12}" y="295" font-size="13" fill="#9198a1">{l}</text>')
        x += w + 12
    s.append('</svg>')
    open(os.path.join(HERE, "banner.svg"), "w").write("".join(s))


if __name__ == "__main__":
    pipeline("light"); pipeline("dark"); banner()
    print("diagrams written to", HERE)
