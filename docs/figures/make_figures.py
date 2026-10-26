"""Builds the README figures (light and dark versions) from the files in v3/results.

    python docs/figures/make_figures.py

Every number is read from, or copied with its source from, v3/results/.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "v3", "results")

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e8eb", base="#c8ccd1",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#a9adb3", budget="#d03b3b"),
    "dark":  dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#22272e", base="#3d444d",
                  blue="#3987e5", orange="#d95926", aqua="#199e70", gray="#545b64", budget="#e66767"),
}

for f in ("Inter", "DejaVu Sans"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break


def style(ax, t, xgrid=False, ygrid=True):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["base"])
    ax.tick_params(colors=t["ink2"], length=0, labelsize=10)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], lw=1)
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], lw=1)
    ax.set_axisbelow(True)


def fig(t, w, h):
    f = plt.figure(figsize=(w, h), dpi=200)
    f.patch.set_facecolor(t["bg"])
    return f


def title(f, t, main, sub):
    h = f.get_figheight()
    f.text(0.02, 1 - 0.16 / h, main, ha="left", va="top", fontsize=14, weight="bold", color=t["ink"])
    f.text(0.02, 1 - 0.50 / h, sub, ha="left", va="top", fontsize=10, color=t["ink2"])


def save(f, name, mode):
    f.savefig(os.path.join(HERE, f"{name}_{mode}.png"), facecolor=f.get_facecolor())
    plt.close(f)


def rounded_barh(ax, y, w, color, h=0.62):
    ax.barh(y, w, height=h, color=color, edgecolor="none")


# ---------------------------------------------------------------- 1. accuracy vs speed
# Pi 5 latency (3 threads): results/night_a1/SUMMARY.md (A0 137, A1 256), README v2 / paper tables (X3D-S 196),
# results/kd2/PI_CLIP_BENCH.md (CLIP fp32, 4 frames, 3 threads: 2003 ms).
# Video AUC (5 seeds): paper_final/paper_tables.md (A0 0.919, X3D-S 0.889), night_a1/SUMMARY.md (A1 0.928),
# kd2/KD2_SUMMARY.md (CLIP mlp 0.968).
MODELS = [("MoViNet-A0", 137, 0.919, "blue", "PiLENS v3, every clip"),
          ("X3D-S", 196, 0.889, "orange", "PiLENS v2"),
          ("MoViNet-A1", 256, 0.928, "gray", "1.9x slower, gain not significant"),
          ("CLIP ViT-B/16", 2003, 0.968, "aqua", "alert check only")]


def fig_speed(mode):
    t = THEMES[mode]
    f = fig(t, 9, 4.6)
    title(f, t, "Accuracy vs speed on the Raspberry Pi 5",
          "UCF-Crime video AUC (5 seeds) against CPU time per 2.4 s clip. A new clip arrives every 0.6 s.")
    ax = f.add_axes([0.08, 0.13, 0.88, 0.62])
    style(ax, t, xgrid=True)
    ax.axvspan(600, 2600, color=t["budget"], alpha=0.07, lw=0)
    ax.axvline(600, color=t["budget"], lw=1.5, ls=(0, (4, 3)))
    ax.text(615, 0.873, "0.6 s budget: slower models\ncannot score every clip", color=t["budget"], fontsize=9, va="bottom")
    for name, ms, auc, c, note in MODELS:
        ax.scatter(ms, auc, s=110, color=t[c], edgecolor=t["bg"], linewidth=2, zorder=3)
        ha = "right" if ms > 1500 else "left"
        x0 = ms * (0.94 if ms > 1500 else 1.06)
        dy = {"X3D-S": -0.0105, "MoViNet-A0": -0.0105}.get(name, 0.004)
        ax.text(x0, auc + dy, f"{name}  {auc:.3f}", color=t["ink"], fontsize=10, weight="bold", ha=ha)
        ax.text(x0, auc + dy - 0.0062, f"{ms:,} ms · {note}", color=t["ink2"], fontsize=9, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(90, 2600)
    ax.set_xticks([100, 200, 300, 600, 1000, 2000])
    ax.set_xticklabels(["100", "200", "300", "600", "1,000", "2,000"])
    ax.set_ylim(0.87, 0.985)
    ax.set_xlabel("Pi 5 CPU time per clip, ms (log scale)", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Video AUC", color=t["ink2"], fontsize=10)
    save(f, "accuracy_vs_speed", mode)


# ---------------------------------------------------------------- 2. CLIP alert check
# results/kd2/KD2_SUMMARY.md, cascade table, r = 0, mean of 5 seeds, UCF-Crime test set.
OPS = [("Strict\n(val FPR 5%)", 4.7, 1.6, 63.0, 61.7),
       ("Balanced\n(val FPR 10%)", 11.6, 2.8, 76.9, 74.0),
       ("Sensitive\n(Youden)", 16.4, 3.6, 83.6, 80.3)]


def fig_cascade(mode):
    t = THEMES[mode]
    f = fig(t, 9, 4.4)
    title(f, t, "The CLIP check removes most false alerts",
          "UCF-Crime test set, video level, mean of 5 seeds. Grey = MoViNet-A0 alone, blue = A0 + CLIP check.")
    x = np.arange(len(OPS)); w = 0.36
    for k, (key, lim, lab) in enumerate([(1, 20, "False alerts on normal videos (%)  ·  lower is better"),
                                         (3, 100, "Crimes detected, recall (%)  ·  higher is better")]):
        ax = f.add_axes([0.06 + k * 0.49, 0.16, 0.42, 0.56])
        style(ax, t)
        a = [o[key] for o in OPS]; b = [o[key + 1] for o in OPS]
        ax.bar(x - w / 2 - 0.01, a, w, color=t["gray"], edgecolor="none")
        ax.bar(x + w / 2 + 0.01, b, w, color=t["blue"], edgecolor="none")
        for i in range(len(OPS)):
            ax.text(x[i] - w / 2 - 0.01, a[i] + lim * 0.015, f"{a[i]:.1f}", ha="center", va="bottom", fontsize=9, color=t["ink2"])
            ax.text(x[i] + w / 2 + 0.01, b[i] + lim * 0.015, f"{b[i]:.1f}", ha="center", va="bottom", fontsize=9, color=t["ink"], weight="bold")
        ax.set_xticks(x); ax.set_xticklabels([o[0] for o in OPS], fontsize=9, color=t["ink2"])
        ax.set_ylim(0, lim)
        ax.set_title(lab, loc="left", fontsize=10, color=t["ink"], pad=8)
    save(f, "clip_check", mode)


# ---------------------------------------------------------------- 3. backbones
# Video AUC, mean of 5 seeds, same protocol. paper_final/paper_tables.md (A0, X3D-S, 2D CNN rows),
# night_a1/SUMMARY.md (A1), other_backbones/OTHER_BACKBONES.md (the rest), kd2/KD2_SUMMARY.md (CLIP).
BACKBONES = [
    ("CLIP ViT-B/16 (alert check, not per clip)", 0.968, "aqua"),
    ("R(2+1)D-18, fine-tuned (31.3M)", 0.928, "gray"),
    ("MoViNet-A1, frozen", 0.928, "gray"),
    ("MoViNet-A0, frozen (3.1M) · PiLENS", 0.919, "blue"),
    ("MobileNetV3 + mean pool", 0.906, "gray"),
    ("MobileNetV3 + LSTM", 0.904, "gray"),
    ("MobileNetV3 + Transformer", 0.894, "gray"),
    ("X3D-S, frozen · PiLENS v2", 0.889, "gray"),
    ("MobileNetV3-TSM, frozen", 0.886, "gray"),
    ("R(2+1)D-18, frozen", 0.884, "gray"),
    ("MobileNetV3-TSM, fine-tuned", 0.870, "gray"),
    ("MC3-18, frozen", 0.868, "gray"),
    ("ShuffleNet-3D, from scratch", 0.811, "gray"),
    ("MobileNetV3-TSM, from scratch", 0.804, "gray"),
    ("SlowFast-light, from scratch", 0.794, "gray"),
    ("S3D, frozen", 0.784, "gray"),
]


def fig_backbones(mode):
    t = THEMES[mode]
    f = fig(t, 9, 6.6)
    title(f, t, "Video models compared under one protocol",
          "UCF-Crime video AUC, mean of 5 seeds, thresholds from validation, test used once. Axis starts at 0.75.")
    ax = f.add_axes([0.40, 0.06, 0.55, 0.78])
    style(ax, t, xgrid=True, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    n = len(BACKBONES)
    for i, (name, v, c) in enumerate(BACKBONES):
        y = n - 1 - i
        rounded_barh(ax, y, v - 0.75, t[c])
        bold = c != "gray"
        ax.text(v - 0.75 + 0.003, y, f"{v:.3f}", va="center", fontsize=9, color=t["ink"] if bold else t["ink2"], weight="bold" if bold else "normal")
        ax.text(-0.006, y, name, va="center", ha="right", fontsize=9.5, color=t["ink"] if bold else t["ink2"], weight="bold" if bold else "normal")
    ax.set_yticks([])
    ax.set_xlim(0, 0.24)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xticklabels(["0.75", "0.80", "0.85", "0.90", "0.95"])
    ax.set_ylim(-0.7, n - 0.3)
    save(f, "backbones", mode)


# ---------------------------------------------------------------- 4. ROC
def read_roc(name):
    with open(os.path.join(RES, "paper_final", f"roc_{name}.csv")) as fh:
        r = list(csv.DictReader(fh))
    return np.array([float(x["fpr"]) for x in r]), np.array([float(x["tpr"]) for x in r])


def auc(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def fig_roc(mode):
    t = THEMES[mode]
    f = fig(t, 6.2, 5.6)
    title(f, t, "ROC curve, UCF-Crime test set", "Video level, deployed MLP head, seed 0.")
    ax = f.add_axes([0.13, 0.11, 0.82, 0.70])
    style(ax, t, xgrid=True)
    ax.plot([0, 1], [0, 1], color=t["base"], lw=1, ls=(0, (3, 3)))
    for name, label, c in [("X3DS", "X3D-S (v2)", "orange"), ("MoViNetA0", "MoViNet-A0 (v3)", "blue")]:
        x, y = read_roc(name)
        ax.plot(x, y, color=t[c], lw=2.2, label=f"{label}   AUC {auc(x, y):.3f}", solid_capstyle="round")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate", color=t["ink2"], fontsize=10)
    ax.set_ylabel("True positive rate", color=t["ink2"], fontsize=10)
    save(f, "roc", mode)


# ---------------------------------------------------------------- 5. Pi latency
# Same sources as figure 1; X3D-S 196 ms; budget = 600 ms hop.
def fig_latency(mode):
    t = THEMES[mode]
    f = fig(t, 9, 3.6)
    title(f, t, "Time to score one clip on the Raspberry Pi 5 CPU",
          "Median ms per 2.4 s clip, ONNX Runtime, 3 threads. Must stay under 600 ms to keep up with the camera.")
    ax = f.add_axes([0.17, 0.14, 0.78, 0.56])
    style(ax, t, xgrid=True, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    rows = [("MoViNet-A0 · v3", 137, "blue", "4.4x headroom"), ("X3D-S · v2", 196, "gray", "3.1x headroom"),
            ("MoViNet-A1", 256, "gray", "2.3x headroom")]
    for i, (n, v, c, note) in enumerate(rows):
        y = len(rows) - 1 - i
        rounded_barh(ax, y, v, t[c], h=0.58)
        ax.text(v + 8, y, f"{v} ms  ·  {note}", va="center", fontsize=9.5, color=t["ink"] if c == "blue" else t["ink2"], weight="bold" if c == "blue" else "normal")
        ax.text(-10, y, n, va="center", ha="right", fontsize=10, color=t["ink"] if c == "blue" else t["ink2"], weight="bold" if c == "blue" else "normal")
    ax.axvline(600, color=t["budget"], lw=1.5, ls=(0, (4, 3)))
    ax.text(592, 2.45, "600 ms budget", color=t["budget"], fontsize=9, ha="right")
    ax.set_yticks([]); ax.set_xlim(0, 660); ax.set_ylim(-0.6, 2.7)
    f.text(0.17, 0.02, "CLIP ViT-B/16 (4 frames) takes about 2.0 s, so it runs only after an alert, never on every clip.",
           fontsize=9, color=t["ink2"])
    save(f, "pi_latency", mode)


if __name__ == "__main__":
    for m in THEMES:
        fig_speed(m); fig_cascade(m); fig_backbones(m); fig_roc(m); fig_latency(m)
    print("figures written to", HERE)
