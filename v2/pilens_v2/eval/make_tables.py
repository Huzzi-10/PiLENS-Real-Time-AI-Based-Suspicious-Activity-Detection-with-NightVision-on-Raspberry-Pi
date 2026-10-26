# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Fill the paper tables from the result files (only measured numbers).

    python -m pilens_v2.eval.make_tables --runs runs --bench bench_results \
        --logs logs --alert replay/alert_eval.json --out docs/v2_results.md

Rows appear only for results that exist; nothing is estimated. Re-run it after
every new experiment and commit the markdown together with the JSON files.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .. import spec


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def pm(v, scale=1.0, nd=1):
    """(mean, std) -> 'm +- s'."""
    if not v:
        return "-"
    return f"{v[0] * scale:.{nd}f} +- {v[1] * scale:.{nd}f}"


def ms(d, key="median_ms"):
    return f"{d[key]:.1f}" if d and key in d else "-"


def table(header, rows):
    if not rows:
        return "_No results yet._\n"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def cls14_section(runs_dir):
    rows, detail = [], []
    for p in sorted(Path(runs_dir).glob("*/summary.json")):
        s = _load(p)
        if not s or "accuracy_mean_std" not in s:
            continue
        rows.append([p.parent.name, s["head"], len(s["folds"]), len(s["seeds"]),
                     pm(s["accuracy_mean_std"], 100), pm(s["macro_f1_mean_std"], 100)])
        pc = s.get("per_class", {})
        if pc:
            detail.append(f"\n**Per-class ({p.parent.name})**\n\n" + table(
                ["Class", "Precision", "Recall", "F1"],
                [[c, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}"] for c, v in pc.items()]))
            cm = s.get("confusion_matrix_first_seed_summed_over_folds")
            if cm:
                short = [c[:5] for c in s.get("classes", spec.CLASSES_14)]
                detail.append(f"\n**Confusion matrix ({p.parent.name}, rows = true, first seed, summed over folds)**\n\n"
                              + table([""] + short, [[short[i]] + r for i, r in enumerate(cm)]))
    head = table(["Run", "Head", "Folds", "Seeds", "Accuracy (%)", "Macro-F1 (%)"], rows)
    return head + f"\nChance = {100 / len(spec.CLASSES_14):.1f}%. Mean +- std over folds.\n" + "".join(detail)


def mil_section(runs_dir):
    rows = []
    for p in sorted(Path(runs_dir).glob("*/summary.json")):
        s = _load(p)
        if not s or "video_auc" not in s:
            continue
        prs = [r.get("test_video_pr_at_val_thr", {}) for r in s.get("runs", [])]
        prec = sum(x.get("precision", 0) for x in prs) / max(1, len(prs))
        rec = sum(x.get("recall", 0) for x in prs) / max(1, len(prs))
        rows.append([p.parent.name, s["head"], len(s["seeds"]), pm(s["video_auc"], 1, 3), pm(s["frame_auc"], 1, 3),
                     pm(s["frame_ap"], 1, 3), pm(s.get("frame_auc_smoothed"), 1, 3), f"{prec:.2f} / {rec:.2f}"])
    return table(["Run", "Head", "Seeds", "Video AUC", "Frame AUC", "Frame AP",
                  "Frame AUC (smoothed, separate row)", "Video P / R at val thr"], rows)


def _flags(v):
    return "-" if v is None else ("no" if v == 0 else hex(v))


def bench_section(bench_dir):
    rows, sysrow = [], None
    for p in sorted(Path(bench_dir).glob("*.json")):
        b = _load(p)
        if not b or "latency" not in b:
            continue
        lat, t = b["latency"], b.get("health", {}).get("temp_c") or {}
        rows.append([b["tag"], b["backbone"], b["threads"], b["runs"], ms(lat["preprocess"]), ms(lat["backbone"]),
                     ms(lat["binary_head"]), f"{ms(lat['stage1_total'])} / {ms(lat['stage1_total'], 'p95_ms')}",
                     ms(lat["cls14_head"]), f"{t.get('start', '-')} / {t.get('max', '-')}",
                     _flags(b.get("health", {}).get("throttled_flags"))])
        sysrow = b.get("system")
    out = table(["Tag", "Model", "Threads", "Runs", "Preprocess", "Backbone", "Binary head",
                 "Stage 1 median / p95", "14-class head", "Temp start / max (C)", "Throttled"], rows)
    if rows:
        out += "\nAll times in ms (median unless noted). "
    if sysrow:
        out += (f"System: {sysrow.get('board')}, {sysrow.get('os')}, Python {sysrow.get('python')}, "
                f"ONNX Runtime {sysrow.get('onnxruntime')}.\n")
    return out


def live_section(logs_dir):
    rows = []
    for p in sorted(Path(logs_dir).glob("*/summary.json")):
        s = _load(p)
        if not s or "capture_fps" not in s:
            continue
        a = s.get("args", {})
        setting = ("no gate" if a.get("no_motion") else "gate") + (", binary only" if a.get("no_stage2") else "")
        c, h = s.get("clip_latency") or {}, s.get("health", {})
        num = lambda k, stat, nd: (f"{h[k][stat]:.{nd}f}" if (h.get(k) or {}).get(stat) is not None else "-")
        rows.append([p.parent.name, setting, f"{s['capture_fps']:.1f}",
                     f"{ms(c)} / {ms(c, 'p95_ms')}", f"{100 * s['motion_gate_skip_frac']:.0f}%",
                     num("cpu_pct", "mean", 0), num("ram_mb", "max", 0), num("temp_c", "max", 1),
                     s["alerts"], f"{s['duration_s'] / 60:.1f} min"])
    return table(["Run", "Setting", "Stream throughput (fps)", "Clip latency median / p95 (ms)", "Gate skipped",
                  "CPU % mean", "RAM MB max", "Temp max (C)", "Alerts", "Duration"], rows)


def alert_section(path):
    res = _load(path) if path else None
    if not res:
        return table([], [])
    f = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "-"
    return table(["Rule", "Motion gate", "Detection rate", "Event-to-alert median / p95 (s)", "False alarms / hour"],
                 [[r["rule"], "on" if r["motion_gate"] else "off", f(r["detection_rate"]),
                   f"{f(r['event_to_alert_median_s'])} / {f(r['event_to_alert_p95_s'])}",
                   f(r["false_alarms_per_hour"])] for r in res])


def build(runs="runs", bench="bench_results", logs="logs", alert=None):
    parts = [
        "# PiLENS v2 results (auto-generated)\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m pilens_v2.eval.make_tables`. "
        "Only measured numbers; do not edit by hand, re-run the script.\n",
        "## Table 1: 14-class recognition (Task A)\n", cls14_section(runs),
        "\n## Table 2: weakly supervised anomaly detection (Task B)\n", mil_section(runs),
        "\n## Table 3: Raspberry Pi model latency\n", bench_section(bench),
        "\n## Table 4: full system on the Pi\n", live_section(logs),
        "\n## Table 5: alert logic on the test videos\n", alert_section(alert),
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bench", default="bench_results")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--alert", default="replay/alert_eval.json")
    ap.add_argument("--out", default="docs/v2_results.md")
    args = ap.parse_args()
    md = build(args.runs, args.bench, args.logs, args.alert)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
