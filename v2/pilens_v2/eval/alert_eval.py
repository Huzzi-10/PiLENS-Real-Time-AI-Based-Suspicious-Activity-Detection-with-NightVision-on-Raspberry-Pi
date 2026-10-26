# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Alert-level evaluation from replay logs (k-of-n and motion-gate ablations).

For every rule it reports: detected events, median / p95 event-to-alert time
(seconds of video from event start to the alert, add the measured Pi clip
latency for the wall-clock number), and false alarms per hour (alerts outside
any annotated event, counted on all test videos, Normal ones included).

The threshold must be the one chosen on VALIDATION; never tune it here.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..data.annotations import read_events_csv, read_official_temporal
from .metrics import event_to_alert, simulate_alerts

RULES = [(1, 1), (2, 3), (3, 5)]


def read_replay(path):
    lines = Path(path).read_text().splitlines()
    head = dict(kv.split("=") for kv in lines[0].lstrip("# ").split())
    rows = list(csv.DictReader(lines[1:]))
    return (np.array([float(r["t"]) for r in rows]), np.array([int(r["motion"]) for r in rows]),
            np.array([float(r["score"]) for r in rows]), int(head["frames"]), float(head["fps"]))


def evaluate(replay_dir, events, threshold, rules=RULES, cooldown=30.0, use_motion=True, tolerance=5.0):
    results = []
    logs = {p.stem: read_replay(p) for p in sorted(Path(replay_dir).glob("*.csv"))}
    for k, n in rules:
        delays, missed, false, hours = [], 0, 0, 0.0
        for vid, (t, motion, score, frames, fps) in logs.items():
            s = np.where(motion.astype(bool), score, 0.0) if use_motion else score
            alerts = simulate_alerts(t, s, threshold, k, n, cooldown)
            ev = [(a / fps, b / fps) for a, b in events.get(vid, [])]
            hours += frames / fps / 3600.0
            false += sum(1 for x in alerts if not any(a <= x <= b + tolerance for a, b in ev))
            for a, b in ev:
                d = event_to_alert(alerts, a, b + tolerance)
                if d is None:
                    missed += 1
                else:
                    delays.append(d)
        n_events = len(delays) + missed
        results.append({
            "rule": f"{k} of {n}", "motion_gate": use_motion, "events": n_events,
            "detected": len(delays), "detection_rate": len(delays) / max(1, n_events),
            "event_to_alert_median_s": float(np.median(delays)) if delays else None,
            "event_to_alert_p95_s": float(np.percentile(delays, 95)) if delays else None,
            "false_alarms": false, "hours": hours, "false_alarms_per_hour": false / max(1e-9, hours),
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--ann", required=True, help="official temporal txt or our annotation CSV")
    ap.add_argument("--threshold", type=float, required=True, help="from validation (pilens_v2.json)")
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--tolerance", type=float, default=5.0, help="s after event end still counted as hit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = (read_events_csv(args.ann)[0] if args.ann.endswith(".csv") else read_official_temporal(args.ann))
    res = []
    for gate in (True, False):
        res += evaluate(args.replay, events, args.threshold, cooldown=args.cooldown,
                        use_motion=gate, tolerance=args.tolerance)
    print(f"{'rule':8s} {'gate':5s} {'detect':>8s} {'e2a med':>8s} {'e2a p95':>8s} {'FA/h':>7s}")
    for r in res:
        fmt = lambda v: f"{v:8.2f}" if v is not None else "     n/a"
        print(f"{r['rule']:8s} {str(r['motion_gate']):5s} {r['detection_rate']:8.3f} "
              f"{fmt(r['event_to_alert_median_s'])} {fmt(r['event_to_alert_p95_s'])} {r['false_alarms_per_hour']:7.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
