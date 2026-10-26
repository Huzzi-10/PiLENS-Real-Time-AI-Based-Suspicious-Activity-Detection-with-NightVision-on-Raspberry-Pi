"""PiLENS cascade runtime: MoViNet-A0 on every clip + CLIP ViT-B/16 confirmation on every alert (Pi 5, ONNX Runtime CPU).

Same pipeline as step6_realtime.py, plus:
  (1) A0 alerts are counted at three operating points at once (Youden / val-FPR10% / val-FPR5%,
      thresholds from thr_reco in heads_meta.json);
  (2) on every A0 alert, CLIP looks at 4 frames of that clip and confirms or rejects it
      (in the alert thread, so A0 never stops).
The summary reports, per operating point, A0 alerts vs CLIP-confirmed alerts, alerts per hour and CLIP time.

Run on the Pi (inside the v3 folder, venv active; export_movinet/ and export_clip_b16/ must be here):
  python step6_cascade.py --dir export_movinet --source camera --max-sec 7200 --quiet
  python step6_cascade.py --dir export_movinet --source samples/Steal.mp4 --quiet

Threads:
  capture   : frames from camera/video, crop, rolling buffer (73 frames), motion gate, sends a clip every HOP frames
  inference : backbone + heads, "2 of the last 3 clips" alert rule
  alert     : on alert, CLIP check, snapshot and type hint are saved and printed
Output in out/: scores.csv, alerts/, alerts.jsonl
"""
import os, sys, time, json, argparse, threading, queue, subprocess, collections
import numpy as np
import cv2
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="export")
ap.add_argument("--source", default="camera", help="'camera' ya video file")
ap.add_argument("--thr", type=float, default=0.753, help="binary score threshold (mlp seed0 ka val-chosen Youden threshold = 0.753)")
ap.add_argument("--threads", type=int, default=3, help="backbone threads (benchmark mein best = 3)")
ap.add_argument("--hop", type=int, default=18, help="naya clip har itne frame par (18 = 0.6 sec @30FPS)")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--motion", type=float, default=0.003, help="motion gate: itne se kam hilne wale clip skip (0 = gate band)")
ap.add_argument("--cooldown", type=float, default=10, help="alert ke baad itne sec naya alert nahi")
ap.add_argument("--out", default="out")
ap.add_argument("--max-sec", type=float, default=0, help="itne sec baad ruk jao (0 = Ctrl+C tak)")
ap.add_argument("--fast", action="store_true", help="video ko bina pace kiye chalao (koi clip drop nahi)")
ap.add_argument("--clip-dir", default="export_clip_b16", help="CLIP folder (export_clip_b16)")
ap.add_argument("--clip-model", default="fp32", choices=["fp32", "int8"], help="fp32 = accuracy naapi hui; int8 = 2x tez, accuracy unverified")
ap.add_argument("--clip-threads", type=int, default=3)
ap.add_argument("--no-clip", action="store_true", help="CLIP band (sirf A0)")
ap.add_argument("--quiet", action="store_true", help="har clip ki line print mat karo (lambe run ke liye)")
a, _ = ap.parse_known_args()

_meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
_cl = _meta.get("clip", {})
NF, STEP, SIDE, CROP = _cl.get("frames", 13), _cl.get("stride", 6), _cl.get("side", 182), _cl.get("crop", 160)
MEAN, STD = _cl.get("mean", 0.45), _cl.get("std", 0.225)
SPAN, WARM = (NF - 1) * STEP + 1, 60
CLIP_IDX = [i for i in (0, 4, 8, 12) if i < NF]
OPS = [("youden", "youden"), ("fpr10", "val_FPR_10pct"), ("fpr5", "val_FPR_5pct")]   # (naam, heads_meta thr_reco key)
_tr = _meta.get("thr_reco") or {}
if not all(k in _tr for _, k in OPS):                    # purana export (X3D): sirf --thr, fpr5 ke naam se
    OPS = [("fpr5", "val_FPR_5pct")]; _tr = {"val_FPR_5pct": a.thr}
THR = {op: float(_tr[k]) for op, k in OPS}
cv2.setNumThreads(1)                      # opencv inference ke cores na khaye


def temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        return float("nan")


def sess(path, th):
    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def prep(fr):
    """frame -> 160x160 RGB uint8 (training jaisa: short side 182, center crop)"""
    h, w = fr.shape[:2]; s = SIDE / min(h, w)
    fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
    h, w = fr.shape[:2]; y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    return cv2.cvtColor(fr[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB)


def frames_camera():
    import select, tempfile
    w, h = a.width, a.height
    cmd = ["rpicam-vid", "-t", "0", "--nopreview", "--codec", "yuv420", "--width", str(w), "--height", str(h),
           "--framerate", str(a.fps), "-o", "-"]
    n = w * h * 3 // 2
    errf = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=n * 2)

    def fail(why):
        errf.seek(0); tail = errf.read().decode(errors="ignore").strip().splitlines()[-8:]
        print(f"CAMERA PROBLEM: {why}. rpicam-vid ne ye kaha:\n  " + "\n  ".join(tail or ["(kuch nahi)"]))
        print("Karo: (1) 'pkill -f rpicam' (purana atka hua process), (2) 'rpicam-hello --list-cameras', (3) camera ribbon cable check, (4) Pi reboot.")
    try:
        first = True
        while True:
            if not select.select([p.stdout], [], [], 20 if first else 5)[0]:
                fail("20 sec tak koi frame nahi aaya" if first else "frames aana ruk gaye"); return
            buf = p.stdout.read(n)
            if len(buf) < n:
                fail("camera stream band ho gayi"); return
            first = False
            yield cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420), time.time()
    finally:
        p.terminate()


def frames_file():
    if not os.path.exists(a.source):
        print(f"FILE NAHI MILI: {a.source} (naam aur folder check karo: ls samples)")
        return
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened() or not cap.read()[0]:
        print(f"VIDEO KHUL/DECODE NAHI HUI: {a.source}. Shayad codec (AV1/HEVC/VP9) opencv nahi padh sakta; "
              "ffmpeg se H.264 mein badal do (PI_GUIDE ya chat dekho).")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or a.fps
    t0, i = time.time(), 0
    while True:
        ok, fr = cap.read()
        if not ok:
            return
        if not a.fast:                                   # camera ki tarah asli speed par
            wait = t0 + i / fps - time.time()
            if wait > 0:
                time.sleep(wait)
        i += 1
        yield fr, time.time()


def put_latest(q, item, st):
    """inference peeche ho to purana clip phenko (live stream mein latest hi kaam ka hai)"""
    while True:
        try:
            q.put_nowait(item); return
        except queue.Full:
            try:
                q.get_nowait(); st["dropped"] += 1
            except queue.Empty:
                pass


def capture(q, stop, st):
    src = frames_camera() if a.source == "camera" else frames_file()
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    buf, mot, n = collections.deque(maxlen=SPAN), collections.deque(maxlen=a.hop), 0
    raw = collections.deque(maxlen=SPAN)                 # CLIP ke liye asli frames (13-frame clip ke 0,4,8,12)
    for fr, tcap in src:
        if stop.is_set():
            break
        n += 1
        buf.append(prep(fr)); raw.append(fr)
        mot.append(float((mog.apply(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (80, 60))) > 0).mean()))
        if len(buf) == SPAN and n % a.hop == 0:
            item = dict(frames=np.stack(list(buf)[::STEP]), motion=float(np.mean(mot)), t_end=tcap, last=fr, idx=n,
                        craw=[raw[i * STEP] for i in CLIP_IDX])
            if a.source != "camera" and a.fast:
                q.put(item)                              # fast mode: koi clip drop nahi
            else:
                put_latest(q, item, st)
    st["frames"] = n
    q.put(None)


def infer(sb, sh, meta, q, aq, stop, st):
    R, thr = meta["recent_clips"], a.thr
    hist, flags, last_alert = [], collections.deque(maxlen=3), -1e9
    TH = dict(THR)
    fl = {op: collections.deque(maxlen=3) for op in TH}; la = {op: -1e9 for op in TH}
    csv = open(os.path.join(a.out, "scores.csv"), "w")
    csv.write("t_sec,frame,motion,gated,score,flags,infer_ms,e2e_ms,temp_c\n")
    t_start = time.time(); k = 0
    while True:
        item = q.get()
        if item is None:
            break
        t0 = time.perf_counter(); k += 1
        gated = a.motion > 0 and item["idx"] > WARM and item["motion"] < a.motion
        score, logits = 0.0, None
        if gated:
            st["gated"] += 1
        else:
            x = (item["frames"].astype(np.float32) / 255 - MEAN) / STD           # T,H,W,C
            x = np.ascontiguousarray(x.transpose(3, 0, 1, 2)[None])
            f = sb.run(None, {"clip": x})[0]
            hist = (hist + [f[0]])[-R:]
            rec = np.stack([hist[0]] * (R - len(hist)) + hist)[None].astype(np.float32)
            s, logits = sh.run(None, {"feat": f, "recent": rec})
            score = float(s[0])
        st["smax"] = max(st["smax"], score)
        if a.quiet and k % 500 == 0:
            print(f"[{time.time() - t_start:7.0f}s] clips {k} | max score ab tak {st['smax']:.3g} | A0 alerts {dict(st['a0'])} | CLIP confirmed {dict(st['ok'])}", flush=True)
        flags.append(score >= thr)
        ms = (time.perf_counter() - t0) * 1000
        e2e = (time.time() - item["t_end"]) * 1000
        st["clips"] += 1
        if not gated:                                    # sirf asli inference wale clips ki timing
            st["ms"].append(ms); st["e2e"].append(e2e)
        tc = temp(); st["tmax"] = max(st["tmax"], tc if tc == tc else 0)
        ts = time.time() - t_start
        csv.write(f"{ts:.2f},{item['idx']},{item['motion']:.4f},{int(gated)},{score:.4f},{sum(flags)},{ms:.0f},{e2e:.0f},{tc:.1f}\n")
        if k % 5 == 0:
            csv.flush()
        tag = "gated (hilna kam)" if gated else f"score {score:.2e}"
        if not a.quiet:
            print(f"[{ts:6.1f}s] clip {k:3d} | motion {item['motion']:.4f} | {tag} | flags {sum(flags)}/3 | {ms:.0f} ms | e2e {e2e:.0f} ms | {tc:.0f}C")
        fired = []
        for op in TH:                                    # wahi 2-of-3 + cooldown rule, har operating point alag
            fl[op].append(score >= TH[op])
            if sum(fl[op]) >= 2 and logits is not None and time.time() - la[op] > a.cooldown:
                la[op] = time.time(); fired.append(op); st["a0"][op] += 1
        if fired:
            z = np.exp(logits[0] - logits[0].max()); z /= z.sum()
            top = [(meta["classes"][i], float(z[i])) for i in np.argsort(-z)[:3]]
            aq.put(dict(frame=item["last"], craw=item["craw"], score=score, top=top, idx=item["idx"], t=ts, ops=fired))
            st["alerts"] += 1
    csv.close()
    aq.put(None)


class Clip:
    """CLIP ViT-B/16 image encoder + binary/groups heads (kd2 export). 4 frames -> avg embedding -> score."""
    def __init__(s, d):
        m = json.load(open(os.path.join(d, "heads_meta_clip.json")))
        s.m, s.S = m, m.get("input_size", 224)
        s.mean, s.std = np.array(m["rgb_mean"], np.float32), np.array(m["rgb_std"], np.float32)
        s.enc = sess(os.path.join(d, f"clip_b16_{a.clip_model}.onnx"), a.clip_threads)
        s.hb = sess(os.path.join(d, "clip_head_binary.onnx"), 1)
        pg = os.path.join(d, "clip_head_groups.onnx")
        s.hg = sess(pg, 1) if os.path.exists(pg) else None
        s.thr = {op: float(m["seed0_thresholds"][f"{op}_r0"]["thrC"]) for op, _ in OPS}

    def prep(s, fr):                                     # short side 224, center crop 224, RGB, CLIP mean/std
        h, w = fr.shape[:2]; k = s.S / min(h, w)
        fr = cv2.resize(fr, (max(s.S, int(round(w * k))), max(s.S, int(round(h * k)))), interpolation=cv2.INTER_CUBIC)
        h, w = fr.shape[:2]; y0, x0 = (h - s.S) // 2, (w - s.S) // 2
        x = cv2.cvtColor(fr[y0:y0 + s.S, x0:x0 + s.S], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        return ((x - s.mean) / s.std).transpose(2, 0, 1)

    def __call__(s, frames):
        x = np.stack([s.prep(f) for f in frames]).astype(np.float32)
        e = s.enc.run(None, {"x": x})[0].mean(0, keepdims=True).astype(np.float32)        # 1,512
        p = float(1 / (1 + np.exp(-float(s.hb.run(None, {"feat": e})[0].reshape(-1)[0]))))
        grp = None
        if s.hg is not None:
            g = s.hg.run(None, {"x": e[None]})[0].reshape(-1); z = np.exp(g - g.max()); z /= z.sum()
            grp = (s.m["groups"][int(z.argmax())], float(z.max()))
        return p, grp


def alert(aq, st, clip):
    os.makedirs(os.path.join(a.out, "alerts"), exist_ok=True)
    while True:
        it = aq.get()
        if it is None:
            break
        path = os.path.join(a.out, "alerts", f"alert_{it['idx']:06d}.jpg")
        cv2.imwrite(path, it["frame"])
        rec = dict(t_sec=round(it["t"], 1), frame=it["idx"], score=round(it["score"], 4), a0_ops=it["ops"], top3=it["top"], image=path)
        line = ""
        if clip is not None:
            t0 = time.perf_counter(); p, grp = clip(it["craw"]); cms = (time.perf_counter() - t0) * 1000
            st["cms"].append(cms)
            ok = {op: p >= clip.thr[op] for op in it["ops"]}
            for op, v in ok.items():
                st["ok"][op] += int(v)
            rec.update(clip_p=p, clip_ms=round(cms), clip_confirm=ok, clip_group=grp)
            line = f"\n  CLIP p={p:.3g} ({cms:.0f} ms) | " + " ".join(f"{op}:{'CONFIRM' if v else 'reject'}" for op, v in ok.items()) + \
                   (f" | group {grp[0]} {grp[1]:.2f}" if grp else "")
        open(os.path.join(a.out, "alerts.jsonl"), "a").write(json.dumps(rec) + "\n")
        print("\n" + "!" * 60 + f"\n  A0 ALERT  t={it['t']:.1f}s  score={it['score']:.3g}  ops {','.join(it['ops'])}\n  " +
              " | ".join(f"{c} {q:.2f}" for c, q in it["top"]) + line + f"\n  snapshot: {path}\n" + "!" * 60 + "\n", flush=True)


def main():
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
    sb, sh = sess(os.path.join(a.dir, meta.get("backbone", "backbone_x3ds.onnx")), a.threads), sess(os.path.join(a.dir, "heads.onnx"), 1)
    print("SCRIPT VERSION: step6_cascade v1.0 (2026-10-13)")
    print(f"model {meta.get('model', 'x3d_s')} | onnxruntime {ort.__version__} | source {a.source} | threads {a.threads} | hop {a.hop} | gate {a.motion}")
    sb.run(None, {"clip": np.zeros((1, 3, NF, CROP, CROP), np.float32)})          # warmup
    q, aq, stop = queue.Queue(maxsize=1), queue.Queue(), threading.Event()
    st = dict(smax=0.0, clips=0, gated=0, dropped=0, alerts=0, frames=0, tmax=0.0, ms=[], e2e=[],
              a0={op: 0 for op, _ in OPS}, ok={op: 0 for op, _ in OPS}, cms=[])
    clip = None
    if not a.no_clip:
        if not os.path.isdir(a.clip_dir):
            print(f"CLIP FOLDER NAHI MILA: {a.clip_dir} (ya --no-clip lagao)"); return
        clip = Clip(a.clip_dir)
        t1 = time.perf_counter(); clip([np.zeros((480, 640, 3), np.uint8)] * len(CLIP_IDX))
        print(f"CLIP {a.clip_model} | threads {a.clip_threads} | warmup {(time.perf_counter() - t1) * 1000:.0f} ms | thrC {clip.thr}")
    print("A0 thresholds:", {op: round(v, 4) for op, v in THR.items()})
    th = [threading.Thread(target=capture, args=(q, stop, st), daemon=True),
          threading.Thread(target=infer, args=(sb, sh, meta, q, aq, stop, st), daemon=True),
          threading.Thread(target=alert, args=(aq, st, clip), daemon=True)]
    t0 = time.time()
    for t in th:
        t.start()
    try:
        while th[1].is_alive():
            th[1].join(0.5)
            if a.max_sec and time.time() - t0 > a.max_sec:
                stop.set(); break
    except KeyboardInterrupt:
        print("\nruk raha hun...")
        stop.set()
    th[0].join(5)
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(None) if th[1].is_alive() else None
    th[1].join(10); th[2].join(60)
    ms, e2e = np.array(st["ms"] or [0]), np.array(st["e2e"] or [0])
    print(f"\n== SUMMARY == {time.time() - t0:.0f} s | frames {st['frames']} | clips {st['clips']} | gated {st['gated']} | "
          f"dropped {st['dropped']} | alerts {st['alerts']} | max score {st['smax']:.3g} | max temp {st['tmax']:.1f}C")
    print(f"per-clip compute: median {np.median(ms):.0f} | p95 {np.percentile(ms, 95):.0f} ms")
    print(f"clip-end se decision tak (e2e): median {np.median(e2e):.0f} | p95 {np.percentile(e2e, 95):.0f} ms"
          + ("  (fast mode mein ye matlab nahi rakhta)" if a.fast else ""))
    hrs = (time.time() - t0) / 3600
    print(f"\n== CASCADE == run {hrs * 60:.1f} min | source {a.source}")
    print("op      | A0 thr | A0 alerts | CLIP confirmed | A0 per hour | cascade per hour")
    for op, k in OPS:
        n0, n1 = st["a0"][op], st["ok"][op]
        rate = lambda n: (f"{n / hrs:.2f}" if n else f"0 (95% upper {3.0 / hrs:.2f})") if hrs > 0 else "-"
        print(f"{op:7s} | {THR[op]:.3f}  | {n0:9d} | {(str(n1) if clip else '-'):>14s} | {rate(n0):>11s} | {(rate(n1) if clip else '-'):>16s}")
    if st["cms"]:
        c = np.array(st["cms"]); print(f"CLIP confirm time ({a.clip_model}): median {np.median(c):.0f} | max {c.max():.0f} ms | n {len(c)}")
    print("END OF REPORT")


main()
