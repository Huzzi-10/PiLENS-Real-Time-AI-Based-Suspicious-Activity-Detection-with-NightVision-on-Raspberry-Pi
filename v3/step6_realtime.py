"""PiLENS Step 6: real-time pipeline without the CLIP check (Pi 5, ONNX Runtime CPU). Works with the
MoViNet-A0 export (export_movinet/) and the older X3D-S export (export/).

3 threads:
  capture   : frames from camera/video, crop, rolling buffer (73 frames), motion gate, sends a clip every HOP frames
  inference : backbone + heads (binary + crime type), "2 of the last 3 clips" alert rule
  alert     : on alert, saves and prints a snapshot and the top-3 classes

Run on the Pi (inside the v3 folder, venv active):
  python step6_realtime.py --dir export_movinet --thr 0.935 --source test.mp4          # video at real speed (like a camera)
  python step6_realtime.py --dir export_movinet --thr 0.935 --source test.mp4 --fast   # video without pacing (speed/score check only)
  python step6_realtime.py --dir export_movinet --thr 0.935 --source camera            # live Camera Module 3 (needs rpicam-vid)
The default --thr 0.753 belongs to the X3D-S export; for MoViNet-A0 use 0.935 (val-FPR5%, the deployed point).
Stop with Ctrl+C (or use --max-sec). Output in out/: scores.csv, alerts/, alerts.jsonl
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
a, _ = ap.parse_known_args()

_meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
_cl = _meta.get("clip", {})
NF, STEP, SIDE, CROP = _cl.get("frames", 13), _cl.get("stride", 6), _cl.get("side", 182), _cl.get("crop", 160)
MEAN, STD = _cl.get("mean", 0.45), _cl.get("std", 0.225)
SPAN, WARM = (NF - 1) * STEP + 1, 60
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
    for fr, tcap in src:
        if stop.is_set():
            break
        n += 1
        buf.append(prep(fr))
        mot.append(float((mog.apply(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (80, 60))) > 0).mean()))
        if len(buf) == SPAN and n % a.hop == 0:
            item = dict(frames=np.stack(list(buf)[::STEP]), motion=float(np.mean(mot)), t_end=tcap, last=fr, idx=n)
            if a.source != "camera" and a.fast:
                q.put(item)                              # fast mode: koi clip drop nahi
            else:
                put_latest(q, item, st)
    st["frames"] = n
    q.put(None)


def infer(sb, sh, meta, q, aq, stop, st):
    R, thr = meta["recent_clips"], a.thr
    hist, flags, last_alert = [], collections.deque(maxlen=3), -1e9
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
        print(f"[{ts:6.1f}s] clip {k:3d} | motion {item['motion']:.4f} | {tag} | flags {sum(flags)}/3 | {ms:.0f} ms | e2e {e2e:.0f} ms | {tc:.0f}C")
        if sum(flags) >= 2 and logits is not None and time.time() - last_alert > a.cooldown:
            last_alert = time.time()
            z = np.exp(logits[0] - logits[0].max()); z /= z.sum()
            top = [(meta["classes"][i], float(z[i])) for i in np.argsort(-z)[:3]]
            aq.put(dict(frame=item["last"], score=score, top=top, idx=item["idx"], t=ts))
            st["alerts"] += 1
    csv.close()
    aq.put(None)


def alert(aq, st):
    os.makedirs(os.path.join(a.out, "alerts"), exist_ok=True)
    while True:
        it = aq.get()
        if it is None:
            break
        path = os.path.join(a.out, "alerts", f"alert_{it['idx']:06d}.jpg")
        cv2.imwrite(path, it["frame"])
        rec = dict(t_sec=round(it["t"], 1), frame=it["idx"], score=round(it["score"], 3), top3=it["top"], image=path)
        open(os.path.join(a.out, "alerts.jsonl"), "a").write(json.dumps(rec) + "\n")
        print("\n" + "!" * 60 + f"\n  ALERT  t={it['t']:.1f}s  score={it['score']:.2e}\n  " +
              " | ".join(f"{c} {p:.2f}" for c, p in it["top"]) + f"\n  snapshot: {path}\n" + "!" * 60 + "\n")


def main():
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
    sb, sh = sess(os.path.join(a.dir, meta.get("backbone", "backbone_x3ds.onnx")), a.threads), sess(os.path.join(a.dir, "heads.onnx"), 1)
    print(f"model {meta.get('model', 'x3d_s')} | onnxruntime {ort.__version__} | source {a.source} | thr {a.thr} | threads {a.threads} | hop {a.hop} | gate {a.motion}")
    sb.run(None, {"clip": np.zeros((1, 3, NF, CROP, CROP), np.float32)})          # warmup
    q, aq, stop = queue.Queue(maxsize=1), queue.Queue(), threading.Event()
    st = dict(smax=0.0, clips=0, gated=0, dropped=0, alerts=0, frames=0, tmax=0.0, ms=[], e2e=[])
    th = [threading.Thread(target=capture, args=(q, stop, st), daemon=True),
          threading.Thread(target=infer, args=(sb, sh, meta, q, aq, stop, st), daemon=True),
          threading.Thread(target=alert, args=(aq, st), daemon=True)]
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
    th[1].join(10); th[2].join(5)
    ms, e2e = np.array(st["ms"] or [0]), np.array(st["e2e"] or [0])
    print(f"\n== SUMMARY == {time.time() - t0:.0f} s | frames {st['frames']} | clips {st['clips']} | gated {st['gated']} | "
          f"dropped {st['dropped']} | alerts {st['alerts']} | max score {st['smax']:.3g} | max temp {st['tmax']:.1f}C")
    print(f"per-clip compute: median {np.median(ms):.0f} | p95 {np.percentile(ms, 95):.0f} ms")
    print(f"clip-end se decision tak (e2e): median {np.median(e2e):.0f} | p95 {np.percentile(e2e, 95):.0f} ms"
          + ("  (fast mode mein ye matlab nahi rakhta)" if a.fast else ""))
    print("END OF REPORT")


main()
