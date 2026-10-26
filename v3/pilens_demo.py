#!/usr/bin/env python3
"""PiLENS live demo (Raspberry Pi 5): camera or video -> MoViNet-A0 -> 2-of-3 alert rule, with FPS and latency on screen.

Run on the Pi (inside the v3 folder, with the venv active):
  python pilens_demo.py --check                          # 1-minute self-test (PASS / WARN / FAIL list)
  python pilens_demo.py                                  # live camera in a window (q = quit)
  python pilens_demo.py --source samples                 # every video in a folder, one after another
  python pilens_demo.py --source Steal.mp4 --record      # one video, and also record the screen
  python pilens_demo.py --web 8000                       # view in a phone/laptop browser: http://<pi-ip>:8000
  python pilens_demo.py --max-sec 300                    # stop by itself after 5 minutes (writes the report)
  python pilens_demo.py --gpio                           # LED on GPIO 17 and buzzer on GPIO 18
Window keys: q or Esc = quit | s = screenshot | f = full screen
Output of every run: demo_out/<date_time>/  report.html, report.txt, report.json, summary.png,
                     scores.csv (per clip), fps.csv (per second), alerts/ (snapshots), demo.mp4 (--record)
One line per run in demo_out/all_runs.csv
"""
import os, sys, re, time, json, argparse, threading, queue, subprocess, collections, shutil, datetime, platform, glob, html, select
import importlib.util

if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"          # Pi desktop (Wayland) par opencv window Xwayland se khulti hai

import numpy as np
import cv2
import onnxruntime as ort

VERSION = "pilens_demo v1.3 (2026-10-12, 4 groups by default, --detail for 13 classes)"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
GROUPS = [("Violence", ("Abuse", "Assault", "Fighting", "Shooting", "Arrest")),
          ("Theft", ("Burglary", "Robbery", "Shoplifting", "Stealing")),
          ("Fire / damage", ("Arson", "Explosion", "Vandalism")),
          ("Road accident", ("RoadAccidents",))]
OP_NAMES = {"fpr5": "val FPR 5%", "fpr10": "val FPR 10%", "youden": "val Youden", "manual": "manual", "default": "default"}

ap = argparse.ArgumentParser(description="PiLENS live demo")
ap.add_argument("--source", default="camera", help="'camera', ek video file, ya videos wala folder")
ap.add_argument("--dir", default=None, help="model folder (default: export_movinet, phir export)")
ap.add_argument("--op", default="auto", choices=["auto", "fpr5", "fpr10", "youden"],
                help="threshold rule, val set par chuna hua (auto = fpr5, MoViNet par 0.935)")
ap.add_argument("--thr", type=float, default=None, help="threshold khud do (--op ko override karta hai)")
ap.add_argument("--threads", type=int, default=3, help="backbone threads (Pi 5 par best = 3)")
ap.add_argument("--hop", type=int, default=18, help="naya clip har itne frame par (18 = 0.6 s @ 30 FPS)")
ap.add_argument("--motion", type=float, default=0.003, help="motion gate (0 = band)")
ap.add_argument("--cooldown", type=float, default=10, help="ek alert ke baad itne sec naya alert nahi")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--rotate180", action="store_true", help="camera ulta laga ho to")
ap.add_argument("--max-sec", type=float, default=0, help="itne sec baad khud ruk jao (0 = q / Ctrl+C tak)")
ap.add_argument("--loop", action="store_true", help="video khatam ho to dobara chalao")
ap.add_argument("--no-window", action="store_true", help="window mat kholo (SSH par)")
ap.add_argument("--fullscreen", action="store_true")
ap.add_argument("--record", action="store_true", help="screen ki video demo.mp4 mein save karo")
ap.add_argument("--web", type=int, default=0, help="port (jaise 8000): browser mein live dekho")
ap.add_argument("--gpio", action="store_true", help="alert par LED (GPIO17) aur buzzer (GPIO18)")
ap.add_argument("--led-pin", type=int, default=17)
ap.add_argument("--buzzer-pin", type=int, default=18)
ap.add_argument("--out", default="demo_out")
ap.add_argument("--detail", action="store_true", help="hint ke saath 13 classes ke naam bhi dikhao (default: sirf 4 groups)")
ap.add_argument("--check", action="store_true", help="sirf system check karo aur ruk jao")
a = ap.parse_args()

cv2.setNumThreads(1)                     # opencv AI ke cores na khaye
LOCK = threading.Lock()


# ---------------------------------------------------------------- system info
def read_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        return float("nan")


def read_freq():
    try:
        return int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) / 1000
    except Exception:
        return float("nan")


def throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=3).strip().split("=")[-1]
    except Exception:
        return "n/a"


def ram_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")


def device_name():
    try:
        return open("/proc/device-tree/model").read().strip("\x00 \n")
    except Exception:
        return platform.node() or platform.machine()


def os_name():
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME"):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return platform.platform()


class CpuMeter:
    def __init__(self):
        self.prev = self._read()

    @staticmethod
    def _read():
        try:
            v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
        except Exception:
            return None

    def sample(self):
        cur = self._read()
        if not cur or not self.prev or cur[0] == self.prev[0]:
            return float("nan")
        busy = 100 * (1 - (cur[1] - self.prev[1]) / (cur[0] - self.prev[0]))
        self.prev = cur
        return busy


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def p95(v):
    return float(np.percentile(v, 95)) if len(v) else float("nan")


def fnum(x, fmt="{:.0f}", na="-"):
    return na if x is None or x != x else fmt.format(x)


# ---------------------------------------------------------------- model
def find_model_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    names = [a.dir] if a.dir else ["export_movinet", "export", os.path.join("results", "movinet_a0_v2", "export")]
    for n in names:
        for p in (n, os.path.join(here, n)):
            if os.path.isfile(os.path.join(p, "heads_meta.json")):
                return p
    return None


def make_session(path, threads):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class Model:
    def __init__(self, d, threads):
        self.dir = d
        self.meta = m = json.load(open(os.path.join(d, "heads_meta.json")))
        cl = m.get("clip", {})
        self.NF, self.STEP = int(cl.get("frames", 13)), int(cl.get("stride", 6))
        self.SIDE, self.CROP = int(cl.get("side", 182)), int(cl.get("crop", 160))
        self.MEAN, self.STD = float(cl.get("mean", 0.45)), float(cl.get("std", 0.225))
        self.COLOR = m.get("color", "rgb")      # "gray" = model grayscale par train hua (raat / IR ke liye)
        self.SPAN = (self.NF - 1) * self.STEP + 1
        self.name = m.get("model", "x3d_s")
        mv = re.search(r"movinet_(a\d)", self.name)
        self.pretty = (f"MoViNet-{mv.group(1).upper()}" if mv else ("X3D-S" if "x3d" in self.name else self.name)) + (" gray" if self.COLOR == "gray" else "")
        self.classes = list(m.get("classes", []))
        self.R = int(m.get("recent_clips", 5))
        self.bb_path = os.path.join(d, m.get("backbone", "backbone_x3ds.onnx"))
        self.hd_path = os.path.join(d, "heads.onnx")
        self.sb, self.sh = make_session(self.bb_path, threads), make_session(self.hd_path, 1)
        self.bb_in = self.sb.get_inputs()[0].name
        self.hd_in = [i.name for i in self.sh.get_inputs()]
        outs = [o.name for o in self.sh.get_outputs()]
        self.hd_out = ["binary_score", "class_logits"] if {"binary_score", "class_logits"} <= set(outs) else outs[:2]
        self.hist = []
        self.size_mb = (os.path.getsize(self.bb_path) + os.path.getsize(self.hd_path)) / 1e6

    def threshold(self):
        rec = self.meta.get("thr_reco", {}) or {}
        opts = {"fpr5": rec.get("val_FPR_5pct"), "fpr10": rec.get("val_FPR_10pct"), "youden": rec.get("youden")}
        if a.thr is not None:
            return float(a.thr), "manual"
        op = a.op
        if op == "auto":                    # X3D par fpr5 = 1.000 (saturate), wahan Youden
            op = "fpr5" if opts["fpr5"] is not None and opts["fpr5"] < 0.999 else "youden"
        if opts.get(op) is None:
            return (0.753 if "x3d" in self.name else 0.5), "default"
        return float(opts[op]), op

    def prep(self, fr):
        """frame -> model crop (RGB uint8): short side SIDE, center crop CROP (training jaisa)"""
        h, w = fr.shape[:2]; s = self.SIDE / min(h, w)
        fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = fr.shape[:2]; y0, x0 = (h - self.CROP) // 2, (w - self.CROP) // 2
        fr = fr[y0:y0 + self.CROP, x0:x0 + self.CROP]
        if self.COLOR == "gray":                # training jaisa: gray, 3 channel mein
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            return np.repeat(g[:, :, None], 3, 2)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    def crop_box(self, w, h):
        """original frame mein wo square jo model dekhta hai"""
        side = self.CROP * min(w, h) / self.SIDE
        return (w - side) / 2, (h - side) / 2, side

    def reset(self):
        self.hist = []

    def run(self, frames):
        x = frames.astype(np.float32) / 255
        if self.MEAN != 0 or self.STD != 1:
            x = (x - self.MEAN) / self.STD
        x = np.ascontiguousarray(x.transpose(3, 0, 1, 2)[None])
        f = self.sb.run(None, {self.bb_in: x})[0].astype(np.float32)
        self.hist = (self.hist + [f[0]])[-self.R:]
        rec = np.stack([self.hist[0]] * (self.R - len(self.hist)) + self.hist)[None].astype(np.float32)
        feeds = {self.hd_in[0]: f}
        if len(self.hd_in) > 1:
            feeds[self.hd_in[1]] = rec
        out = self.sh.run(self.hd_out, feeds)
        return float(np.ravel(out[0])[0]), (np.ravel(out[1]) if len(out) > 1 else None)

    def hint(self, logits):
        """category hint: top-3 classes + 4 crime groups (Normal class ho to hata do)"""
        if logits is None or not self.classes:
            return None
        z = np.exp(logits - logits.max()); z /= z.sum()
        keep = [i for i, c in enumerate(self.classes) if not c.lower().startswith("normal")]
        tot = float(sum(z[i] for i in keep)) or 1.0
        p = {self.classes[i]: float(z[i]) / tot for i in keep}
        top = sorted(p.items(), key=lambda kv: -kv[1])[:3]
        g = [(gn, sum(p.get(c, 0.0) for c in cs)) for gn, cs in GROUPS]
        g += [(c, v) for c, v in p.items() if not any(c in cs for _, cs in GROUPS)]
        g.sort(key=lambda kv: -kv[1])
        return dict(top3=[(c, round(v, 3)) for c, v in top], group=g[0][0], group_p=round(g[0][1], 3),
                    groups3=[(c, round(v, 3)) for c, v in g[:3]])


# ---------------------------------------------------------------- frame sources
def cam_fail(st, why, errf=None):
    tail = []
    if errf is not None:
        try:
            errf.seek(0); tail = errf.read().decode(errors="ignore").strip().splitlines()[-6:]
        except Exception:
            pass
    st["camera_error"] = why
    print(f"\nCAMERA PROBLEM: {why}." + ("\n  camera ne ye kaha:\n    " + "\n    ".join(tail) if tail else ""))
    print("  Karo: (1) pkill -f rpicam  (2) rpicam-hello --list-cameras  (3) ribbon cable check  (4) sudo reboot")


def frames_camera(stop, st):
    exe = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
    if exe is None:
        yield from frames_webcam(stop, st)
        return
    import tempfile
    w, h = a.width, a.height
    cmd = [exe, "-t", "0", "--nopreview", "--codec", "yuv420", "--width", str(w), "--height", str(h),
           "--framerate", str(a.fps), "-o", "-"]
    if a.rotate180:
        cmd += ["--rotation", "180"]
    n = w * h * 3 // 2
    errf = tempfile.TemporaryFile()
    # apna process group: Ctrl+C sirf is program ko lage, camera ko hum khud aaram se band karte hain
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=n * 2, start_new_session=True)
    st["source_desc"] = f"Camera {w}x{h} @ {a.fps:.0f} FPS"
    info = dict(name="camera", new=True)
    try:
        first = True
        while not stop.is_set():
            if not select.select([p.stdout], [], [], 20 if first else 5)[0]:
                cam_fail(st, "20 sec tak koi frame nahi aaya" if first else "frames aana ruk gaye", errf); return
            buf = p.stdout.read(n)
            if len(buf) < n:
                if not stop.is_set():
                    cam_fail(st, "camera stream band ho gayi", errf)
                return
            first = False
            fr = cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
            yield fr, time.time(), info
            info = dict(name="camera", new=False)
    finally:
        p.terminate()                                    # camera ko aaram se band karo (Ctrl+C se driver atakta tha)
        try:
            p.wait(3)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                pass
        st["camera_closed"] = time.time()


def frames_webcam(stop, st):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cam_fail(st, "rpicam-vid nahi mila aur USB/laptop camera bhi nahi khula"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    st["source_desc"] = "USB/laptop camera"
    info = dict(name="camera", new=True)
    try:
        while not stop.is_set():
            ok, fr = cap.read()
            if not ok:
                cam_fail(st, "camera se frame nahi aaya"); return
            yield fr, time.time(), info
            info = dict(name="camera", new=False)
    finally:
        cap.release()


def video_list(src):
    if os.path.isdir(src):
        return sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith(VIDEO_EXT))
    return [src]


def frames_files(stop, st):
    files = video_list(a.source)
    if not files:
        st["camera_error"] = f"is folder mein koi video nahi: {a.source}"
        print(st["camera_error"]); return
    st["source_desc"] = (f"{len(files)} videos from {a.source}" if len(files) > 1 else f"Video {os.path.basename(files[0])}")
    while not stop.is_set():
        played = 0
        for path in files:
            if stop.is_set():
                return
            if not os.path.exists(path):
                print(f"FILE NAHI MILI: {path}  (naam ke capital/small letters bhi check karo: ls)"); continue
            cap = cv2.VideoCapture(path)
            if not cap.isOpened() or not cap.read()[0]:
                print(f"VIDEO NAHI KHULI: {path} (codec masla; ffmpeg se H.264 mp4 bana lo)"); continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or a.fps
            if not 5 <= fps <= 120:
                fps = a.fps
            print(f"> video: {os.path.basename(path)} ({fps:.0f} FPS)")
            info = dict(name=os.path.basename(path), new=True)
            t0, i = time.time(), 0
            while not stop.is_set():
                ok, fr = cap.read()
                if not ok:
                    break
                wait = t0 + i / fps - time.time()      # camera ki tarah asli speed par
                if wait > 0:
                    time.sleep(wait)
                i += 1; played += 1
                yield fr, time.time(), info
                info = dict(name=info["name"], new=False)
            cap.release()
        if not played:
            st["camera_error"] = "koi video nahi chali (file ka naam/folder check karo: ls)"
            return
        if not a.loop:
            return


# ---------------------------------------------------------------- GPIO (optional)
class Gpio:
    def __init__(self, on):
        self.kind, self.dev = "off", None
        if not on:
            return
        try:
            from gpiozero import LED, Buzzer
            self.dev = (LED(a.led_pin), Buzzer(a.buzzer_pin)); self.kind = "gpiozero"
        except Exception as e:
            if shutil.which("pinctrl"):
                self.kind = "pinctrl"; self._pin(a.led_pin, 0); self._pin(a.buzzer_pin, 0)
            else:
                print(f"GPIO band: gpiozero nahi chala ({str(e)[:60]}) aur pinctrl bhi nahi mila")
        if self.kind != "off":
            print(f"GPIO: {self.kind} | LED GPIO{a.led_pin} | buzzer GPIO{a.buzzer_pin}")

    def _pin(self, pin, v):
        subprocess.run(["pinctrl", "set", str(pin), "op", "dh" if v else "dl"], capture_output=True, timeout=2)

    def _set(self, which, v):
        try:
            if self.kind == "gpiozero":
                d = self.dev[which]; d.on() if v else d.off()
            elif self.kind == "pinctrl":
                self._pin(a.led_pin if which == 0 else a.buzzer_pin, v)
        except Exception:
            pass

    def alert(self):
        if self.kind == "off":
            return

        def pattern():
            self._set(0, 1)
            for _ in range(3):
                self._set(1, 1); time.sleep(0.15); self._set(1, 0); time.sleep(0.15)
            time.sleep(2.5); self._set(0, 0)
        threading.Thread(target=pattern, daemon=True).start()

    def close(self):
        self._set(0, 0); self._set(1, 0)


# ---------------------------------------------------------------- web view (optional)
class Web:
    PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>PiLENS live</title><style>body{margin:0;background:#111;color:#ddd;font:15px sans-serif;text-align:center}
img{max-width:100%;height:auto}#s{padding:8px}</style></head><body><img src="/stream"><div id=s></div>
<script>setInterval(()=>fetch('/stats').then(r=>r.json()).then(j=>{document.getElementById('s').textContent=
`camera ${j.camera_fps} FPS | AI ${j.ai_ms} ms/clip | score ${j.score} | alerts ${j.alerts} | ${j.temp_c} C`}).catch(()=>{}),1000)</script>
</body></html>"""

    def __init__(self, port, stats_fn):
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        self.jpeg, self.clients, self.stopped, self.cond = None, 0, False, threading.Condition()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *x):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache"); self.end_headers()
                    with outer.cond:
                        outer.clients += 1
                    last = None
                    try:
                        while not outer.stopped:
                            with outer.cond:
                                outer.cond.wait(1.0); jpg = outer.jpeg
                            if jpg is None or jpg is last:
                                continue
                            last = jpg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with outer.cond:
                            outer.clients -= 1
                    return
                body, ctype = ((json.dumps(stats_fn()).encode(), "application/json") if self.path.startswith("/stats")
                               else (outer.PAGE.encode(), "text/html"))
                self.send_response(200); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0", port), H); self.srv.daemon_threads = True
        except OSError as e:
            self.srv = None
            print(f"WEB VIEW band: port {port} nahi khula ({e}); doosra port do, jaise --web 8080"); return
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        try:
            ips = subprocess.check_output(["hostname", "-I"], text=True, timeout=3).split()
        except Exception:
            ips = []
        print("WEB VIEW: browser mein kholo -> " + " | ".join(f"http://{ip}:{port}" for ip in (ips or ["<pi-ip>"])))

    def push(self, jpg):
        with self.cond:
            self.jpeg = jpg; self.cond.notify_all()

    def close(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()
        try:
            if self.srv is not None:
                self.srv.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------- pipeline threads
def put_latest(q, item, st, count=True):
    """AI peeche ho to purana clip chhod do (live mein latest hi kaam ka hai)"""
    while True:
        try:
            q.put_nowait(item); return
        except queue.Full:
            try:
                q.get_nowait()
                if count:
                    st["dropped"] += 1
            except queue.Empty:
                pass


def capture(model, q, shared, stop, st):
    src = frames_camera(stop, st) if a.source == "camera" else frames_files(stop, st)
    buf, mot = collections.deque(maxlen=model.SPAN), collections.deque(maxlen=a.hop)
    mog, n, nf = None, 0, 0
    try:
        for fr, tcap, info in src:
            if stop.is_set():
                break
            if info["new"]:                                  # nayi video: purane frames clip mein na milen
                buf.clear(); mot.clear(); nf = 0
                mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
            n += 1; nf += 1
            buf.append(model.prep(fr))
            mot.append(float((mog.apply(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (80, 60))) > 0).mean()))
            with LOCK:
                shared["frame"], shared["fidx"], shared["src"] = fr, n, info["name"]
                shared["buffer_fill"] = len(buf) / model.SPAN
                st["cam_ts"].append(tcap)
                st["first_ts"] = st.get("first_ts") or tcap; st["last_ts"] = tcap
            if len(buf) == model.SPAN and nf % a.hop == 0:
                put_latest(q, dict(frames=np.stack(list(buf)[::model.STEP]), motion=float(np.mean(mot)), t_end=tcap,
                                   last=fr, idx=n, warm=nf > 60, src=info["name"]), st)
    finally:
        src.close()
        st["frames"] = n
        put_latest(q, None, st, count=False)


def infer(model, thr, q, aq, shared, stop, st, gpio, out):
    flags, last_alert, cur = collections.deque(maxlen=3), -1e9, None
    csv = open(os.path.join(out, "scores.csv"), "w")
    csv.write("t_sec,frame,source,motion,gated,score,flags,alert,ai_ms,e2e_ms,temp_c,hint_group,hint_top1\n")
    k = 0
    while True:
        item = q.get()
        if item is None:
            break
        if item["src"] != cur:
            cur = item["src"]; flags.clear(); model.reset()
        t0 = time.perf_counter(); k += 1
        gated = a.motion > 0 and item["warm"] and item["motion"] < a.motion
        score, logits = 0.0, None
        if not gated:
            score, logits = model.run(item["frames"])
        ms = (time.perf_counter() - t0) * 1000
        e2e = (time.time() - item["t_end"]) * 1000
        flags.append(score >= thr)
        hint = model.hint(logits) if logits is not None else None
        now = time.time(); ts = now - st["t0"]
        alert = sum(flags) >= 2 and logits is not None and now - last_alert > a.cooldown
        tc = read_temp()
        with LOCK:
            st["clips"] += 1
            if gated:
                st["gated"] += 1
            else:
                st["ms"].append(ms); st["e2e"].append(e2e); st["scores"].append(score)
            st["smax"] = max(st["smax"], score)
            shared["ai"] = dict(score=score, gated=gated, flags=list(flags), ms=ms, e2e=e2e, motion=item["motion"],
                                hint=hint, t=now)
            shared["hist"].append((ts, score, gated, alert)); st["hist_all"].append((ts, score, gated, alert))
            if alert:
                last_alert = now; st["alerts"] += 1
                shared["alert"] = dict(until=now + 4.0, score=score, hint=hint, n=st["alerts"], t=ts, src=item["src"])
        top1 = f"{hint['top3'][0][0]} {hint['top3'][0][1]:.2f}" if hint else ""
        csv.write(f"{ts:.2f},{item['idx']},{item['src']},{item['motion']:.4f},{int(gated)},{score:.4f},{sum(flags)},"
                  f"{int(alert)},{ms:.0f},{e2e:.0f},{tc:.1f},{hint['group'] if hint else ''},{top1}\n")
        if k % 5 == 0:
            csv.flush()
        if alert:
            gpio.alert()
            aq.put(dict(frame=item["last"], score=score, hint=hint, idx=item["idx"], t=ts, src=item["src"], n=st["alerts"]))
    csv.close()
    aq.put(None)


def alert_saver(aq, out, st):
    os.makedirs(os.path.join(out, "alerts"), exist_ok=True)
    while True:
        it = aq.get()
        if it is None:
            break
        img = f"alerts/alert_{it['n']:02d}_frame.jpg"
        cv2.imwrite(os.path.join(out, img), it["frame"])
        h = it["hint"] or {}
        rec = dict(n=it["n"], t_sec=round(it["t"], 1), source=it["src"], score=round(it["score"], 4),
                   group=h.get("group"), group_p=h.get("group_p"), top3=h.get("top3"), image=img,
                   screen=f"alerts/alert_{it['n']:02d}_screen.jpg")
        st["alert_list"].append(rec)
        with open(os.path.join(out, "alerts.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("\n" + "!" * 64 + f"\n  ALERT {it['n']}  t={it['t']:.1f}s  score={it['score']:.3f}  ({it['src']})")
        if h:
            sh = h["top3"] if a.detail else h["groups3"]
            print(f"  type hint: {h['group']} {h['group_p']:.0%} | " + ", ".join(f"{c} {p:.2f}" for c, p in sh))
        print(f"  snapshot: {os.path.join(out, img)}\n" + "!" * 64 + "\n")


# ---------------------------------------------------------------- drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, DIM = (235, 235, 235), (160, 160, 160), (90, 90, 90)
GREEN, AMBER, RED, BLUE = (90, 200, 90), (0, 180, 255), (60, 60, 235), (235, 170, 70)
CW, CH, VW, VH = 960, 600, 640, 480


def put(img, s, x, y, sc=0.5, col=WHITE, th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, sc, col, th, cv2.LINE_AA)


def put_r(img, s, xr, y, sc=0.5, col=WHITE, th=1):
    (w, _), _ = cv2.getTextSize(s, FONT, sc, th)
    put(img, s, xr - w, y, sc, col, th)


def shade(img, x0, y0, x1, y1, col=(0, 0, 0), alpha=0.55):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array(col, np.float32) * alpha).astype(np.uint8)


def chip(img, s, x_right, y, bg, sc=0.6):
    (w, h), _ = cv2.getTextSize(s, FONT, sc, 2)
    cv2.rectangle(img, (x_right - w - 20, y - h - 10), (x_right, y + 8), bg, -1)
    put(img, s, x_right - w - 10, y, sc, (255, 255, 255), 2)


def fit_frame(fr):
    h, w = fr.shape[:2]
    if (w, h) == (VW, VH):
        return fr, 1.0, 0, 0
    s = min(VW / w, VH / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    view = np.zeros((VH, VW, 3), np.uint8)
    x0, y0 = (VW - nw) // 2, (VH - nh) // 2
    view[y0:y0 + nh, x0:x0 + nw] = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    return view, s, x0, y0


class Painter:
    """960x600 screen: video (640x480) + score timeline below it, stats panel on the right"""
    def __init__(self, model, thr, op):
        self.m, self.thr, self.op = model, thr, op
        self.base = b = np.zeros((CH, CW, 3), np.uint8)
        b[:, VW:] = (30, 30, 30); b[VH:, :VW] = (20, 20, 20)
        dev = device_name()
        put(b, "PiLENS", VW + 16, 36, 0.95, WHITE, 2)
        put(b, f"{model.pretty} | {('Raspberry Pi 5' if 'Raspberry Pi 5' in dev else dev)[:24]}", VW + 16, 58, 0.42, GREY)
        for y, s in ((92, "SPEED"), (292, "AI DECISION"), (500, "SYSTEM")):
            cv2.line(b, (VW + 12, y - 20), (CW - 12, y - 20), (64, 64, 64), 1)
            put(b, s, VW + 16, y, 0.45, BLUE, 1)
        for y, s in ((120, "Camera"), (148, "AI time per clip"), (174, "AI decisions"), (200, "Real-time margin"),
                     (226, "Skipped (AI busy)"), (252, "Screen"), (320, "Anomaly score"), (388, "2-of-3 check"),
                     (414, "Motion"), (440, "Type hint"), (526, "CPU / temp"), (552, "Alerts / time")):
            put(b, s, VW + 16, y, 0.45, GREY)
        put(b, f"alert threshold {thr:.3f} ({OP_NAMES.get(op, op)})", VW + 16, 362, 0.4, GREY)
        put(b, "q quit | s screenshot | f full screen", VW + 16, 588, 0.38, DIM)
        put(b, "Anomaly score, last 60 s (dot = one 2.4 s clip, red line = threshold)", 10, VH + 18, 0.4, GREY)

    def draw(self, fr, s):
        c = self.base.copy()
        m, thr, now = self.m, self.thr, time.time()
        ai, al = s["ai"], s["alert"]
        alert_on = al is not None and now < al["until"]
        # --- video
        if m.COLOR == "gray":                   # gray model: screen par bhi gray dikhao (jo model dekhta hai)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); fr = np.repeat(g[:, :, None], 3, 2)
        view, sc, x0, y0 = fit_frame(fr)
        c[:VH, :VW] = view
        h, w = fr.shape[:2]; bx, by, bs = m.crop_box(w, h)
        p0 = (int(x0 + bx * sc), int(y0 + by * sc)); p1 = (int(x0 + (bx + bs) * sc), int(y0 + (by + bs) * sc))
        cv2.rectangle(c, p0, p1, (200, 200, 200), 1)
        put(c, "AI view", p0[0] + 6, p1[1] - 8, 0.42, (220, 220, 220))
        shade(c, 0, 0, 470, 34)
        put(c, (f"LIVE  {s['src_desc']}" if a.source == "camera" else f"VIDEO  {s['src'][:22]}") + f"  |  {m.pretty}", 10, 23, 0.55, WHITE, 1)
        if alert_on:
            state, col = "ALERT", RED
        elif ai is None:
            state, col = f"STARTING {s['buffer_fill']:.0%}", (110, 110, 110)
        elif ai["gated"]:
            state, col = "IDLE: no motion", (110, 110, 110)
        elif sum(ai["flags"]) >= 1:
            state, col = f"CHECKING {sum(ai['flags'])}/3", AMBER
        else:
            state, col = "NORMAL", (60, 150, 60)
        chip(c, state, VW - 10, 28, col)
        if alert_on:
            cv2.rectangle(c, (3, 3), (VW - 4, VH - 4), RED, 6)
            shade(c, 0, VH - 74, VW, VH, (40, 40, 200), 0.75)
            put(c, f"SUSPICIOUS ACTIVITY   score {al['score']:.3f}", 16, VH - 44, 0.75, WHITE, 2)
            hn = al["hint"]
            if hn:
                put(c, f"type hint: {hn['group']} {hn['group_p']:.0%}   (" +
                    ", ".join(f"{cn} {p:.0%}" for cn, p in (hn["top3"] if a.detail else hn["groups3"][1:])[:2]) + ")", 16, VH - 16, 0.55, WHITE, 1)
        # --- speed
        X = CW - 16
        put_r(c, f"{fnum(s['cam_fps'], '{:.1f}')} FPS", X, 122, 0.75, WHITE, 2)
        put_r(c, f"{fnum(ai and (None if ai['gated'] else ai['ms']))} ms  (med {fnum(s['ms_med'])})", X, 148, 0.48)
        put_r(c, f"every {a.hop / max(s['src_fps'], 1):.1f} s", X, 174, 0.48)
        put_r(c, f"{fnum(s['margin'], '{:.1f}')}x", X, 200, 0.48, GREEN if s["margin"] == s["margin"] and s["margin"] > 1.5 else WHITE)
        put_r(c, f"{s['dropped']}", X, 226, 0.48, GREEN if s["dropped"] == 0 else AMBER)
        put_r(c, f"{fnum(s['view_fps'], '{:.0f}')} FPS", X, 252, 0.48, GREY)
        # --- AI decision
        sco = 0.0 if ai is None else ai["score"]
        put_r(c, "-" if ai is None or ai["gated"] else f"{sco:.3f}", X, 322, 0.7, RED if sco >= thr else WHITE, 2)
        bx0, bx1, byy = VW + 16, CW - 16, 334
        cv2.rectangle(c, (bx0, byy), (bx1, byy + 12), (70, 70, 70), -1)
        if ai is not None and not ai["gated"]:
            cv2.rectangle(c, (bx0, byy), (int(bx0 + (bx1 - bx0) * min(sco, 1)), byy + 12), RED if sco >= thr else GREEN, -1)
        tx = int(bx0 + (bx1 - bx0) * thr)
        cv2.line(c, (tx, byy - 4), (tx, byy + 16), (80, 80, 255), 2)
        fl = ai["flags"] if ai else []
        for i in range(3):
            v = fl[i] if i < len(fl) else None
            cv2.circle(c, (X - 8 - (2 - i) * 22, 383), 7, RED if v else ((90, 90, 90) if v is None else GREEN), -1)
        mo = 0 if ai is None else ai["motion"]
        put_r(c, f"{mo * 100:.1f}%  (gate {a.motion * 100:.1f}%)", X, 414, 0.48, GREY if ai and ai["gated"] else WHITE)
        hn = (al["hint"] if alert_on else (ai["hint"] if ai and sum(ai["flags"]) >= 1 else None))
        put_r(c, f"{hn['group']} {hn['group_p']:.0%}" if hn else "-", X, 440, 0.5, AMBER if hn else GREY)
        if hn:
            put_r(c, ", ".join(f"{cn} {p:.0%}" for cn, p in (hn["top3"] if a.detail else hn["groups3"][1:])[:2]), X, 462, 0.4, GREY)
        # --- system
        put_r(c, f"{fnum(s['cpu'])}%  |  {fnum(s['temp'], '{:.1f}')} C", X, 526, 0.48)
        el = int(now - s["t0"])
        put_r(c, f"{s['alerts']}  |  {el // 60:02d}:{el % 60:02d}", X, 552, 0.48, RED if s["alerts"] else WHITE)
        # --- score timeline (under the video)
        gx0, gx1, gy0, gy1 = 40, VW - 16, VH + 30, CH - 20
        cv2.rectangle(c, (gx0, gy0), (gx1, gy1), (45, 45, 45), 1)
        put_r(c, "1", gx0 - 6, gy0 + 6, 0.38, GREY); put_r(c, "0", gx0 - 6, gy1 + 2, 0.38, GREY)
        ty = int(gy1 - (gy1 - gy0) * thr)
        for xx in range(gx0, gx1, 12):
            cv2.line(c, (xx, ty), (min(xx + 6, gx1), ty), (80, 80, 255), 1)
        put(c, "-60 s", gx0, CH - 5, 0.36, DIM); put_r(c, "now", gx1, CH - 5, 0.36, DIM)
        tnow = now - s["t0"]
        pts = []
        for ts, scv, g, alt in s["hist"]:
            if ts < tnow - 60:
                continue
            xx = int(gx1 - (tnow - ts) / 60 * (gx1 - gx0))
            if g:
                cv2.line(c, (xx, gy1 - 3), (xx, gy1), DIM, 1); continue
            yy = int(gy1 - (gy1 - gy0) * min(max(scv, 0), 1))
            pts.append((xx, yy))
            if alt:
                cv2.line(c, (xx, gy0), (xx, gy1), RED, 1)
                (put if xx + 50 < gx1 else put_r)(c, "ALERT", xx + 3 if xx + 50 < gx1 else xx - 3, gy0 + 12, 0.38, RED)
        if len(pts) > 1:
            cv2.polylines(c, [np.array(pts, np.int32)], False, (120, 120, 120), 1, cv2.LINE_AA)
        for (xx, yy), (ts, scv, g, alt) in zip(pts, [h_ for h_ in s["hist"] if h_[0] >= tnow - 60 and not h_[2]]):
            cv2.circle(c, (xx, yy), 3, RED if scv >= thr else GREEN, -1)
        return c


# ---------------------------------------------------------------- report
def svg_chart(xs, ys, thr=None, marks=(), ymax=None, ylab="", h=200, w=860, col="#2b7bd6", dots=True):
    if not xs:
        return "<p>(no data)</p>"
    L, R, T, B = 56, 16, 14, 30
    xmax = max(max(xs), 1e-6); ymax = ymax or max(max(ys) * 1.15, 1e-6)
    X = lambda v: L + (w - L - R) * v / xmax
    Y = lambda v: T + (h - T - B) * (1 - min(max(v, 0), ymax) / ymax)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;background:#fff;border:1px solid #ddd">']
    for frac in (0, 0.5, 1):
        yv = ymax * frac
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="#eee"/>'
                 f'<text x="{L - 6}" y="{Y(yv) + 4:.1f}" font-size="11" text-anchor="end" fill="#666">{yv:.3g}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1):
        xv = xmax * frac
        o.append(f'<text x="{X(xv):.1f}" y="{h - 10}" font-size="11" text-anchor="middle" fill="#666">{xv:.0f} s</text>')
    if thr is not None:
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(thr):.1f}" y2="{Y(thr):.1f}" stroke="#d33" stroke-dasharray="5 4"/>'
                 f'<text x="{w - R}" y="{Y(thr) - 4:.1f}" font-size="11" text-anchor="end" fill="#d33">threshold {thr:.3f}</text>')
    for mx in marks:
        o.append(f'<line x1="{X(mx):.1f}" x2="{X(mx):.1f}" y1="{T}" y2="{h - B}" stroke="#d33"/>')
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.2"/>')
    if dots and len(xs) <= 1500:
        o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2" fill="{"#d33" if thr is not None and y >= thr else col}"/>'
              for x, y in zip(xs, ys)]
    o.append("</svg>")
    return "".join(o)


def clean(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def write_report(out, R, fps_rows, hist, model, thr):
    json.dump(clean(R), open(os.path.join(out, "report.json"), "w"), indent=1)
    v = R["verdicts"]
    lines = [f"PiLENS DEMO REPORT  ({R['when']})", "=" * 60,
             f"Device      : {R['device']} | {R['os']}",
             f"Model       : {R['model']} ({R['model_mb']:.1f} MB ONNX) | onnxruntime {R['onnxruntime']} | {R['threads']} threads",
             f"Source      : {R['source']}",
             f"Threshold   : {R['thr']:.3f} ({R['thr_rule']}) | alert = 2 of last 3 clips | cooldown {a.cooldown:.0f} s",
             f"Duration    : {R['duration_s']:.0f} s ({R['duration_s'] / 60:.1f} min)", "",
             "SPEED",
             f"  camera / video FPS      : {fnum(R['camera_fps'], '{:.1f}')} (lowest 1-s window {fnum(R['camera_fps_min'], '{:.1f}')}), frames {R['frames']}",
             f"  AI time per clip        : median {fnum(R['ai_ms_median'])} ms | p95 {fnum(R['ai_ms_p95'])} ms | max {fnum(R['ai_ms_max'])} ms",
             f"  clip end -> decision    : median {fnum(R['e2e_ms_median'])} ms | p95 {fnum(R['e2e_ms_p95'])} ms",
             f"  AI decision every       : {R['hop_ms']:.0f} ms (each clip = {R['clip_sec']:.1f} s of video) | real-time margin {fnum(R['margin'], '{:.1f}')}x",
             f"  clips                   : {R['clips']} total | {R['clips_inferred']} AI | {R['clips_gated']} skipped by motion gate | {R['clips_dropped']} skipped (AI busy)",
             f"  screen FPS              : {fnum(R['screen_fps'], '{:.1f}')}", "",
             "SYSTEM",
             f"  CPU use                 : mean {fnum(R['cpu_mean'])}% | max {fnum(R['cpu_max'])}%",
             f"  temperature             : start {fnum(R['temp_start'], '{:.1f}')} C | max {fnum(R['temp_max'], '{:.1f}')} C | throttled {R['throttled_end']}",
             f"  program RAM             : max {fnum(R['ram_max_mb'])} MB", "",
             "ALERTS",
             f"  alerts                  : {R['alerts']} | max score {R['score_max']:.4f} | AI clips above threshold {R['clips_above_thr']}"]
    for al in R["alert_list"]:
        tops = ", ".join(f"{c} {p:.2f}" for c, p in (al.get("top3") or []))
        lines.append(f"   #{al['n']} t={al['t_sec']}s {al['source']} score {al['score']:.3f} | hint {al.get('group')} ({tops})")
    lines += ["", "CHECKS"] + [f"  [{s}] {t}" for s, t in v] + ["", "Note: " + R["note"], "END OF REPORT"]
    txt = "\n".join(lines)
    open(os.path.join(out, "report.txt"), "w").write(txt + "\n")

    xs = [h[0] for h in hist if not h[2]]; ys = [h[1] for h in hist if not h[2]]
    marks = [h[0] for h in hist if h[3]]
    fp = [(r[0], r[1]) for r in fps_rows[2:] if r[1] == r[1]]; tp = [(r[0], r[4]) for r in fps_rows if r[4] == r[4]]
    fx, fy = [x for x, _ in fp], [y for _, y in fp]; tx, ty = [x for x, _ in tp], [y for _, y in tp]
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(val))}</td></tr>" for k, val in [
        ("Duration", f"{R['duration_s']:.0f} s"), ("Source", R["source"]), ("Model", f"{R['model']} ({R['model_mb']:.1f} MB)"),
        ("Camera / video FPS", f"{fnum(R['camera_fps'], '{:.1f}')} (lowest 1-s window {fnum(R['camera_fps_min'], '{:.1f}')})"),
        ("AI time per clip", f"median {fnum(R['ai_ms_median'])} ms, p95 {fnum(R['ai_ms_p95'])} ms"),
        ("Clip end to decision", f"median {fnum(R['e2e_ms_median'])} ms, p95 {fnum(R['e2e_ms_p95'])} ms"),
        ("AI decision every", f"{R['hop_ms']:.0f} ms (real-time margin {fnum(R['margin'], '{:.1f}')}x)"),
        ("Clips", f"{R['clips']} total, {R['clips_inferred']} AI, {R['clips_gated']} motion-gated, {R['clips_dropped']} skipped"),
        ("CPU / temperature", f"mean {fnum(R['cpu_mean'])}%, max {fnum(R['temp_max'], '{:.1f}')} C, throttled {R['throttled_end']}"),
        ("Alerts", f"{R['alerts']} (threshold {R['thr']:.3f}, {R['thr_rule']}, 2 of 3 clips)")])
    checks = "".join(f"<li><b>{s}</b> {html.escape(t)}</li>" for s, t in v)
    gal = "".join(f"<figure><img src='{al['screen']}' onerror=\"this.src='{al['image']}'\"><figcaption>Alert {al['n']} at "
                  f"{al['t_sec']} s, {html.escape(str(al['source']))}, score {al['score']:.3f}, hint {html.escape(str(al.get('group')))}"
                  f"</figcaption></figure>" for al in R["alert_list"]) or "<p>No alerts in this run.</p>"
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>PiLENS demo report</title><style>body{{font:15px/1.45 sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #eee;padding:6px 8px}}td:first-child{{color:#555;width:38%}}
figure{{display:inline-block;margin:6px;width:420px;max-width:100%}}figure img{{width:100%}}h2{{margin-top:28px}}</style></head><body>
<h1>PiLENS demo report</h1><p>{html.escape(R['when'])} | {html.escape(R['device'])} | {html.escape(R['os'])}</p>
<img src="summary.png" style="width:100%;max-width:900px">
<h2>Results</h2><table>{rows}</table><h2>Checks</h2><ul>{checks}</ul>
<h2>Anomaly score per clip</h2><p style="color:#666">One dot per AI clip (2.4 s of video, every 0.6 s). Dashed line = alert threshold, solid red line = alert.</p>{svg_chart(xs, ys, thr=thr, marks=marks, ymax=1.0, ylab="score (red line = threshold, red bar = alert)")}
<h2>Camera FPS per second</h2>{svg_chart(fx, fy, ymax=max(fy + [a.fps]) * 1.1 if fy else None, ylab="frames per second", col="#2a9d55", dots=False)}
<h2>CPU temperature</h2>{svg_chart(tx, ty, ymax=85, ylab="deg C", col="#e07b39", dots=False) if ty else "<p>(not available)</p>"}
<h2>Alerts</h2>{gal}<p style="color:#666">{html.escape(R['note'])}</p></body></html>"""
    open(os.path.join(out, "report.html"), "w").write(page)
    return txt


def summary_image(R):
    img = np.full((CH, CW, 3), 30, np.uint8)
    put(img, "PiLENS demo summary", 30, 50, 1.0, WHITE, 2)
    put(img, f"{R['when']} | {R['model']} | {R['source'][:50]}", 30, 80, 0.5, GREY)
    big = [("Camera FPS", fnum(R["camera_fps"], "{:.1f}")), ("AI ms / clip", fnum(R["ai_ms_median"])),
           ("Decision delay ms", fnum(R["e2e_ms_median"])), ("Alerts", str(R["alerts"]))]
    for i, (k, val) in enumerate(big):
        x = 30 + i * 230
        cv2.rectangle(img, (x, 110), (x + 210, 230), (50, 50, 50), -1)
        put(img, val, x + 16, 185, 1.4, RED if k == "Alerts" and R["alerts"] else WHITE, 3)
        put(img, k, x + 16, 215, 0.5, GREY)
    y = 280
    for s, t in R["verdicts"]:
        put(img, f"[{s}]", 30, y, 0.55, GREEN if s == "PASS" else (AMBER if s in ("INFO", "WARN") else RED), 2)
        put(img, t[:95], 120, y, 0.5, WHITE); y += 32
    put(img, f"Duration {R['duration_s']:.0f} s | clips {R['clips']} (AI {R['clips_inferred']}, motion-gated {R['clips_gated']})"
             f" | max temp {fnum(R['temp_max'], '{:.1f}')} C | CPU {fnum(R['cpu_mean'])}%", 30, CH - 30, 0.5, GREY)
    return img


# ---------------------------------------------------------------- check mode
def run_check(out):
    res = []

    def add(status, what, detail="", fix=""):
        res.append((status, what, detail, fix))
        print(f"[{status:4s}] {what}: {detail}" + (f"\n        KARO: {fix}" if fix else ""))

    print(f"SCRIPT VERSION: {VERSION}  |  SYSTEM CHECK\n")
    add("PASS", "Device", f"{device_name()} | {os_name()} | {platform.machine()} | Python {platform.python_version()}")
    add("PASS", "Libraries", f"numpy {np.__version__} | onnxruntime {ort.__version__} | opencv {cv2.__version__}")
    d = find_model_dir()
    if d is None:
        add("FAIL", "Model folder", "export_movinet/ (ya export/) nahi mila",
            "PiLENS folder mein export_movinet/ rakho (andar backbone_movinet_a0_v2.onnx, heads.onnx, heads_meta.json)")
    else:
        try:
            m = Model(d, a.threads)
            thr, op = m.threshold()
            add("PASS", "Model", f"{m.pretty} from {d}/ ({m.size_mb:.1f} MB) | clip {m.NF} x {m.CROP} | threshold {thr:.3f} ({OP_NAMES.get(op, op)})")
            x = np.random.randint(0, 255, (m.NF, m.CROP, m.CROP, 3), np.uint8)
            for _ in range(2):
                m.run(x)
            t = []
            for _ in range(10):
                t0 = time.perf_counter(); m.run(x); t.append((time.perf_counter() - t0) * 1000)
            hop_ms = a.hop / a.fps * 1000
            st = "PASS" if med(t) < hop_ms * 0.67 else ("WARN" if med(t) < hop_ms else "FAIL")
            add(st, "AI speed", f"median {med(t):.0f} ms per clip (budget {hop_ms:.0f} ms) on {a.threads} threads",
                "" if st == "PASS" else "doosre bhaari programs (browser) band karo; fan check karo")
        except Exception as e:
            add("FAIL", "Model load", str(e)[:160], "files dobara copy karo (zip theek se unzip hui?)")
    gui = [ln for ln in cv2.getBuildInformation().splitlines() if ln.strip().startswith("GUI")]
    gui_ok = bool(gui) and "NONE" not in gui[0].upper()
    disp = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if gui_ok and disp:
        add("PASS", "Window", f"opencv GUI ({gui[0].split(':')[-1].strip()}) aur desktop display mila")
    elif not gui_ok:
        add("WARN", "Window", "opencv 'headless' hai, window nahi khulegi",
            "pip uninstall -y opencv-python-headless && pip install opencv-python   (ya --web 8000 use karo)")
    else:
        add("WARN", "Window", "desktop display nahi mila (SSH?)", "Pi ke desktop par terminal kholo, ya --no-window --web 8000")
    cam_tool = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    if cam_tool:
        try:
            busy = subprocess.run(["pgrep", "-f", "rpicam-vid|libcamera-vid"], capture_output=True, text=True).stdout.split()
            if busy:
                add("WARN", "Camera busy", f"pehle se camera process chal raha hai (pid {' '.join(busy)})", "pkill -f rpicam")
            r = subprocess.run([cam_tool, "--list-cameras"], capture_output=True, text=True, timeout=15)
            cams = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if re.match(r"^\d+\s*:", ln.strip())]
            if cams:
                add("PASS", "Camera", cams[0][:90])
            else:
                add("FAIL", "Camera", "koi camera nahi mila", "ribbon cable check karo (Pi band karke), phir sudo reboot")
        except subprocess.TimeoutExpired:
            add("FAIL", "Camera", "camera tool atak gaya", "sudo reboot")
    else:
        add("WARN", "Camera", "rpicam-hello nahi mila (ye Raspberry Pi nahi?)", "video file se demo chalao: --source video.mp4")
    tc, th = read_temp(), throttled()
    add("PASS" if (tc != tc or tc < 70) and th in ("0x0", "n/a") else "WARN", "Temperature",
        f"{fnum(tc, '{:.1f}')} C | throttled {th}", "" if th in ("0x0", "n/a") else "power adapter (5V 5A) aur fan check karo")
    free = shutil.disk_usage(".").free / 1e9
    add("PASS" if free > 2 else "WARN", "Disk", f"{free:.1f} GB khali", "" if free > 2 else "purani recordings delete karo")
    gp = "gpiozero" if importlib.util.find_spec("gpiozero") else ("pinctrl" if shutil.which("pinctrl") else "none")
    add("INFO", "GPIO (optional)", f"{gp} | --gpio se LED GPIO{a.led_pin} aur buzzer GPIO{a.buzzer_pin}")
    vids = [f for f in glob.glob("*") + glob.glob("samples/*") if f.lower().endswith(VIDEO_EXT)]
    add("INFO", "Videos here", ", ".join(sorted(vids)[:8]) or "koi video nahi (demo ke liye UCF clips samples/ mein rakho)")
    bad = [r for r in res if r[0] == "FAIL"]
    print("\n" + ("SAB THEEK HAI. Ab chalao:  python pilens_demo.py" if not bad else
                  f"{len(bad)} cheez(ein) theek karni hain (upar KARO dekho), phir --check dobara."))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"check_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"), "w") as f:
        f.write("\n".join(f"[{s}] {w}: {dd}" + (f" | fix: {fx}" if fx else "") for s, w, dd, fx in res) + "\n")
    return 1 if bad else 0


# ---------------------------------------------------------------- main
def main():
    if a.check:
        sys.exit(run_check(a.out))
    d = find_model_dir()
    if d is None:
        print("MODEL FOLDER NAHI MILA: export_movinet/ (ya export/) is folder mein rakho, ya --dir dekar batao.")
        sys.exit(1)
    print(f"SCRIPT VERSION: {VERSION}")
    try:
        model = Model(d, a.threads)
    except Exception as e:
        print(f"MODEL LOAD NAHI HUA ({d}): {e}\nFiles dobara copy karo; --check chala ke dekho."); sys.exit(1)
    thr, op = model.threshold()
    stamp = datetime.datetime.now()
    out = os.path.join(a.out, stamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"model: {model.pretty} | clip {model.NF} x {model.CROP} | threshold {thr:.3f} ({OP_NAMES.get(op, op)}) | "
          f"threads {a.threads} | hop {a.hop} | motion gate {a.motion} | source {a.source}")
    print(f"output folder: {out}")
    model.run(np.zeros((model.NF, model.CROP, model.CROP, 3), np.uint8)); model.reset()   # warmup

    st = dict(t0=time.time(), clips=0, gated=0, dropped=0, alerts=0, frames=0, smax=0.0, ms=[], e2e=[], scores=[],
              cam_ts=collections.deque(maxlen=400), alert_list=[], camera_error=None, source_desc=a.source, hist_all=[])
    shared = dict(frame=None, fidx=0, src="", ai=None, alert=None, hist=collections.deque(maxlen=200), buffer_fill=0.0)
    temp_start, thr_start = read_temp(), throttled()
    gpio = Gpio(a.gpio)
    q, aq, stop = queue.Queue(maxsize=1), queue.Queue(), threading.Event()
    painter = Painter(model, thr, op)
    cpu = CpuMeter()
    live = dict(cpu=float("nan"), temp=temp_start, cam_fps=float("nan"), view_fps=float("nan"))

    def stats_now():
        with LOCK:
            ai = shared["ai"]
            return dict(camera_fps=fnum(live["cam_fps"], "{:.1f}"), ai_ms=fnum(med(st["ms"][-50:])), alerts=st["alerts"],
                        score=("-" if not ai or ai["gated"] else f"{ai['score']:.3f}"), temp_c=fnum(live["temp"], "{:.1f}"))
    web = Web(a.web, stats_now) if a.web else None

    win = None
    if not a.no_window:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("Desktop display nahi mila (SSH?): window ke baghair chal raha hun. Live dekhna ho to --web 8000 lagao.")
        else:
            try:
                win = "PiLENS demo"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, CW, CH)
                if a.fullscreen:
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except cv2.error:
                win = None
                print("WINDOW NAHI KHULI: opencv 'headless' install hai. Ek dafa ye chalao:\n"
                      "  pip uninstall -y opencv-python-headless && pip install opencv-python\n"
                      "Abhi window ke baghair chal raha hun (report phir bhi banegi).")
    writer, rec_next = None, 0.0
    if a.record:
        writer = cv2.VideoWriter(os.path.join(out, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (CW, CH))
        if not writer.isOpened():
            print("recording shuru nahi hui (codec masla), baqi demo chalta rahega"); writer = None

    th = [threading.Thread(target=capture, args=(model, q, shared, stop, st), daemon=True),
          threading.Thread(target=infer, args=(model, thr, q, aq, shared, stop, st, gpio, out), daemon=True),
          threading.Thread(target=alert_saver, args=(aq, out, st), daemon=True)]
    for t in th:
        t.start()
    print("chal raha hai..." + (" (window par q dabao band karne ke liye)" if win else " (Ctrl+C ek dafa dabao band karne ke liye)"))

    fps_rows, view_ts = [], collections.deque(maxlen=120)
    last_fidx, last_alert_n, next_sec, next_print, last_jpg = -1, 0, time.time() + 1, time.time() + 5, 0.0
    stop_reason, canvas = None, None
    try:
        while True:
            now = time.time()
            if a.max_sec and now - st["t0"] >= a.max_sec:
                stop_reason = f"--max-sec {a.max_sec:.0f} poore"; break
            if not th[1].is_alive():
                stop_reason = st["camera_error"] or "video khatam"; break
            with LOCK:
                fr, fidx = shared["frame"], shared["fidx"]
                cam = list(st["cam_ts"])
                s = dict(ai=shared["ai"], alert=shared["alert"], hist=list(shared["hist"]), src=shared["src"],
                         buffer_fill=shared["buffer_fill"], dropped=st["dropped"], alerts=st["alerts"],
                         ms_med=med(st["ms"][-50:]), t0=st["t0"])
            recent = [t for t in cam if t > now - 2.0]
            live["cam_fps"] = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 and recent[-1] > recent[0] else float("nan")
            vr = [t for t in view_ts if t > now - 2.0]
            live["view_fps"] = (len(vr) - 1) / (vr[-1] - vr[0]) if len(vr) > 2 and vr[-1] > vr[0] else float("nan")
            new = fr is not None and fidx != last_fidx
            if new:
                src_fps = a.fps if a.source == "camera" else (live["cam_fps"] if live["cam_fps"] == live["cam_fps"] else a.fps)
                s.update(cam_fps=live["cam_fps"], view_fps=live["view_fps"], cpu=live["cpu"], temp=live["temp"],
                         src_fps=src_fps, src_desc=st["source_desc"], stopped_reason=None,
                         margin=(a.hop / src_fps * 1000) / s["ms_med"] if s["ms_med"] == s["ms_med"] and s["ms_med"] > 0 else float("nan"))
                canvas = painter.draw(fr, s)
                last_fidx = fidx; view_ts.append(now)
                if win:
                    cv2.imshow(win, canvas)
                if writer is not None:
                    if now - rec_next > 1.0:
                        rec_next = now
                    while rec_next <= now:
                        writer.write(canvas); rec_next += 1 / 15
                if web is not None and web.clients > 0 and now - last_jpg > 0.1:
                    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        web.push(jpg.tobytes()); last_jpg = now
                if s["alert"] is not None and s["alert"]["n"] != last_alert_n:
                    last_alert_n = s["alert"]["n"]
                    os.makedirs(os.path.join(out, "alerts"), exist_ok=True)
                    cv2.imwrite(os.path.join(out, "alerts", f"alert_{last_alert_n:02d}_screen.jpg"), canvas)
            if win:
                key = cv2.waitKey(1 if new else 5) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "q dabaya"; break
                if key == ord("s") and canvas is not None:
                    p = os.path.join(out, f"screenshot_{int(now - st['t0']):05d}s.png"); cv2.imwrite(p, canvas); print("screenshot:", p)
                if key == ord("f"):
                    full = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
                try:
                    if last_fidx >= 0 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) == 0:
                        stop_reason = "window band ki"; break
                except cv2.error:
                    stop_reason = "window band ki"; break
            elif not new:
                time.sleep(0.005)
            if now >= next_sec:
                next_sec += 1.0
                if now - next_sec > 2:
                    next_sec = now + 1.0
                live["cpu"], live["temp"] = cpu.sample(), read_temp()
                fps_rows.append((round(now - st["t0"], 1), live["cam_fps"], live["view_fps"], live["cpu"], live["temp"], ram_mb(), read_freq()))
            if now >= next_print:
                next_print = now + 5
                ai = s["ai"]
                print(f"[{now - st['t0']:6.0f}s] camera {fnum(live['cam_fps'], '{:.1f}')} FPS | AI {fnum(s['ms_med'])} ms/clip | "
                      f"score {('-' if not ai or ai['gated'] else format(ai['score'], '.3f'))} | clips {st['clips']} "
                      f"(gated {st['gated']}) | alerts {st['alerts']} | CPU {fnum(live['cpu'])}% | {fnum(live['temp'], '{:.1f}')} C")
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    print(f"\nruk raha hun ({stop_reason})...")
    stop.set()
    dur = time.time() - st["t0"]
    th[0].join(6)
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    put_latest(q, None, st, count=False)
    th[1].join(10); th[2].join(5)
    gpio.close()
    if writer is not None:
        writer.release()
    if web is not None:
        web.close()

    # ---- report
    ts_all = list(st["cam_ts"])
    per_sec = [r[1] for r in fps_rows[3:] if r[1] == r[1]]
    ms, e2e, scores = st["ms"], st["e2e"], st["scores"]
    hop_ms = a.hop / a.fps * 1000
    span = (st.get("last_ts") or 0) - (st.get("first_ts") or 0)
    cam_fps = (st["frames"] - 1) / span if span > 0 else float("nan")
    cpu_v = [r[3] for r in fps_rows if r[3] == r[3]]; tmp_v = [r[4] for r in fps_rows if r[4] == r[4]]; ram_v = [r[5] for r in fps_rows if r[5] == r[5]]
    R = dict(version=VERSION, when=stamp.strftime("%Y-%m-%d %H:%M"), device=device_name(), os=os_name(),
             model=model.pretty, model_dir=d, model_mb=round(model.size_mb, 2), onnxruntime=ort.__version__, opencv=cv2.__version__,
             threads=a.threads, source=st["source_desc"], thr=thr, thr_rule=OP_NAMES.get(op, op), hop_frames=a.hop, hop_ms=hop_ms,
             clip_sec=model.SPAN / a.fps, motion_gate=a.motion, duration_s=round(dur, 1), stop_reason=stop_reason,
             frames=st["frames"], camera_fps=round(cam_fps, 2), camera_fps_min=round(min(per_sec), 1) if per_sec else float("nan"),
             screen_fps=round(med([r[2] for r in fps_rows if r[2] == r[2]]), 1) if fps_rows else float("nan"),
             clips=st["clips"], clips_inferred=len(ms), clips_gated=st["gated"], clips_dropped=st["dropped"],
             ai_ms_median=round(med(ms), 1), ai_ms_p95=round(p95(ms), 1), ai_ms_max=round(max(ms), 1) if ms else float("nan"),
             e2e_ms_median=round(med(e2e), 1), e2e_ms_p95=round(p95(e2e), 1),
             margin=round(hop_ms / med(ms), 2) if ms else float("nan"),
             cpu_mean=round(float(np.mean(cpu_v)), 1) if cpu_v else float("nan"), cpu_max=round(max(cpu_v), 1) if cpu_v else float("nan"),
             temp_start=temp_start, temp_max=round(max(tmp_v), 1) if tmp_v else float("nan"), throttled_start=thr_start,
             throttled_end=throttled(), ram_max_mb=round(max(ram_v), 0) if ram_v else float("nan"),
             alerts=st["alerts"], alert_list=st["alert_list"], score_max=round(st["smax"], 4),
             clips_above_thr=int(sum(x >= thr for x in scores)), camera_error=st["camera_error"])
    V = []
    if a.source == "camera":
        V.append(("PASS" if cam_fps >= 0.9 * a.fps else "WARN", f"Camera ran at {cam_fps:.1f} FPS (target {a.fps:.0f}); every frame goes into the AI buffer"))
    else:
        V.append(("INFO", f"Video played in real time at {cam_fps:.1f} FPS (paced like a camera)"))
    if ms:
        V.append(("PASS" if med(ms) < hop_ms else "WARN", f"AI keeps up: {med(ms):.0f} ms per clip vs {hop_ms:.0f} ms between decisions ({hop_ms / med(ms):.1f}x margin)"))
    V.append(("PASS" if st["dropped"] == 0 else "WARN", f"{st['dropped']} clips skipped because the AI was busy"))
    th_end = R["throttled_end"]
    V.append(("PASS" if th_end in ("0x0", "n/a") and (not tmp_v or max(tmp_v) < 80) else "WARN",
              f"Thermal: max {fnum(R['temp_max'], '{:.1f}')} C, throttled flag {th_end}"))
    hours = dur / 3600
    if st["alerts"] == 0 and a.source == "camera" and hours > 0:
        V.append(("INFO", f"0 alerts in {dur / 60:.1f} min of live camera (95% upper bound {3 / hours:.0f} false alarms per hour)"))
    else:
        V.append(("INFO", f"{st['alerts']} alert(s); max score {st['smax']:.3f}"))
    R["verdicts"] = V
    R["note"] = ("Live demo numbers include the screen drawing" + (" and recording" if writer is not None else "") +
                 "; the paper's Pi numbers come from the headless runs (step5/step6). The type hint is a hint "
                 "(33% top-1 over 13 classes, 59% over 4 groups on the UCF-Crime test set), not a verdict.")
    with open(os.path.join(out, "fps.csv"), "w") as f:
        f.write("t_sec,camera_fps,screen_fps,cpu_pct,temp_c,ram_mb,cpu_mhz\n")
        for r in fps_rows:
            f.write(",".join(fnum(x, "{:.2f}", "") for x in r) + "\n")
    hist = st["hist_all"]
    simg = summary_image(R)
    cv2.imwrite(os.path.join(out, "summary.png"), simg)
    txt = write_report(out, R, fps_rows, hist, model, thr)
    allp = os.path.join(a.out, "all_runs.csv")
    new_file = not os.path.exists(allp)
    with open(allp, "a") as f:
        if new_file:
            f.write("when,source,duration_s,camera_fps,ai_ms_median,ai_ms_p95,e2e_ms_median,clips,gated,dropped,alerts,max_score,temp_max,throttled,folder\n")
        f.write(f"{R['when']},{R['source'].replace(',', ' ')},{R['duration_s']},{R['camera_fps']},{R['ai_ms_median']},{R['ai_ms_p95']},"
                f"{R['e2e_ms_median']},{R['clips']},{R['clips_gated']},{R['clips_dropped']},{R['alerts']},{R['score_max']},"
                f"{R['temp_max']},{R['throttled_end']},{out}\n")
    print("\n" + txt)
    print(f"\nREPORT: {os.path.join(out, 'report.html')}  (browser mein kholo)")
    if win and stop_reason != "window band ki":
        try:
            cv2.imshow(win, simg)
            print("window par summary hai: koi bhi key dabao band karne ke liye")
            cv2.waitKey(0 if not a.max_sec else 8000)
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
