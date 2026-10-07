# PiLENS v2: step-by-step commands

Har roadmap step ([`v2_engineering_notes.md`](v2_engineering_notes.md), section 13) ke liye
copy-paste commands. Saara v2 code `pilens_v2/` mein hai, v1 (`Programs/`) ko haath nahi lagaya.

```
pilens_v2/
  spec.py                 13 frames, stride 6, 160 crop, 32 segments, classes (ek hi jagah)
  preprocess.py           resize 182 -> center crop 160, gray->3ch, normalise (Kaggle + Pi same)
  data/splits.py          official splits, dedupe, 15% video-level val, leak check     (Step 0/2)
  data/annotations.py     frame-level GT asli frame count se
  data/daynight.py        Normals ka automatic Day/Night + night-shortcut check
  train/extract_features  X3D-S features, 32 clips/video, "valid: N/N"                 (Step 2)
  train/train_cls14.py    Task A: 14-class, 4 folds x seeds, val par epoch selection    (Step 3)
  train/train_mil.py      Task B: binary MIL, 5 seeds, val par threshold                (Step 3)
  export/export_onnx.py   backbone + 2 heads -> ONNX, parity check                      (Step 4)
  bench/benchmark.py      Pi latency (median/p95), temperature, versions                (Step 5)
  runtime/run.py          threaded live pipeline: capture / motion / inference / alert  (Step 6)
  runtime/replay.py       test videos offline -> per-hop scores
  eval/alert_eval.py      event-to-alert, FA/hour, "2 of 3" on/off, motion gate on/off
  export/quantize.py      INT8 (day + night calibration)                                (Step 9)
```

Tests: `python -m pytest tests -q` (synthetic data par poora flow: training -> ONNX -> replay -> live).

## Sabse aasaan raasta (3 kaam)

| Kahan | Kya karna hai | File |
|---|---|---|
| Kaggle | Notebook upload, dataset add, GPU T4 x2, **Run All** (Steps 2, 3, 4). End mein `pilens_outputs.zip` | [`notebooks/PiLENS_v2_Kaggle.ipynb`](../notebooks/PiLENS_v2_Kaggle.ipynb) |
| Pi | `exports/` copy, phir `bash scripts/pi_setup.sh` aur `bash scripts/pi_benchmark_all.sh` (Steps 5, 6) | [`scripts/`](../scripts) |
| Laptop | Zip ke `splits/`, `runs/*/summary.json` aur Pi ke `bench_results/`, `logs/` repo mein daal kar `python -m pilens_v2.eval.make_tables` -> `docs/v2_results.md` | [`pilens_v2/eval/make_tables.py`](../pilens_v2/eval/make_tables.py) |

Phir paper: [`docs/paper/pilens_v2.tex`](paper/pilens_v2.tex) mein har laal `[TBD]` ko `docs/v2_results.md` se bharo.
Sir ka message aur consent form: [`v2_supervisor_and_ethics.md`](v2_supervisor_and_ethics.md).
Pi par hamesha chalane ke liye (boot par start): [`scripts/pilens.service`](../scripts/pilens.service).

---

## Step 0: repo setup

v2 ab `main` par merge ho chuka hai, isliye alag `v2-pipeline` branch ki zaroorat nahi. Sirf v1 ka tag
lagana hai, **merge se pehle wale commit `386de78`** par (warna tag v2 ko point karega):

```bash
git fetch origin
git tag -a v1-baseline -m "v1: ResNet18+LSTM, 2 classes (baseline)" 386de78
git push origin v1-baseline
```

Bina terminal ke: GitHub repo -> *Releases* -> *Draft a new release* -> *Choose a tag* mein `v1-baseline`
likho -> *Target* mein *Recent commits* se `386de78 Add files via upload` chuno -> *Publish release*.

## Step 2: features (Kaggle, 2x T4)

Kaggle notebook: Add data `minmints/ufc-crime-full-dataset`, GPU T4 x2, Internet ON.

```bash
!git clone -b v2-pipeline https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi pilens
%cd pilens
!pip install -q pytorchvideo
ROOT=/kaggle/input/ufc-crime-full-dataset

# 1) splits (+ frame count / fps). Output check karo: unique_videos 1900, duplicates_removed 50,
#    anomaly train+val 1610, test 290, cls14 har fold test 168 (12 x 14)
!python -m pilens_v2.data.splits --root $ROOT --out splits --probe

# 2) quick test (LIMIT = 8)
!python -m pilens_v2.train.extract_features --root $ROOT --index splits/video_index.csv --out /kaggle/working/feats_x3ds --limit 8

# 3) poora, dono GPU par ek saath (resume hota hai, bani hui files skip)
!python -m pilens_v2.train.extract_features --root $ROOT --out /kaggle/working/feats_x3ds --shard 0/2 --device cuda:0 > log0.txt 2>&1 &
!python -m pilens_v2.train.extract_features --root $ROOT --out /kaggle/working/feats_x3ds --shard 1/2 --device cuda:1 > log1.txt 2>&1 &
# dono khatam hone ke baad:
!python -m pilens_v2.train.extract_features --root $ROOT --out /kaggle/working/feats_x3ds --validate-only
# "valid: 1900/1900" aaye tab Output -> New Dataset (feats-x3ds) banao
```

Features ka format: `feats_x3ds/<video_id>.npy` shape (32, 2048) float16, aur `video_meta.csv`
(`video_id,label,num_frames,rel_path`). Agar purana cache kisi aur format mein hai to ya dobara
extract karo, ya sirf `pilens_v2/train/common.py` ka `FeatureStore.get()` badlo.

Lambi Normal videos (5.8 ghante) bhi chalengi: har clip ke liye seek hota hai, poori video decode nahi hoti.

### Manual annotations (xlsx -> CSV)

Code ko ek simple CSV chahiye: `video_id,start_frame,end_frame,day_night,usable` (ek event = ek row).
Apne xlsx ke column naam ke hisaab se:

```python
import pandas as pd
x = pd.read_excel("annotations.xlsx")
x = x.rename(columns={"Video": "video_id", "Start": "start_frame", "End": "end_frame",
                      "Day/Night": "day_night", "Usable": "usable"})   # <- apne column naam
x[["video_id", "start_frame", "end_frame", "day_night", "usable"]].to_csv("annotations.csv", index=False)
```

Day/Night (Normals ke liye automatic, thresholds aapke manual labels par calibrate hote hain):

```bash
!python -m pilens_v2.data.daynight --root $ROOT --ann annotations.csv --out splits/day_night.csv
```

Output mein har split ke Normal vs Anomaly ka night % aata hai. Agar Normals mein night bahut kam hai
to night-shortcut ka risk hai: paper mein likho aur Day vs Night test alag report karo.

## Step 3: B2 + B3 reproduce

```bash
FEATS=/kaggle/input/feats-x3ds
# Task A: 14-class (teeno heads, 4 folds x 3 seeds)
!python -m pilens_v2.train.train_cls14 --feats $FEATS --splits splits --head linear    --out runs/cls14_linear
!python -m pilens_v2.train.train_cls14 --feats $FEATS --splits splits --head mil_topk  --out runs/cls14_mil_topk
!python -m pilens_v2.train.train_cls14 --feats $FEATS --splits splits --head tconv_mil --out runs/cls14_tconv_mil

# Task B: binary MIL (5 seeds)
ANN=$ROOT/Temporal_Anomaly_Annotation_for_Testing_Videos.txt   # path dataset mein check karo
!python -m pilens_v2.train.train_mil --feats $FEATS --splits splits --test-ann $ANN --head mlp   --out runs/mil_mlp
!python -m pilens_v2.train.train_mil --feats $FEATS --splits splits --test-ann $ANN --head tconv --out runs/mil_tconv
```

Har run `summary.json` likhta hai (mean +- std, per-class P/R/F1, confusion matrix, smoothed row alag).
Hyperparameters CLI flags hain (`--lr --epochs --iters --dropout ...`). Agar purane log ke numbers
match na hon to wahi settings flags se do; **tuning sirf validation par**, test ko dekh kar kuch na badlo.
`runs/*/summary.json` commit karo (`.pt` files git mein nahi jaati).

## Step 4: ONNX export + parity

```bash
!python -m pilens_v2.export.export_onnx --out exports \
    --binary-ckpt runs/mil_mlp/seed0.pt --cls-ckpt runs/cls14_linear/fold1_seed0.pt \
    --parity-video $ROOT/<koi bhi test video>.mp4
```

Har parity line `OK` honi chahiye (warna script fail hoti hai). `exports/` folder (3 `.onnx` +
`pilens_v2.json`, jisme validation wala threshold hai) Pi par copy karo.

## Step 5: Pi par FP32 benchmark

Pi setup (Raspberry Pi OS Bookworm 64-bit):

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-flask
python3 -m venv --system-site-packages ~/pilens-venv && source ~/pilens-venv/bin/activate
pip install -r requirements-v2-pi.txt
cp .env.example .env   # email + STREAM_HOST (Tailscale IP) + camera exposure/gain bharo
```

```bash
# 20 warm-up + 200 runs, har stage alag
python -m pilens_v2.bench.benchmark --exports exports --threads 4 --tag fp32_fan
# threads ka asar
for t in 1 2 3 4; do python -m pilens_v2.bench.benchmark --exports exports --threads $t --tag fp32_t$t; done
# 30 min thermal run, fan ke saath aur bina
python -m pilens_v2.bench.benchmark --exports exports --duration 1800 --tag fp32_fan_30min
python -m pilens_v2.bench.benchmark --exports exports --duration 1800 --tag fp32_nofan_30min
```

`bench_results/*.json` mein median/p95/mean+-std, temperature (start/max), clock, throttle flags,
Pi OS / Python / ONNX Runtime versions aur thread count sab hai. `*_timeline.csv` se graph banao.
Ye files commit karo: yahi paper ke measured numbers hain.

## Step 6: threaded pipeline

```bash
python -m pilens_v2.runtime.run --exports exports --duration 1800 --log-dir logs/live_fan
# motion gate ablation
python -m pilens_v2.runtime.run --exports exports --no-motion --duration 1800 --log-dir logs/live_nogate
# file se (apni fps par paced) - end-to-end reaction time ke liye
python -m pilens_v2.runtime.run --exports exports --source test.mp4 --no-email --log-dir logs/file_test
```

`logs/*/summary.json` mein **dono numbers alag**: `capture_fps` (stream throughput) aur
`clip_latency` (ek clip ka inference time). "31.34 FPS / 32 ms" wala sawal isi se clear hoga.
`hops.csv` mein har hop ka preprocess / backbone / head ms, score aur motion; `alerts.csv` mein
alert ka waqt, top-3 classes aur stage-2 ka time.

Stream: `.env` mein `STREAM_HOST` ko Pi ka Tailscale IP (100.x.y.z) do, phir Tailscale wale
device se `http://100.x.y.z:8000/video`. Default `127.0.0.1` hai, port kabhi public na karo.

Alert-level evaluation (test videos, k-of-n aur motion gate ablation):

```bash
python -m pilens_v2.runtime.replay --exports exports --root $ROOT --ids splits/anomaly_test.txt --out replay/test
python -m pilens_v2.eval.alert_eval --replay replay/test --ann $ANN --threshold <pilens_v2.json wala threshold> --out replay/alert_eval.json
```

Ye "1 of 1 / 2 of 3 / 3 of 5" ke liye detection rate, event-to-alert (median, p95) aur false alarms
per hour deta hai. Event-to-alert video-time mein hai; wall-clock number ke liye Pi ki median clip
latency add karo (ya `--source` file run ke `alerts.csv` se seedha lo).

## Step 9: INT8

```bash
# calib_videos.txt: ~20 day + ~20 night video paths
python -m pilens_v2.export.quantize --model exports/backbone.onnx --calib-list calib_videos.txt --out exports/backbone_int8.onnx
python -m pilens_v2.bench.benchmark --exports exports --backbone backbone_int8.onnx --tag int8
```

Accuracy drop ke liye INT8 backbone se features dobara nikal kar (ya `replay` INT8 ke saath chala kar)
wahi evaluation chalao.

## Night / crop ablations

- `--gray3` flag (extract_features, export_onnx, benchmark) = grayscale -> 3 channel preprocessing.
  Export mein ye `pilens_v2.json` mein likha jata hai aur Pi runtime khud wahi use karta hai.
- `--crop-mode squash` = poora frame 160x160 (kinare kat-te nahi) vs default `center`.
- Motion ROI ke liye `MotionGate.box` (union box) runtime mein mojood hai; Track B (YOLO + ByteTrack)
  abhi implement nahi kiya (roadmap Step 10).

## Kya abhi baaki hai (code mein nahi)

- Step 1 (sir se sawal), Step 7 (apna day/night data + consent), Step 8 (MoViNet), Step 10 (Track B), Step 11 (paper).
- Night augmentation ke saath fine-tune (Step 7): abhi frozen features ka raasta hai, augmentation
  end-to-end training ke waqt add karni hogi.
- NCNN runtime: abhi sirf ONNX Runtime.
