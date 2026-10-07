# PiLENS System Engineering Notes (v2)

Oct 7, 2026 · Huzaifa Asad

> Ye notes v2 ka "source of truth" hain. Code mein in ka mapping:
> constants `pilens_v2/spec.py`, commands [`v2_quickstart.md`](v2_quickstart.md),
> paper tables [`v2_paper_tables.md`](v2_paper_tables.md).

## 1. Maqsad aur paper ka claim

PiLENS ek low-cost, night-vision, edge-deployed suspicious activity detection system hai jo Raspberry Pi 5 par realtime chalta hai. Paper ka ek-line claim: UCF-Crime ki 14 classes par X3D-S/MoViNet jaise halke video models ko two-stage cascade (pehle binary anomaly, phir alert par 14-class) mein chala kar Pi 5 par accuracy ke saath latency, FPS aur temperature ka measured benchmark dena.

- Target journal: IEEE Access (manuscript: abstract, introduction, methodology block diagrams, literature review, results).
- Asli contribution: edge benchmarking, night-vision data aur cascade design. Sirf accuracy ke liye bahut models pehle se hain.
- v1 (3rd semester ka kaam): ResNet18 + LSTM, 2 classes (Normal/Suspicious), Program_3. Ye `v1-baseline` tag karke freeze hai aur paper mein baseline banega. Uska GUI, email, GPIO, Flask stream aur Tailscale ka code naye system mein reuse hota hai.
- v1 ki kamiyan: koi video-level split nahi tha, sirf loss report hota tha, tracking nahi thi, aur temporal verification nahi thi. Isliye v1 ke numbers paper mein cite nahi karne.
- Scope jo abhi nahi karna: VideoMamba aur cloud models, face recognition (Program_2), mobile app, GUI polish. Ye paper ke baad.

## 2. Locked decisions

1. Ek hi paper, v1 vs v2 ablation ke saath. Model experiments alag paper nahi, usi paper ka Methodology aur Results hissa hain (sir se confirm karna).
2. Do-stage cascade: stage 1 binary (Normal/Suspicious) har clip par, stage 2 (14-class) sirf alert par. Dono ek hi backbone share karte hain.
3. Pehla model X3D-S, uske baad MoViNet-A0/A1. Wahi split, wahi features wali setup, wahi heads, taake comparison fair rahe.
4. Real untrimmed videos use hongi, event-trimmed clips nahi. Spreadsheet sirf ye batati hai ke event kahan hai.
5. Official UCF-Crime splits use honge (14-class 4-fold, anomaly-detection split), taake literature se compare ho sake.
6. Video-level split, kabhi window-level nahi. Ek video ke saare windows sirf ek hi split mein. Test set par sirf pehle se fix settings chalengi, tuning sirf validation split par.
7. Class imbalance: WeightedRandomSampler DataLoader level par, taake dominant class overfit na karwaye.
8. Pi par runtime: ONNX Runtime ya NCNN. TensorRT Pi par nahi chalta (NVIDIA GPU ke liye hai). Pehle FP32 ka number, phir zaroorat ho to INT8.
9. Sirf measured numbers paper mein jayenge. Pi par jo number measure nahi hua, wo andaaza hai aur alag likha jayega.
10. Training Kaggle GPU par (2x T4), Pi ke liye alag export aur benchmark ka step.

## 3. Hardware aur network

| Cheez | Detail |
|---|---|
| Board | Raspberry Pi 5, 8 GB RAM |
| Camera | Infrared night-vision (NoIR) camera module |
| IR lighting | Even IR illuminator, taake grayscale frames barabar roshan hon |
| Camera settings | Fixed exposure aur gain (auto-exposure se flicker aata hai) |
| Outputs | GPIO LED aur buzzer, alert clip save, Gmail se email |
| GUI aur stream | PyQt6 GUI, Flask MJPEG stream (port 8000) |
| Remote monitoring | Tailscale (private network, remote access aur controls) |
| Training machine | Kaggle, 2x T4, 31 GB RAM |

Network ke qawaid:

- Remote monitoring ke liye sirf Tailscale, Flask port ko public internet par expose nahi karna.
- Email credentials code mein hardcode nahi, config ya environment variable mein.
- Pi ka CPU temperature har benchmark ke saath log hoga, kyunki thermal throttling FPS gira sakti hai. Active cooling (fan ya heatsink) ka zikr paper mein hoga.

## 4. Dataset aur split

Dataset public Kaggle `minmints/ufc-crime-full-dataset` hai: 1950 videos (50 Normal duplicate, yani 1900 unique), sab 320x240 aur 30 FPS, sab readable.

| Cheez | Detail |
|---|---|
| Classes | UCF-Crime ki 14 classes (anomaly classes + Normal ke saath ka setup) |
| 14-class split | Official 4-fold, har class mein 38 train / 12 test videos |
| Anomaly-detection split | 1610 train / 290 test (root-level `Anomaly_Train.txt` real list hai) |
| Gotcha | `Anomaly_Detection_splits/Anomaly_Train.txt` khali hai, isliye root wali use karni hai |
| Annotations | Manual xlsx: 950 anomaly videos, 947 usable, event start/end, Day/Night, usable flags |
| Night share | Anomaly videos lagbhag aadhi night ki (Fighting 22/50, Robbery 68/150, Shoplifting 10/50) |

Dhyan rakhne wali baatein:

- Duplicate Normal videos: `Normal_Videos_for_Event_Recognition` aur `Testing_Normal_Videos_Anomaly` ki 50 files same hain. Dedupe zaroori hai, warna ek hi video train aur test dono mein chali jayegi.
- Normals par Day/Night label nahi: agar normals zyadatar din ke hue to model "andhera = anomaly" seekh lega. Normals ka Day/Night automatic nikalna hai.
- Weak labels: training videos mein sirf video-level label hai, event chhota hissa hota hai. Isliye MIL (multiple instance learning) use hota hai.
- Class imbalance: WeightedRandomSampler DataLoader level par, aur macro-F1 report karna sirf accuracy nahi.
- Validation: train se 15% videos video-level par alag.
- Normals bahut bade hain: total 65 GB, kuch videos ghanton lambi (ek 5.8 ghante ki, 5 GB). Zaroorat ho to subset use karo.
- Split files git mein commit karni hain.

## 5. Model aur features

Backbone: X3D-S, Kinetics-400 pretrained (K600 nahi, official PyTorch weights K400 hain). MoViNet-A0/A1 dusre number par.

| Parameter | Value |
|---|---|
| Clip | 13 frames, stride 6 (30 FPS par lagbhag 2.4 sec ki video) |
| Crop | 160x160 (320x240 frame ko 182 par resize, phir center crop) |
| Segments | Har untrimmed video ke 32 temporal segments, har segment ke beech ek 2.4 sec ka clip |
| Feature | 2048-d pre-logit vector, 1900 unique videos ke liye cache (`feats_x3ds`, `video_meta.csv`) |
| Stage 1 head | Binary MIL (MLP ya TConv), loss: top-3 ranking hinge + smoothness + sparsity |
| Stage 2 head | 14-class (linear ya mil_topk), features par chhota layer |
| Pi runtime | ONNX Runtime ya NCNN, FP32 pehle, INT8 baad mein |

Dhyan rakhne wali baatein:

- 320x240 ko resize karke 160x160 center crop karne se width ka lagbhag 1/3 hissa kat jata hai, to kinare ke events miss ho sakte hain. Isko ablation mein test karna hai.
- Night frames near-grayscale hote hain, aur Kinetics RGB par train hai (domain gap). Grayscale ko 3 channel mein replicate karo, training aur inference mein ek jaisi preprocessing, aur RandomGrayscale ke saath noise, brightness aur blur augmentation.
- End-to-end fine-tune (trimmed clips) overfit hua (train ~92-97%, val 25-34%). Isliye abhi frozen features + head ka raasta hai.
- Heads (linear, mil_topk, tconv_mil) 14-class par noise ke andar barabar nikle, to sabse sasta head (linear) chuno.
- X3D-S ka cost lagbhag 2 GFLOPs per clip hai. MoViNet ko ONNX mein export karna streaming design ki wajah se mushkil ho sakta hai.

## 6. Architecture

Backbone har clip par sirf ek baar chalta hai. Binary head har clip ka score deta hai, aur 14-class head tabhi chalta hai jab 3 mein se 2 clips suspicious hon.

- Stage 1 (hamesha on): frame buffer, motion gate, clip, X3D-S feature, binary head.
- Stage 2 (sirf alert par): pichle 3-5 clips ke cached features ka average, phir 14-class head (top-3 classes confidence ke saath).
- Confidence kam ho to "Unknown suspicious" dikhao, kyunki 14-class accuracy abhi ~33% hai.

## 7. Realtime pipeline aur latency

Realtime ka matlab har frame par model chalana nahi hai. Camera 30 FPS par frames deta rahega, model har ~0.5-1.2 sec mein ek naya clip score karega.

| Thread | Kaam | Rule |
|---|---|---|
| 1. Capture | Camera se frames lena, ring buffer mein rakhna | Kabhi block nahi hona chahiye, warna frames drop honge |
| 2. Inference | Buffer se clip, X3D-S, binary head, vote, alert par 14-class head | Ek hi backbone, ek hi baar per clip |
| 3. Alert | LED, buzzer, 5 sec clip save, email, stream | Alag thread, taake inference ruke nahi |

Latency:

- Buffer khud compute nahi leta. Delay ka asli source ye hai ke model ko ek poora clip chahiye (13 frames, stride 6, lagbhag 2.4 sec).
- Rolling buffer: sirf pehli baar 2.4 sec bharne ka intezaar, uske baad har hop par naya clip.
- Hop chhota (0.5-0.6 sec) = score zyada baar, lekin CPU load zyada.
- Clip chhota (jaise 8 frames, lagbhag 1.2 sec) = latency kam, lekin accuracy girne ka khatra, retrain karke test karna hoga.
- "3 mein se 2 clips" rule false alarms kam karta hai, lekin ek-do hop ki delay badhata hai. Ye tradeoff paper mein table banao.
- 14-class head cached 2048-d features par chalta hai, to alert ke waqt extra bojh lagbhag zero hai. Asli spike alert ke baad ke kaam (video encode, email) se aata hai, isliye wo alag thread mein.

Andaaze (Pi par measure nahi hue): X3D-S ek clip par ONNX Runtime CPU par 0.2-0.5 sec, aur event se alert tak ka reaction time 1.5-3 sec. Paper mein sirf measured numbers jayenge.

"31.34 FPS / 32 ms" target: ye number kis cheez ka hai, pehle define karo. Clip latency (ek clip ka inference time) aur stream throughput (camera frames per second jo pipeline handle kare) alag cheezein hain. Dono ko alag report karo.

## 8. Night vision, motion gate aur crop

Night vision:

- IR/NoIR frames near-grayscale hote hain, aur Kinetics-pretrained models RGB par train hain (domain gap).
- Grayscale ko 3 channel mein replicate karo. Wahi preprocessing training aur inference dono mein.
- Training mein RandomGrayscale ke saath noise, brightness aur blur augmentation.
- Apne night aur day clips banao, alag night test set rakho, phir fine-tune.
- Camera par fixed exposure/gain aur even IR lighting.
- IR LED ke paas ud'te keede motion mein bright blobs banate hain. Unhe minimum area aur persistence se filter karo.

Motion gate (sabse sasta stage):

- Frame ko chhota karo (jaise 320x180 grayscale), Gaussian blur, phir MOG2 background subtraction (`detectShadows=True`).
- Motion area kam ho to YOLO aur action model chalao hi mat. Isse Pi ka bahut compute bachta hai.
- Motion akela crop source nahi hai: parde, ped, shadows, light flicker bhi motion banate hain, aur har frame mein box jitter karta hai.

Crop strategy (paper ka ablation):

| Variant | Kya hai |
|---|---|
| A. Full frame | Abhi ka baseline, detection/tracking ki zaroorat nahi |
| B. Union box | Paas wale tracked persons ka union box, 15-20% padding |
| C. Motion ROI | Motion wale hisse ka crop |
| D. Single-person crop | v1 jaisa, interaction wali classes (Fight, Robbery) kat jati hain |

- Union box ya EMA smoothing se ROI ek clip ke andar stable rakho, warna jitter model ko noise dikhata hai.
- Training aur deployment ka input same hona chahiye, to Track B mein training data bhi offline detector + tracker se banega.

Noise kam karne ke Pi-friendly tareeqe: Gaussian ya bilateral blur, 2-3 frame temporal averaging, minimum blob area, persistence check.

Tracking: YOLO11 + ByteTrack (`yolo.track(persist=True)`), per-person ID ke alag buffers. Ye sabse mehnga hissa hai, isliye sirf motion gate ke baad chalega.

## 9. Ab tak ke results (X3D-S, Kaggle T4)

Test sets sirf pehle se fix settings ke saath chalayi gayi, aur epoch selection train ke andar ki validation split se hua.

**Task A: 14-class recognition** (official 4 folds, 38 train / 12 test videos per class, frozen features + head, untrimmed videos, 4 folds x 3 seeds, mean +- std over folds)

| Head | Accuracy (%) | Macro-F1 (%) |
|---|---|---|
| linear | 33.2 +- 0.3 | 31.8 +- 0.3 |
| mil_topk | 33.4 +- 0.4 | 32.1 +- 0.6 |
| tconv_mil | 32.5 +- 0.9 | 31.0 +- 1.1 |

Chance 7.1% hai. Teeno heads noise ke andar barabar hain. End-to-end fine-tune (fold 1, trimmed clips) overfit hua (val 27-34%, train ~92%) aur rok diya gaya, uska test nahi hua.

**Task B: weakly supervised anomaly detection** (official split: 1610 train, 290 test, val = train ka 15%, binary MIL, 5 seeds, frame-level GT test video ke asli frame count se)

| Head | Video AUC | Frame AUC | Frame AP |
|---|---|---|---|
| MLP | 0.912 +- 0.005 | 0.806 +- 0.010 | 0.205 |
| TConv | 0.903 +- 0.008 | 0.812 +- 0.005 | 0.207 |

Gaussian smoothing (sigma = 1 segment, pehle se fix) se frame AUC lagbhag +0.01 (0.815 / 0.818). Ise alag row mein report karo, primary number unsmoothed hai.

Isse kya samajh aata hai:

- Binary stage (video AUC ~0.91) mazboot hai, 14-class stage (~33%) kamzor. Isliye cascade ka design binary par tika hai.
- 32 segments par 99.6% videos mein kam se kam ek clip event ke andar aata hai (median 15/32), to sparse sampling bottleneck nahi hai.
- Paper mein macro-F1, per-class precision/recall/F1 aur confusion matrix bhi dena hai. Kamzor classes confusion matrix mein dikhengi, unhe discussion mein explain karo.

Purane notebook ke numbers cite nahi karne: AUC 1.0000 synthetic random features se aaya tha, aur frame-level 0.7472 galat 512-frame ground-truth alignment se tha, saath mein 7 model variants test set par compare hue the.

## 10. Pi deployment aur benchmark plan

Export aur runtime:

1. X3D-S backbone + head ko ONNX mein export karo, phir parity check (PyTorch vs ONNX output).
2. Pi 5 par ONNX Runtime (CPU) chalao. NCNN alternative hai.
3. TensorRT Pi par nahi chalta (NVIDIA GPU ke liye). Use na karo.
4. Pehle FP32 ka number lo. Zaroorat pade to FP16 ya INT8 quantization (calibration data day aur night dono se), aur accuracy drop report karo.
5. MoViNet ONNX export mushkil ho sakta hai (streaming design), isliye X3D-S pehle.

Pi par kya measure karna hai (har benchmark ke liye):

| Metric | Kaise |
|---|---|
| Clip latency | Ek clip ka inference time (median aur p95) |
| Stream throughput | Camera frames per second jo pipeline handle kare, end to end |
| Event-to-alert time | Event shuru hone se alert tak ka waqt |
| CPU temperature | Har run ke saath log, throttling dekhne ke liye |
| CPU aur RAM use | Cascade ke saath aur bina |
| Power | Agar meter ho to |
| Accuracy drop | FP32 vs INT8 |

Benchmark rules:

- Pehle 10-20 warm-up runs, phir kam se kam 100 runs, aur median/p95/mean+-std report karo.
- Camera capture, motion gate aur alert thread chalte hue measure karo, sirf model akela nahi.
- Fan ya heatsink ke saath aur bina, lambe run (jaise 30 min) mein temperature aur FPS ka graph.
- Har stage ka alag time: capture, preprocessing, backbone, head, alert.
- Pi OS, Python, ONNX Runtime ke versions aur threads ki count paper mein likho.

## 11. Alert logic aur evaluation protocol

Alert logic:

- Binary head har clip ka suspicious score deta hai.
- Threshold validation split par fix hoga, test par kabhi tune nahi.
- Temporal verification: "3 mein se 2 clips suspicious" (ya majority voting over sliding windows) tabhi alert, sirf ek clip par nahi.
- Alert par stage 2 chalta hai: pichle 3-5 clips ke features ka average, phir 14-class head. Top-3 classes confidence ke saath, confidence kam ho to "Unknown suspicious".
- Alert ke baad cooldown rakho, taake ek hi event par baar baar email/buzzer na ho.
- Alert actions: GPIO LED aur buzzer, pichle ~5 sec ka clip `Intruders/` mein save, Gmail se video ke saath email, Flask stream aur Tailscale se remote dekhna.

Evaluation protocol (Manaheel ya kisi aur ke saath common contract):

| Cheez | Fix kiya hua rule |
|---|---|
| Split | Video-level, fixed file list, git mein commit |
| Metrics (14-class) | Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix |
| Metrics (binary) | Video AUC, frame AUC, frame AP, aur chune hue threshold par precision/recall |
| Model spec | Input 13 frames, stride 6, 160x160, 3 channel; output binary score aur 14 class logits |
| Export format | ONNX (parity check ke saath) |
| Seeds | Kam se kam 3-5 seeds, mean +- std |
| Tuning | Sirf validation split par |
| Test set | Sirf aakhri baar, pehle se fix settings ke saath |
| Latency budget | Pi par measure hone ke baad fix hoga (clip latency aur throughput alag) |

Paper ki ablation tables:

1. v1 (ResNet18-LSTM) vs v2 (X3D-S) vs MoViNet.
2. Crop: full frame, union box, motion ROI, single-person.
3. Preprocessing: RGB vs grayscale-3ch, night augmentation on/off.
4. Cascade: sirf binary vs binary + 14-class, aur "3 mein se 2" rule on/off.
5. FP32 vs FP16 vs INT8: accuracy, latency, temperature.
6. Day vs night test set alag-alag.

## 12. Risks, ethics aur open questions

Risks:

- 14-class accuracy kam (~33%): claim ko realistic rakho, binary detection ko headline banao, 14-class ko "assistive" likho.
- Domain gap (UCF-Crime vs Pi ka NoIR camera): UCF-Crime ke videos 320x240 CCTV hain, Pi ka camera alag hai. Apne Pi footage par external check zaroori hai, warna deployment claim kamzor rahega.
- Night shortcut: normals mein night kam hua to model andhera = anomaly seekh sakta hai.
- Thermal throttling: lambe run mein FPS gir sakti hai.
- MoViNet export: ONNX mein mushkil ho sakta hai.
- Tuning ka leakage: koi bhi tuning test set par nahi.
- Alert fatigue: false alarms zyada hue to system practical nahi, isliye FAR (false alarms per hour) bhi report karo.

Ethics aur data:

- Surveillance data mein logon ke chehre hote hain. UCF-Crime ka license aur terms of use check karo.
- v1 ka dataset WhatsApp videos se bana tha. Agar wo kisi aur ke videos hain to publication ke liye permission/license ka masla aa sakta hai.
- Naye clips (apni team) ke liye written consent aur department ki ethics requirement.
- Email credentials aur Tailscale access ko secure rakho.

Sir se poochne wale sawal:

1. Ek paper hoga ya do (system + model)?
2. Target journal/conference aur template (IEEE Access template)?
3. Authorship order?
4. Dataset recording ke liye consent/ethics ki kya requirement hai?

Open questions (apne faisle):

- "31.34 FPS / ~32 ms" ka matlab: clip latency ya stream throughput?
- Final class list: sab 14 classes ya shuru mein 3-4 focus classes (Normal, Fighting, Shoplifting, Robbery)?
- Crop strategy ka default (union box + padding)?
- Clip length: 13 frames (2.4 sec) rakhna ya chhota?
- Kya Track B (YOLO + ByteTrack crops) paper ke is version mein hoga ya baad mein?

## 13. Roadmap: kaam ka order

Ek time par ek hi step. Har step ke saath "done jab" likha hai. Har step ki command [`v2_quickstart.md`](v2_quickstart.md) mein hai.

- [ ] Step 0: repo setup. Purani repo ko `v1-baseline` tag karo, naya branch `v2-pipeline`, split files aur ye notes commit karo. Done jab: GitHub par tag dikhe.
- [ ] Step 1: sir se baat. Upar ke 4 sawal. Done jab: jawab notes mein likhe hon.
- [ ] Step 2: B1 features. Kaggle par feature extraction (LIMIT = 8 quick test, phir poora), feature files valid: N/N, backup aur Output se "New Dataset" save. Ye pehle chal chuka hai (1900 videos).
- [ ] Step 3: B2 aur B3 dobara reproduce. 14-class aur binary MIL, fixed seeds, validation par hi tuning. Done jab: experiment log ke numbers dobara mil jayen.
- [ ] Step 4: ONNX export. X3D-S backbone + head, parity check.
- [ ] Step 5: Pi par FP32 benchmark. Clip latency, throughput, temperature. Ye pehla measured Pi number hai.
- [ ] Step 6: threaded pipeline. Capture, inference, alert threads, rolling buffer, motion gate, "3 mein se 2" rule.
- [ ] Step 7: night data. Apne day/night clips (consent ke saath), alag night test set, grayscale augmentation, fine-tune.
- [ ] Step 8: MoViNet. Wahi setup, X3D-S ke muqable table.
- [ ] Step 9: quantization. FP16/INT8, accuracy drop aur latency.
- [ ] Step 10: Track B (agar time ho). YOLO11 + ByteTrack, crop ablation.
- [ ] Step 11: paper likhna. Abstract, introduction, related work, system design (diagram), dataset, experimental setup, results, discussion/limitations, conclusion.

Abhi nahi karna: VideoMamba, cloud models, face recognition, mobile app, GUI polish.
