# GPU benchmark, Kaggle Tesla T4 (2026-10-16, gpu_bench v1.0 run with onnxruntime-gpu 1.23.2, CUDA EP, fp32)
Same ONNX files as the Pi. Synthetic input. Batch 1 unless stated. "with copy" = numpy in/out (CPU<->GPU copy included).

**Note (2026-10-19 audit):** the CLIP rows of this run were removed. They were measured on a small stand-in model, not the real CLIP ViT-B/16, so they are not valid. CLIP on the Pi is measured in `../kd2/PI_CLIP_BENCH.md`.


| Model | Pi 5 CPU (3 thr) ms | T4 GPU compute ms | T4 with copy ms (p95) | speedup | clips/s max | cameras/GPU (1 clip per 0.6 s, no motion gate) |
|---|---|---|---|---|---|---|
| MoViNet-A0 | 137 | 39.4 | 40.3 (40.5) | 3.4x | 26.5 | 16 |
| MoViNet-A1 | 256 | 64.0 | 65.0 (65.2) | 3.9x | 16.7 | 10 |
| X3D-S | 196 | 12.1 | 12.6 (12.7) | 15.5x | 89.0 | 53 |
Heads 0.2 ms. Kaggle-CPU preprocess per clip ~5 ms. A0 full clip latency (preprocess + backbone + heads) 45.6 ms (hop 600 ms -> 13x margin).
Batching: MoViNet A0/A1 per-clip time flat across batch 1-32 (~38 / ~62 ms per clip) -> no throughput gain; X3D ~11-12 ms per clip. Threads 1-8: same totals.
fp16 (onnxconverter-common, keep_io_types): A0 cos 0.99999, 39.0 ms (tiny gain); A1 cos 1.00000, no gain; X3D cos NaN -> fp16 broken, not used.
Sustained 60 s A0: flat 41.1-41.2 ms (v1.0 and v1.2).
Takeaway: on GPU the ranking flips (X3D 3.2x faster than A0) while on Pi CPU A0 is fastest; A0 still more accurate (video AUC 0.919 vs 0.884, significant). Report GPU as reference, Pi as deployment. Never claim cameras-with-gate numbers as measured.
## v1.2 run (2026-10-16, same T4, ORT 1.23.2 CUDA EP) - all v1.0 numbers reproduced
Latency: A0 40.24, A1 64.97, X3D 12.67; full A0 clip 46.2 ms.
v1.2 p95 (with copy): A0 40.4, A1 65.1, X3D 12.8. Max clips/s at batch 1: A0 26.4, A1 16.7, X3D 88.7 (cameras at 0.6 s hop: 16 / 10 / 53). Preprocessing per clip on Kaggle CPU about 5 ms. (Values copied from the pasted v1.2 Kaggle log, 2026-10-16.)
Telemetry, 60 s A0 soak (rows 10-40 s; paste cut at 50 s): 41.1-41.2 ms, GPU util 95-98%, temp 77-78 C, power 64-67 W, SM clock 1455-1470 MHz, GPU0 memory 7171 MB (includes ORT pools of all loaded sessions).
GPU memory added per model: A0 207 MB, A1 160 MB, X3D 94 MB.
Node placement (ORT profiling): all nodes on CUDA EP, no CPU fallback. A0 285 nodes, 19.3 ms kernel time, Conv 16.6 ms (86%); A1 429 nodes 25.2 ms, Conv 20.9; X3D 285 nodes 21.5 ms, Conv 19.3. -> MoViNet's lack of batch gain is NOT CPU fallback; 3D depthwise convs map poorly to GPU.
TensorRT: not available on Kaggle (libnvinfer.so.10 missing) -> paper says "could not be evaluated".
