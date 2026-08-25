# Benchmark — YOLO on the StarFive VisionFive 2

Hardware: **JH7110, 4× SiFive U74 @ 1.5 GHz, RISC-V `rv64gc` without RVV**, 4 GB RAM, Debian riscv64 (glibc 2.36), heatsink + Noctua fan.
Source: **Tapo camera** over LAN-only RTSP, `640×360` substream. Zero cloud.

## Methodology

- One real RTSP frame (640×360) captured once, saved raw to `/dev/shm`, fed N times to the model.
- 12–14 iterations, **first 2 discarded** (kernel warm-up), average/median over the rest.
- 4 threads (OpenMP), fp32 unless stated. `conf=0.30`, `nms=0.45`.
- NCNN detector = static C++ binary (`bin/yolo_ncnn`); opencv runtime = `server/yolo_cam.py`.
- **Runtime `imgsz` must match the export `imgsz`** (models exported with `dynamic=False`): running a 320 model at 256 produces garbage output — see the note at the bottom.

## opencv DNN 4.6 runtime (reference)

opencv 4.6 (the only apt version for riscv64) **cannot load YOLOv8/11/12/26** (`TopK`, `C2PSA`, NMS-free heads): it fails in `shape_utils`. It only handles **YOLOv5** (2020 graph). Steady state:

| Model | Input | ms/frame | FPS |
|---|---|---|---|
| YOLOv5n | 320 | ~630 | 1.6 |
| YOLOv5n | 256 | ~410 | 2.4 |

## NCNN runtime (cross-compiled, this repo)

NCNN loads **all** the models (the Ultralytics ncnn export disables the end-to-end head → classic NMS).

### @320, fp32

| Model | ms/frame | FPS |
|---|---|---|
| YOLOv8n | 1206 | 0.83 |
| YOLO11n | 1211 | 0.83 |
| YOLO26n | 1233 | 0.81 |
| YOLO12n | 1671 | 0.60 |

### @320, fp16 storage (NOT worth it)

| Model | ms/frame | FPS |
|---|---|---|
| YOLOv8n | 1121 | 0.89 |
| YOLO11n | 1320 | 0.76 |
| YOLO26n | 1270 | 0.79 |

fp16 storage **does not help** on the U74: without vector hardware, the software fp16↔fp32 conversion costs as much as the bandwidth it saves. **Stick to fp32.**

### @256, fp32 (recommended)

| Model | ms/frame | FPS | Objects detected |
|---|---|---|---|
| YOLOv8n | 821 | 1.22 | person×2 |
| YOLO26n | 826 | 1.21 | person×1, bottle×1 |
| YOLO11n | 883 | 1.13 | person×1, bottle×1 |

## Cross-platform comparison (board vs Ryzen vs Apple M4 vs RTX 3090)

Same network, **pure inference** (forward on a fixed dummy input, warm-up excluded, median).
**Best available runtime per platform**: board=NCNN (riscv64), Ryzen/Mac=onnxruntime, RTX 3090=onnxruntime CUDA.
It is not the same runtime everywhere (impossible: NCNN has no GPU path here, onnxruntime has no riscv64 wheel) — it is the realistic "how fast does X go on its platform with its best stack" comparison.

Hardware: **VisionFive 2** (4× U74 1.5 GHz, RISC-V no-RVV) · **Ryzen 7 3800XT** (2020 desktop CPU) · **Apple M4** (CPU and CoreML = GPU/ANE) · **RTX 3090** (Ampere, 24 GB, onnxruntime CUDA + cuDNN 9.8).

### YOLOv8n — ms/frame (FPS in parentheses)

| imgsz | Board NCNN | Ryzen CPU | M4 CPU | M4 CoreML | RTX 3090 CUDA |
|------:|-----------:|---------:|-----------:|--------------:|--------------:|
| 320  | 1006 (1.0)  | 12.9 (77) | 5.2 (192) | **1.2 (862)** | 5.0 (200) |
| 640  | 3752 (0.27) | 43.7 (23) | 18.2 (55) | 6.8 (148)     | 5.0 (200) |
| 960  | —           | 97.5 (10) | 43.9 (23) | 16.9 (59)     | 8.0 (125) |
| 1280 | —           | 182 (5.5) | 82.6 (12) | 30.0 (33)     | 13.5 (74) |
| 1920 | ~30 s* (est)| 460 (2.2) | 200 (5.0) | 96.9 (10.3)   | **31.0 (32)** |

### YOLO26n (2026) — ms/frame (FPS)

| imgsz | Board NCNN | M4 CPU | M4 CoreML | RTX 3090 CUDA |
|------:|-----------:|-----------:|--------------:|--------------:|
| 320  | 1307 (0.77) | 5.3 (188) | 2.5 (393) | 4.5 (222) |
| 1920 | — (impractical) | 197 (5.1) | 189 (5.3) | **29 (34.5)** |

### yolo-fastestv2 (ultra-light model) — board only

| Model | imgsz | Board NCNN |
|---|---:|---:|
| yolo-fastestv2 | 352 | **433 ms (2.3 FPS)** |

3× faster than yolo26n/yolov8n on the board: the only near-realtime option on this SoC (at the cost of accuracy).

### Camera at full resolution (stream1 1920×1080) — on the board

Real capture + yolo26n@320 inference (1920→320 letterbox + forward + decode): **1567 ms end-to-end (0.64 FPS)**. Opening the HD RTSP stream costs ~8 s once. The 1920 resize adds ~250 ms/frame on the scalar CPU compared to the 640×360 substream.

### What the numbers say

- **Apple M4 CoreML dominates small models**: 862 FPS @320, even beating the 3090 — at low resolution the GPU kernel-launch latency outweighs the compute, and the ANE is unbeatable there.
- **The RTX 3090 dominates at scale**: flat 200 FPS from 320 to 640, and it holds **1920 at 32 FPS** where everything else collapses. It is the only platform that makes real HD practical.
- **M4 CPU ≈ 3090 @320** (latency-bound), but the 3090 pulls away 6× at 1920.
- **The Ryzen 3800XT** (2020) is the slowest of the modern group, yet still **50–100× the board**.
- **The board** is 2–3 orders of magnitude behind the rest: the price of a scalar RISC-V CPU with no SIMD/vector unit. It remains useful for low-frequency edge tasks (1 scan/second, presence, counting), not fluid video.

*Board @1920 not actually measured (≈30–40 s/frame extrapolated from 640 = 3.7 s): impractical — use the HD source but process at 320.

## Thermals (heatsink + Noctua fan)

| State | Temp |
|---|---|
| Idle (YOLO at 1 fps) | ~42 °C |
| Full 4-core load (40 s) | ~45 °C |
| JH7110 throttling threshold | ~85 °C |

Plenty of headroom: full-speed operation without throttling.

## Note: export imgsz ≠ runtime imgsz

Models are exported with `dynamic=False` at a fixed resolution. Running them at a different one does **not** resize the graph: the output becomes noise (hundreds of boxes with random classes) and slower. Each resolution needs its own export (`*_320`, `*_256`).
