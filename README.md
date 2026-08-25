# visionfive-yolo

**Edge object detection** on the **StarFive VisionFive 2** (JH7110 SoC, 4× SiFive U74 @ 1.5 GHz, RISC-V `rv64gc` **without the RVV vector extension**, 4 GB RAM, Debian riscv64).
Video source: a **TP-Link Tapo** camera over **LAN-only RTSP** (camera/NVR account), **zero cloud**.
Inference runtime: **NCNN cross-compiled** for riscv64. Live web UI over **MJPEG**.

Includes: a static C++ detector, a server with a live UI, reproducible cross-build scripts, **prebuilt NCNN models** (yolov8n/11n/12n/26n at multiple resolutions + yolo-fastestv2) and a complete **cross-platform benchmark** (RISC-V board vs Ryzen vs Apple M4 vs RTX 3090) — see [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Configuration (no credentials in this repo)

The video source and hosts are passed via environment variables — replace the placeholders:

```bash
export RTSP_URL="rtsp://USER:PASS@CAM_IP:554/stream2"   # the camera's NVR account
```

> ⚠️ **AGPL-3.0 license**: this repo ships weights exported from [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8/11/12/26), which are released under **AGPL-3.0**; the whole repo is therefore AGPL-3.0. yolo-fastestv2 and NCNN carry their own (BSD-like) licenses. Commercial use of the Ultralytics models requires an Enterprise license from Ultralytics.

## Why this architecture

- **opencv DNN 4.6** (the only apt-installable version for riscv64 on the vendor image) is from 2022 → it **cannot load YOLOv8/YOLO11/YOLO26** (unsupported `TopK`/`C2PSA`/`Floor` operators). It only handles YOLOv5.
- **NCNN** loads YOLOv8/11/26 (its export automatically falls back to the classic NMS head) and is the practical way to run modern models on this SoC.
- The **ncnn Python binding** would depend on *glibc* **and** the Python version (board 3.11 vs container 3.12/3.13) → fragile. Inference therefore runs in a **static C++ binary** (`yolo_ncnn`) with no glibc/python dependencies, usable on any riscv64 rootfs. The Python server drives it by passing frames through `/dev/shm`.

## ABI constraints (important)

The board runs **glibc 2.36**. Available riscv64 containers ship newer glibc (≥2.39). Therefore:
- the **detector** is linked **fully static** (`-static`) → glibc becomes irrelevant;
- **always fp32**: on a U74 without RVV, int8 is ~9× *slower* (no vector kernels), and fp16 storage does not help either.

## Layout

```
src/yolo_ncnn.cpp         static C++ detector (NCNN), /dev/shm protocol over stdin/stdout
src/bench_ncnn_generic.cpp pure-forward benchmark for any NCNN model
server/mjpeg_yolo_ncnn.py  RTSP -> NCNN -> MJPEG server (drives the C++ binary)
server/mjpeg_yolo.py       RTSP -> opencv-DNN -> MJPEG server (YOLOv5 fallback, no compilation)
server/yolo_cam.py         opencv-DNN CLI benchmark
server/bench_onnx.py       cross-platform onnxruntime benchmark (cpu / coreml / cuda)
scripts/build_all.sh       NCNN cross-build + static detector link (riscv64 Docker via qemu binfmt on x86)
models/                    .param/.bin models (ncnn export from Ultralytics), sizes 256/320/640
onnx/                      ONNX models used for the cross-platform benchmark
bin/                       prebuilt riscv64 binaries (yolo_ncnn, bench_ncnn_generic)
docs/BENCHMARK.md          all measured results
docs/BUILD_NOTES.md        cross-build problems solved along the way
```

## Build pipeline (on an x86 host with Docker + qemu-user-static)

```bash
scripts/build_all.sh      # -> out/lib/libncnn.a  +  out/bin/yolo_ncnn (static)
```

Model export (on a Mac / any x86 machine with Ultralytics):

```bash
yolo export model=yolo11n.pt format=ncnn imgsz=320   # -> yolo11n_ncnn_model/{model.ncnn.param,model.ncnn.bin}
```

## Deploy on the board

```bash
scp bin/yolo_ncnn models/yolo11n.{param,bin} user@<board>:~
ssh user@<board> "setsid sh -c 'RTSP_URL=rtsp://USER:PASS@<cam>:554/stream2 \
  YOLO_BIN=./yolo_ncnn exec python3 mjpeg_yolo_ncnn.py yolo11n 320 0.30 8000' </dev/null >~/y.log 2>&1 &"
```
UI: `http://<board>:8000`

## Performance (VisionFive 2, 4 threads — measured). Full details: [docs/BENCHMARK.md](docs/BENCHMARK.md)

| Runtime | Model | Input | ms/frame | FPS | Notes |
|---|---|---|---|---|---|
| opencv DNN 4.6 | YOLOv5n | 256 | ~410 | 2.4 | 2020 model, only one opencv 4.6 can load |
| opencv DNN 4.6 | YOLOv5n | 320 | ~630 | 1.6 | |
| **NCNN** | **YOLO26n** | **256** | **826** | **1.21** | **2026 model, best quality pick** |
| NCNN | YOLOv8n | 256 | 821 | 1.22 | |
| NCNN | YOLO11n | 256 | 883 | 1.13 | |
| NCNN | yolo-fastestv2 | 352 | 433 | 2.31 | fastest option, lower accuracy |

![YOLO inference latency across platforms, 320 vs 1920 input, log scale](docs/benchmark_chart.svg)

**In short**: modern models (YOLO26n) only run through NCNN and cost ~2× the time of YOLOv5n, in exchange for much better detection quality. fp16 doesn't help, int8 makes it worse (no RVV). The ceiling is the scalar CPU: an NVMe or fp16 will not move FPS. Recommended pick: **YOLO26n @256 (~1.2 FPS)** for quality, YOLOv5n/opencv @256 (2.4 FPS) for fluidity, yolo-fastestv2 (2.3 FPS) as the lightweight compromise.

## Notes

- The MJPEG servers use a *freshest-frame* capture thread (continuous `cap.read()` keeping only the newest frame + FFmpeg low-latency flags), so inference always sees a frame that is a few tens of ms old instead of a stale buffered one.
- `docs/serial_bridge.py`: stdlib-only serial console bridge (for UART debugging, if ever needed).
