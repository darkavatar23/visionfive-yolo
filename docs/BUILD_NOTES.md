# Build notes — cross-compiling NCNN for the VisionFive 2

A chronicle of the real problems hit while cross-compiling NCNN for riscv64 (board glibc 2.36)
on x86 (Ryzen, 16 threads) through Docker + qemu-user-static (binfmt).
Each one was solved; written down so nobody repeats them.

## 1. No Debian `bookworm` riscv64 image exists
Debian promoted `riscv64` to an official architecture only in **trixie**. There is no
`riscv64/debian:bookworm`. glibc available in riscv64 containers: trixie 2.41,
sid 2.43, Ubuntu noble 2.39 — **all newer than the board (2.36)**.

## 2. glibc ABI: built on 2.41 → runs on 2.36
A dynamic binary linked against glibc 2.41 may reference symbols absent from 2.36.
**Solution**: link the detector **fully static** (`-static`) → glibc becomes irrelevant,
it runs on any riscv64 rootfs. Verified: `file` says "statically linked",
`ldd` says "not a dynamic executable".

## 3. The Python binding is fragile twice over
The ncnn Python module would depend on glibc **and** on the Python version.
The container ships Python 3.13, the board 3.11 → an incompatible cp313 `.so`.
**Solution**: no Python binding. Inference runs in a static C++ process,
frames passed from the Python server through `/dev/shm` (stdin/stdout protocol).

## 4. gcc 14 ICE on `convolution_riscv_xtheadvector.cpp`
Even with `NCNN_RVV=OFF`, ncnn still compiles the **T-Head xtheadvector** vector kernels,
and gcc 14 hits an *internal compiler error* on that file.
**Solution**: `-DNCNN_XTHEADVECTOR=OFF` (the U74 has no vector extension of any kind).

## 5. Static linking of `benchncnn` fails
`-static` in `CMAKE_EXE_LINKER_FLAGS` breaks the `benchncnn` link.
**Solution**: `-DNCNN_BUILD_BENCHMARK=OFF` (not needed) and apply `-static` **only**
to the detector in the manual `g++` command. Verified separately that `-static -fopenmp`
produces a valid static riscv64 ELF (libgomp.a ships with gcc-14).

## 6. qemu-user is NOT single-threaded
`make -j16` under qemu uses all cores (measured: 14–16 `cc1plus` processes, load ~12);
`nproc` inside the container correctly reports 16. The slowness is the **TCG overhead**
(~5–10× per instruction), not a lack of parallelism. The serial phases (apt, git clone,
cmake configure, final link) create the false impression of single-threading.
For fast native builds, use the **cross-toolchain** `g++-riscv64-linux-gnu` (runs native x86).

## Final NCNN configuration

```
-DNCNN_RVV=OFF -DNCNN_XTHEADVECTOR=OFF -DNCNN_VULKAN=OFF
-DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_BENCHMARK=OFF
-DNCNN_PYTHON=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=ON
```
Detector: `g++ -O3 -std=c++17 -static -fopenmp -I<ncnn>/src -I<ncnn>/build/src yolo_ncnn.cpp libncnn.a`

## NCNN runtime options evaluated (net.opt)
- `num_threads=4` → uses all 4 U74 cores (confirmed ~91% aggregate CPU).
- `use_fp16_storage/packed` → **no gain** on the U74 (see BENCHMARK.md).
- `use_sgemm_convolution`, `use_winograd_convolution` → ON (useful defaults).
- `use_fp16_arithmetic=OFF` mandatory (no fp16 hardware).

## Bonus: GPU pitfalls met while benchmarking the other platforms
- **onnxruntime-gpu on Windows** silently falls back to CPU unless you call
  `onnxruntime.preload_dlls()` (pip-wheel CUDA DLLs are not on the DLL path) **and**
  pin a compatible cuDNN (`nvidia-cudnn-cu12==9.8.0.87` — 9.24 fails with
  `CUDNN_BACKEND_API_FAILED`). Always check which execution provider actually ran.
- `pip install ultralytics` can **replace** a CUDA torch with the CPU build. Reinstall
  torch from the CUDA index afterwards if you need it.
