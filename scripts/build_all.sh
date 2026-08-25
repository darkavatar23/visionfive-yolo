#!/bin/bash
# All-in-one: cross-build NCNN (lib) + static link of the detector (src/yolo_ncnn.cpp)
# in ONE riscv64 container (qemu binfmt on x86). Board = glibc 2.36.
# - ncnn lib: OpenMP ON (uses the 4 U74 cores), RVV/XTHEADVECTOR OFF (no vector unit)
# - detector: -static -fopenmp (verified: static binary, zero glibc deps)
# - no benchncnn/tools (-static breaks the benchncnn link and it is not needed)
set -e
OUT=/home/matteo/riscv-build/out
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/lib"

docker run --rm --platform linux/riscv64 \
  -v "$OUT":/out -v "$PWD/src":/src \
  debian:trixie-slim bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
echo "=== deps ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential cmake git ca-certificates binutils file >/dev/null
echo "gcc $(gcc -dumpversion)  glibc $(ldd --version|head -1)"

echo "=== clone ncnn ==="
cd /tmp && git clone --depth 1 https://github.com/Tencent/ncnn.git 2>&1 | tail -1
cd ncnn && echo "commit $(git rev-parse --short HEAD)"

echo "=== configure (RVV/XTHEAD OFF, OpenMP ON, lib only) ==="
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_RVV=OFF -DNCNN_XTHEADVECTOR=OFF -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_PYTHON=OFF \
      -DNCNN_SHARED_LIB=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=ON \
      .. 2>&1 | tail -2

echo "=== build lib ncnn (-j$(nproc)) ==="
make -j$(nproc) ncnn 2>&1 | tail -3
cp src/libncnn.a /out/lib/
# save the headers so the detector alone can be rebuilt without recompiling ncnn
mkdir -p /out/include/ncnn
cp /tmp/ncnn/src/*.h /out/include/ncnn/ 2>/dev/null || true
cp /tmp/ncnn/build/src/*.h /out/include/ncnn/ 2>/dev/null || true
echo "libncnn.a: $(ls -la /out/lib/libncnn.a | awk "{print \$5}") bytes, headers saved"

echo "=== build STATIC detector (-static -fopenmp) ==="
g++ -O3 -std=c++17 -static -fopenmp \
    -I/tmp/ncnn/src -I/tmp/ncnn/build/src \
    /src/yolo_ncnn.cpp /out/lib/libncnn.a \
    -o /out/bin/yolo_ncnn
strip /out/bin/yolo_ncnn
echo "--- detector ---"; file /out/bin/yolo_ncnn
echo "static? ->"; ldd /out/bin/yolo_ncnn 2>&1 | head -1
echo "=== ABI CHECK (static: must report no dynamic glibc deps) ==="
objdump -T /out/bin/yolo_ncnn 2>&1 | grep -oE "GLIBC_[0-9]+\.[0-9]+" | sort -uV | tail -3 || echo "(no dynamic glibc symbols: OK for a glibc-2.36 board)"
ls -la /out/bin /out/lib
echo "=== ALL DONE ==="
'
