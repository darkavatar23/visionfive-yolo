#!/bin/bash
# Build ONLY bench_ncnn_generic (static), linking the already cross-compiled libncnn.a.
# Fast: does not rebuild ncnn. Requires out/lib/libncnn.a + out/include/ncnn (from build_all.sh).
set -e
OUT=/home/matteo/riscv-build/out
docker run --rm --platform linux/riscv64 -v "$OUT":/out -v "$PWD/src":/src \
  debian:trixie-slim bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ file >/dev/null
g++ -O3 -std=c++17 -static -fopenmp -I/out/include/ncnn \
    /src/bench_ncnn_generic.cpp /out/lib/libncnn.a -o /out/bin/bench_ncnn_generic
strip /out/bin/bench_ncnn_generic
file /out/bin/bench_ncnn_generic
'
