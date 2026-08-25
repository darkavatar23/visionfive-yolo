#!/usr/bin/env python3
"""Uniform cross-platform benchmark through onnxruntime.
Measures PURE inference (net.run) on a fixed dummy input -> fair comparison across
macOS (CPU/CoreML), Linux (CPU), Windows (CUDA). The RISC-V board uses NCNN separately.
Usage: bench_onnx.py <provider> <n_iter> <onnx...>
  provider: cpu | coreml | cuda
  e.g. bench_onnx.py cpu 30 onnx/yolov8n_320.onnx onnx/yolo26n_320.onnx
"""
import sys
import time

import numpy as np
import onnxruntime as ort

# Windows: the CUDA pip wheels (nvidia-*-cu12) are not on the DLL path until preloaded.
try:
    ort.preload_dlls()
except Exception:
    pass

PROV = {'cpu': ['CPUExecutionProvider'],
        'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider']}[sys.argv[1]]
NITER = int(sys.argv[2])
MODELS = sys.argv[3:]

print(f'provider={sys.argv[1]} ({",".join(p.split("Execution")[0] for p in PROV)})  iter={NITER}')
print(f'onnxruntime {ort.__version__}')
print(f'{"model":<20}{"imgsz":>6}{"ms/frame":>10}{"FPS":>8}   EP-used')
for m in MODELS:
    so = ort.SessionOptions()
    sess = ort.InferenceSession(m, sess_options=so, providers=PROV)
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]  # es [1,3,320,320]
    sz = shape[-1]
    x = np.random.rand(*shape).astype(np.float32)
    name = inp.name
    # warmup
    for _ in range(3):
        sess.run(None, {name: x})
    ts = []
    for _ in range(NITER):
        t = time.time()
        sess.run(None, {name: x})
        ts.append(time.time() - t)
    ts.sort()
    med = ts[len(ts) // 2]
    used = sess.get_providers()[0].split('Execution')[0]
    label = m.split('/')[-1].replace('.onnx', '')
    print(f'{label:<20}{sz:>6}{med*1000:>10.1f}{1/med:>8.1f}   {used}')
