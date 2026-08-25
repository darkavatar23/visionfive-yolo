#!/usr/bin/env python3
"""Comparative YOLO model benchmark through the NCNN detector on the board.
Captures 1 RTSP frame, feeds it N times to each model, measures ms/frame
(discarding warm-up) and prints the detected objects.
Usage: bench_ncnn.py <imgsz> <n_iter> <model1> [model2 ...]
"""
import os
import subprocess
import sys
import time

import cv2

IMGSZ = int(sys.argv[1])
NITER = int(sys.argv[2])
MODELS = sys.argv[3:]
THREADS = int(os.environ.get('THREADS', '4'))
CONF, NMS = 0.30, 0.45
RTSP = os.environ.get('RTSP_URL', 'rtsp://USER:PASS@CAM_IP:554/stream2')
BIN = os.environ.get('YOLO_BIN', './yolo_ncnn')
RAW = '/dev/shm/bench.raw'
NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow'] + [str(i) for i in range(20, 80)]

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer'
cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
for _ in range(5):
    cap.grab()
ok, img = cap.retrieve()
cap.release()
if not ok:
    print('no frame'); sys.exit(1)
h, w = img.shape[:2]
img.tofile(RAW)
print(f'frame {w}x{h}, imgsz={IMGSZ}, {NITER} iter, {THREADS} thread\n')
print(f'{"model":<12} {"ms/frame":>9} {"FPS":>6}   objects')

for m in MODELS:
    proc = subprocess.Popen([BIN, m, str(IMGSZ), str(CONF), str(NMS), str(THREADS)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    proc.stderr.readline()  # READY
    times, objs = [], {}
    for i in range(NITER):
        t0 = time.time()
        proc.stdin.write(f'{RAW} {w} {h}\n'.encode())
        proc.stdin.flush()
        n = int(proc.stdout.readline().decode().strip() or 0)
        last = []
        for _ in range(n):
            p = proc.stdout.readline().decode().split()
            last.append(int(p[0]))
        if i > 1:  # discard 2 warm-up rounds
            times.append(time.time() - t0)
        if i == NITER - 1:
            for c in last:
                nm = NAMES[c] if c < len(NAMES) else str(c)
                objs[nm] = objs.get(nm, 0) + 1
    proc.stdin.close()
    proc.terminate()
    avg = sum(times) / len(times) if times else 0
    ol = ', '.join(f'{k}×{v}' for k, v in sorted(objs.items())) or '-'
    name = os.path.basename(m)
    print(f'{name:<12} {avg*1000:>9.0f} {1/avg:>6.1f}   {ol}')
