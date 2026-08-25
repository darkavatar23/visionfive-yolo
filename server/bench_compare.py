#!/usr/bin/env python3
"""NCNN comparison benchmark: fp32/fp16 x models x imgsz, on the board.
Captures 1 RTSP frame and feeds it N times to each combination. Prints ms/frame, FPS, objects.
Usage: bench_compare.py <n_iter> <imgsz> <fp16 0|1> <model_prefix...>
  e.g. bench_compare.py 14 256 0 yolov8n_256 yolo11n_256 yolo26n_256
"""
import os, subprocess, sys, time, cv2
NITER = int(sys.argv[1]); IMGSZ = int(sys.argv[2]); FP16 = sys.argv[3]
MODELS = sys.argv[4:]
BIN = os.environ.get('YOLO_BIN', './yolo_ncnn')
RTSP = os.environ.get('RTSP_URL', 'rtsp://USER:PASS@CAM_IP:554/stream2')
NAMES = {0:'person',39:'bottle',41:'cup',56:'chair',62:'tv',63:'laptop',67:'cell phone',73:'book'}
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
c = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
for _ in range(5): c.grab()
ok, img = c.retrieve(); c.release()
img.tofile('/dev/shm/bench.raw'); h, w = img.shape[:2]
print(f'frame {w}x{h}  imgsz={IMGSZ}  fp16={FP16}  iter={NITER}\n{"model":<14}{"ms":>7}{"FPS":>7}   objects')
for m in MODELS:
    p = subprocess.Popen([BIN, m, str(IMGSZ), '0.30', '0.45', '4', FP16],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    p.stderr.readline()
    ts, objs = [], {}
    for k in range(NITER):
        t = time.time(); p.stdin.write(f'/dev/shm/bench.raw {w} {h}\n'.encode()); p.stdin.flush()
        n = int(p.stdout.readline() or 0); last = []
        for _ in range(n): last.append(int(p.stdout.readline().split()[0]))
        if k > 1: ts.append(time.time() - t)
        if k == NITER - 1:
            for cc in last: objs[cc] = objs.get(cc, 0) + 1
    p.stdin.close(); p.terminate()
    a = sum(ts) / len(ts)
    ol = ', '.join(f'{NAMES.get(k,k)}×{v}' for k, v in sorted(objs.items())) or '-'
    print(f'{os.path.basename(m):<14}{a*1000:>7.0f}{1/a:>7.2f}   {ol}')
