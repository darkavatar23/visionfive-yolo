#!/usr/bin/env python3
"""YOLO (NCNN) on an RTSP stream with a web UI (MJPEG + stats). LAN-only, zero cloud.
Inference is delegated to the static C++ binary `yolo_ncnn` (model loaded once,
frames passed via /dev/shm). opencv is only used for RTSP capture + drawing + jpeg.

Usage: mjpeg_yolo_ncnn.py <model_prefix> <imgsz> [conf] [port]
  e.g. mjpeg_yolo_ncnn.py models/yolo11n 320 0.30 8000
"""
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

MODEL = sys.argv[1]
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 320
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
THREADS = int(os.environ.get('THREADS', '4'))
NMS = 0.45
RTSP = os.environ.get('RTSP_URL', 'rtsp://USER:PASS@CAM_IP:554/stream2')
BIN = os.environ.get('YOLO_BIN', './yolo_ncnn')
RAW = '/dev/shm/vf2_frame.raw'
SAMPLE_PERIOD = float(os.environ.get('SAMPLE_PERIOD', '0.5'))

NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
         'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
         'toothbrush']
np.random.seed(7)
COLORS = np.random.randint(60, 255, (80, 3)).tolist()

shared = {'jpg': None, 'stats': {'model': os.path.basename(MODEL), 'imgsz': IMGSZ,
                                 'infer_ms': 0, 'age_ms': 0, 'objs': [], 'temp': 0, 'res': ''}}
lock = threading.Lock()
# freshest-frame capture: continuously drain the stream, keep only the newest frame
latest = {'frame': None, 'ts': 0.0}
milk_lock = threading.Lock()


def cpu_temp():
    try:
        return round(int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000, 1)
    except Exception:
        return 0


def milk():
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
        'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0'
    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.3)
            cap.open(RTSP, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        with milk_lock:
            latest['frame'] = img
            latest['ts'] = time.time()


def worker():
    # start the C++ detector (model loaded once)
    proc = subprocess.Popen(
        [BIN, MODEL, str(IMGSZ), str(CONF), str(NMS), str(THREADS)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    # wait for READY on stderr
    ready = proc.stderr.readline().decode(errors='replace').strip()
    print('[detector]', ready, flush=True)
    ema = None
    last_ts = 0.0
    while True:
        cycle0 = time.time()
        with milk_lock:
            img = latest['frame']
            ts = latest['ts']
        if img is None or ts == last_ts:
            time.sleep(0.03)
            continue
        last_ts = ts
        age_ms = int((time.time() - ts) * 1000)
        h, w = img.shape[:2]
        # write the raw BGR frame to /dev/shm and query the detector
        img.tofile(RAW)
        t0 = time.time()
        proc.stdin.write(f'{RAW} {w} {h}\n'.encode())
        proc.stdin.flush()
        n = int(proc.stdout.readline().decode().strip() or '0')
        dets = []
        for _ in range(n):
            parts = proc.stdout.readline().decode().split()
            c = int(parts[0]); s = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            dets.append((c, s, x1, y1, x2, y2))
        dt = time.time() - t0
        ema = dt if ema is None else 0.8 * ema + 0.2 * dt
        objs = {}
        for c, s, x1, y1, x2, y2 in dets:
            col = COLORS[c % 80]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(img, f'{NAMES[c] if c < 80 else c} {s:.2f}',
                        (int(x1), max(12, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            nm = NAMES[c] if c < 80 else str(c)
            objs[nm] = objs.get(nm, 0) + 1
        hud = f'{os.path.basename(MODEL)} NCNN @{IMGSZ}  {ema*1000:.0f} ms  frame {age_ms}ms old  {cpu_temp():.0f}C'
        cv2.rectangle(img, (0, 0), (len(hud) * 9 + 12, 24), (0, 0, 0), -1)
        cv2.putText(img, hud, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        ok, jbuf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            shared['jpg'] = jbuf.tobytes()
            shared['stats'] = {'model': os.path.basename(MODEL), 'imgsz': IMGSZ,
                               'infer_ms': round(ema * 1000), 'age_ms': age_ms,
                               'objs': [f'{k}×{v}' for k, v in sorted(objs.items())],
                               'temp': cpu_temp(), 'res': f'{w}x{h}'}
        rest = SAMPLE_PERIOD - (time.time() - cycle0)
        if rest > 0:
            time.sleep(rest)


PAGE = b"""<!doctype html><html><head><meta charset=utf-8><title>VisionFive 2 - YOLO NCNN live</title>
<style>body{background:#111;color:#0f0;font-family:monospace;margin:0;padding:12px}
h1{font-size:16px;color:#8f8}#wrap{display:flex;gap:16px;flex-wrap:wrap}
img{border:2px solid #0a0;max-width:100%}#stats{font-size:14px;line-height:1.7}
.k{color:#6a6}.v{color:#cfc;font-weight:bold}
#objs span{display:inline-block;background:#032;border:1px solid #0a0;padding:2px 8px;margin:2px;border-radius:4px}</style>
</head><body><h1>StarFive VisionFive 2 - YOLO on NCNN (cross-compiled) - LAN-only RTSP, zero cloud</h1>
<div id=wrap><img src="/stream">
<div id=stats>
<div><span class=k>model:</span> <span class=v id=model>-</span></div>
<div><span class=k>inference:</span> <span class=v id=ms>-</span> ms/frame</div>
<div><span class=k>frame age:</span> <span class=v id=age>-</span> ms</div>
<div><span class=k>resolution:</span> <span class=v id=res>-</span></div>
<div><span class=k>CPU temp:</span> <span class=v id=temp>-</span> C</div>
<div style=margin-top:10px><span class=k>detected objects:</span></div>
<div id=objs></div></div></div>
<script>setInterval(async()=>{let s=await(await fetch('/stats')).json();
model.textContent=s.model+' @'+s.imgsz;ms.textContent=s.infer_ms;age.textContent=s.age_ms;
res.textContent=s.res;temp.textContent=s.temp;
objs.innerHTML=s.objs.length?s.objs.map(o=>'<span>'+o+'</span>').join(''):'<i style=color:#666>none</i>';},700);</script>
</body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == '/stats':
            with lock:
                s = json.dumps(shared['stats']).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(s)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
            try:
                while True:
                    with lock:
                        j = shared['jpg']
                    if j:
                        self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + j + b'\r\n')
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


threading.Thread(target=milk, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()
print(f'UI: http://0.0.0.0:{PORT}  (NCNN {MODEL} @{IMGSZ})', flush=True)
Server(('0.0.0.0', PORT), H).serve_forever()
