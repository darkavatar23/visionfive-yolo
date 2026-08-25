#!/usr/bin/env python3
"""YOLO on an RTSP stream with a web UI (MJPEG + stats), opencv-DNN runtime.
Fallback path for boards where only opencv 4.6 is installable (loads YOLOv5 ONNX only).
LAN-only, zero cloud.
Usage: mjpeg_yolo.py model.onnx imgsz [conf] [port]
Open http://<board-ip>:8000 from any browser on the LAN.
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import time

import cv2
import numpy as np

MODEL = sys.argv[1]
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 320
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
RTSP = os.environ.get('RTSP_URL', 'rtsp://USER:PASS@CAM_IP:554/stream2')

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
# stable per-class color
np.random.seed(7)
COLORS = np.random.randint(60, 255, (80, 3)).tolist()

shared = {'jpg': None, 'stats': {'fps': 0, 'infer_ms': 0, 'objs': [], 'temp': 0, 'res': '', 'model': os.path.basename(MODEL), 'imgsz': IMGSZ}}
lock = threading.Lock()
# freshest-frame capture: continuously drain the stream, keep only the newest frame
latest = {'frame': None, 'ts': 0.0}
milk_lock = threading.Lock()
SAMPLE_PERIOD = float(os.environ.get('SAMPLE_PERIOD', '1.0'))  # 1 frame/sec


def cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return round(int(f.read()) / 1000, 1)
    except Exception:
        return 0


def milk():
    # FFmpeg low-latency options: TCP, no buffering, drop stale frames
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = \
        'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0'
    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ok, img = cap.read()               # ALWAYS consume: the pipe stays empty
        if not ok:
            time.sleep(0.3)
            cap.open(RTSP, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        with milk_lock:
            latest['frame'] = img
            latest['ts'] = time.time()


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return canvas, r


def parse(out, r, conf):
    """Handles yolov5 [1,N,85] (box+obj+80cls) and yolov8/11 [1,84,N] (transposed)."""
    if out.ndim == 3 and out.shape[2] == 85:          # yolov5
        p = out[0]                                     # N x 85
        obj = p[:, 4]
        cls_scores = p[:, 5:] * obj[:, None]
    else:                                              # yolov8/11: [1,84,N] -> N x 84
        p = out[0].T
        cls_scores = p[:, 4:]
    cls = cls_scores.argmax(1)
    sc = cls_scores[np.arange(len(cls)), cls]
    m = sc > conf
    boxes, confs, classes = [], [], []
    for (cx, cy, w, h), s, c in zip(p[m, :4], sc[m], cls[m]):
        boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
        confs.append(float(s))
        classes.append(int(c))
    dets = []
    if boxes:
        for i in np.array(cv2.dnn.NMSBoxes(boxes, confs, conf, 0.45)).flatten():
            x, y, w, h = boxes[i]
            dets.append((classes[i], confs[i], x / r, y / r, (x + w) / r, (y + h) / r))
    return dets


def worker():
    net = cv2.dnn.readNetFromONNX(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    cv2.setNumThreads(4)
    ema = None
    last_ts = 0.0
    age_ms = 0
    while True:
        cycle0 = time.time()
        with milk_lock:
            img = latest['frame']
            ts = latest['ts']
        if img is None or ts == last_ts:   # no fresh frame yet
            time.sleep(0.03)
            continue
        last_ts = ts
        age_ms = int((time.time() - ts) * 1000)   # how fresh the frame is
        lb, r = letterbox(img, IMGSZ)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
        net.setInput(blob)
        out = net.forward()
        t2 = time.time()
        dets = parse(out, r, CONF)
        dt = t2 - cycle0
        ema = dt if ema is None else 0.8 * ema + 0.2 * dt
        objs = {}
        for c, s, x1, y1, x2, y2 in dets:
            col = COLORS[c]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(img, f'{NAMES[c]} {s:.2f}', (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            objs[NAMES[c]] = objs.get(NAMES[c], 0) + 1
        hud = f'{shared["stats"]["model"]} @{IMGSZ}  {ema*1000:.0f} ms  frame {age_ms}ms old  {cpu_temp():.0f}C'
        cv2.rectangle(img, (0, 0), (len(hud) * 9 + 12, 24), (0, 0, 0), -1)
        cv2.putText(img, hud, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            shared['jpg'] = buf.tobytes()
            shared['stats'] = {'fps': round(1 / ema, 1), 'infer_ms': round(ema * 1000),
                               'age_ms': age_ms,
                               'objs': [f'{k}×{v}' for k, v in sorted(objs.items())],
                               'temp': cpu_temp(), 'res': f'{img.shape[1]}x{img.shape[0]}',
                               'model': os.path.basename(MODEL), 'imgsz': IMGSZ}
        # throttle: sample at SAMPLE_PERIOD, always picking the freshest frame next round
        rest = SAMPLE_PERIOD - (time.time() - cycle0)
        if rest > 0:
            time.sleep(rest)


PAGE = b"""<!doctype html><html><head><meta charset=utf-8><title>VisionFive 2 - YOLO live</title>
<style>body{background:#111;color:#0f0;font-family:monospace;margin:0;padding:12px}
h1{font-size:16px;color:#8f8}#wrap{display:flex;gap:16px;flex-wrap:wrap}
img{border:2px solid #0a0;max-width:100%;image-rendering:auto}
#stats{font-size:14px;line-height:1.7}.k{color:#6a6}.v{color:#cfc;font-weight:bold}
#objs span{display:inline-block;background:#032;border:1px solid #0a0;padding:2px 8px;margin:2px;border-radius:4px}</style>
</head><body><h1>StarFive VisionFive 2 - YOLO - LAN-only RTSP, zero cloud</h1>
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
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == '/stats':
            with lock:
                s = json.dumps(shared['stats']).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(s)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
            self.end_headers()
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
print(f'UI: http://0.0.0.0:{PORT}  (source {RTSP.split("@")[-1]}, freshest-frame capture)', flush=True)
Server(('0.0.0.0', PORT), H).serve_forever()
