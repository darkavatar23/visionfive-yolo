#!/usr/bin/env python3
"""YOLO on an RTSP/HTTP-snapshot camera (all LAN). OpenCV DNN, CPU.
Usage: yolo_cam.py model.onnx [n_frames] [imgsz] [conf]
Frame source: Home Assistant camera_proxy (default) or RTSP when RTSP_URL is set in the env.
"""
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

MODEL = sys.argv[1]
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 416
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35

HA = 'http://HA_IP:8123'
ENTITY = 'camera.robot_braccio_meccanico_live_view'
TOKEN = os.environ.get('HA_TOKEN', '')
RTSP = os.environ.get('RTSP_URL', '')

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

cap = None
if RTSP:
    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)


def get_frame():
    if cap is not None:
        for _ in range(3):  # drop stale buffered frames
            cap.grab()
        ok, img = cap.retrieve()
        return img if ok else None
    req = urllib.request.Request(f'{HA}/api/camera_proxy/{ENTITY}',
                                 headers={'Authorization': f'Bearer {TOKEN}'})
    data = urllib.request.urlopen(req, timeout=15).read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return canvas, r


def parse(out, r, conf):
    """Handles yolov5 (1,N,85), yolov8/11 (1,84,N) and end-to-end (1,N,6)."""
    dets = []
    if out.ndim == 3 and out.shape[2] == 85:            # yolov5
        p = out[0]
        cls_scores = p[:, 5:] * p[:, 4:5]
    elif out.ndim == 3 and out.shape[1] in (84, 85):    # yolov8/11 transposed
        p = out[0].T
        cls_scores = p[:, 4:]
    elif out.ndim == 3 and out.shape[2] == 6:           # end-to-end
        for x1, y1, x2, y2, s, c in out[0]:
            if s > conf:
                dets.append((int(c), float(s), x1 / r, y1 / r, x2 / r, y2 / r))
        return dets
    else:
        raise SystemExit(f'unhandled output shape: {out.shape}')
    cls = cls_scores.argmax(1)
    sc = cls_scores[np.arange(len(cls)), cls]
    m = sc > conf
    boxes, confs, classes = [], [], []
    for (cx, cy, w, h), s, c in zip(p[m, :4], sc[m], cls[m]):
        boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
        confs.append(float(s))
        classes.append(int(c))
    for i in np.array(cv2.dnn.NMSBoxes(boxes, confs, conf, 0.45)).flatten():
        x, y, w, h = boxes[int(i)]
        dets.append((classes[int(i)], confs[int(i)], x / r, y / r, (x + w) / r, (y + h) / r))
    return dets


net = cv2.dnn.readNetFromONNX(MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.setNumThreads(4)

print(f'model={MODEL} imgsz={IMGSZ} frame={N_FRAMES} conf={CONF} '
      f'source={"RTSP" if RTSP else "HA snapshot"}')
times = []
for n in range(N_FRAMES):
    t0 = time.time()
    img = get_frame()
    if img is None:
        print(f'[{n}] missed frame')
        continue
    t1 = time.time()
    lb, r = letterbox(img, IMGSZ)
    blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
    net.setInput(blob)
    out = net.forward()
    t2 = time.time()
    dets = parse(out, r, CONF)
    times.append(t2 - t1)
    lbl = ', '.join(f'{NAMES[c]} {s:.2f}' for c, s, *_ in dets) or 'none'
    print(f'[{n}] fetch {t1 - t0:.2f}s | inference {t2 - t1:.2f}s | {lbl}', flush=True)
    for c, s, x1, y1, x2, y2 in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(img, f'{NAMES[c]} {s:.2f}', (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    cv2.imwrite('last_frame.jpg', img)

if times:
    a = sum(times) / len(times)
    print(f'== average inference {a:.2f}s ({1 / a:.2f} FPS) su {len(times)} frame ==')
