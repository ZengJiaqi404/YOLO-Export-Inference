#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位"为什么导出后反而更慢"：把一次推理拆成两段分别计时。

  段 A —— 纯模型前向（同一张预处理好的 640x640 输入张量）
  段 B —— ultralytics 的 model.predict()（含 letterbox 预处理 + 后处理/NMS）

两者之差即框架侧的开销。三个后端用同一张输入张量、同一台机器、同样的迭代次数。

用法：
  python tools/forward_bench.py --frames 30
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--onnx", default="models/best.onnx")
    ap.add_argument("--torchscript", default="models/best.torchscript")
    ap.add_argument("--source", default="assets/d414.mp4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    return ap.parse_args()


def letterbox(im, new=640, color=(114, 114, 114)):
    """与 ultralytics 一致的 letterbox：等比缩放 + 居中灰边填充。"""
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = (new - nh) // 2, new - nh - (new - nh) // 2
    left, right = (new - nw) // 2, new - nw - (new - nw) // 2
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def to_tensor(img_bgr):
    """BGR HWC uint8 -> NCHW float32 RGB /255。"""
    x = img_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    lats = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1000)
    return {"mean": round(statistics.mean(lats), 2),
            "median": round(statistics.median(lats), 2),
            "p95": round(float(np.percentile(lats, 95)), 2)}


def main():
    args = parse_args()
    print(f"[env] {platform.system()} {platform.release()}  cpu={platform.processor() or '?'}  "
          f"cpu_count={os.cpu_count()}")
    print(f"[env] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r}  "
          f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')!r}")

    import torch
    print(f"[env] torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"interop={torch.get_num_interop_threads()}")

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到视频首帧")
    x_np = to_tensor(letterbox(frame, args.imgsz))
    x_pt = torch.from_numpy(x_np)
    print(f"[env] 输入张量 {tuple(x_np.shape)} {x_np.dtype}  原图 {frame.shape}")

    rows = []
    from ultralytics import YOLO
    print(f"[env] 导入 ultralytics 后 torch threads={torch.get_num_threads()} "
          f"（ultralytics 会把 CPU 线程数调到 1，这是它自己的默认行为）")

    # ---------- PyTorch ----------
    m_pt = YOLO(args.pt)
    # predict() 内部用的是融合了 Conv+BN 的模型，纯前向计时也要用同一份，才可比
    net_pt = m_pt.model.fuse().eval()
    with torch.no_grad():
        net_pt(x_pt)
    with torch.no_grad():
        fwd = bench(lambda: net_pt(x_pt), args.iters, args.warmup)
    pre = bench(lambda: m_pt.predict(frame, imgsz=args.imgsz, conf=0.25, iou=0.7,
                                     verbose=False), args.iters, args.warmup)
    rows.append(("PyTorch (.pt)", fwd, pre))

    # ---------- TorchScript ----------
    ts = torch.jit.load(args.torchscript, map_location="cpu").eval()
    with torch.no_grad():
        ts(x_pt)
        fwd = bench(lambda: ts(x_pt), args.iters, args.warmup)
    m_ts = YOLO(args.torchscript)
    pre = bench(lambda: m_ts.predict(frame, imgsz=args.imgsz, conf=0.25, iou=0.7,
                                     verbose=False), args.iters, args.warmup)
    rows.append(("TorchScript", fwd, pre))

    # ---------- ONNX Runtime ----------
    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    sess.run(None, {inp: x_np})
    fwd = bench(lambda: sess.run(None, {inp: x_np}), args.iters, args.warmup)
    m_ox = YOLO(args.onnx)
    pre = bench(lambda: m_ox.predict(frame, imgsz=args.imgsz, conf=0.25, iou=0.7,
                                     verbose=False), args.iters, args.warmup)
    rows.append(("ONNX Runtime", fwd, pre))
    print(f"[env] onnxruntime {ort.__version__}  providers={ort.get_available_providers()}")
    print(f"[env] ORT 图优化级别 = {so.graph_optimization_level}  "
          f"intra_op_threads = {so.intra_op_num_threads or '默认(0=全部核)'}")

    print("\n| 后端 | 纯前向 均值 | 纯前向 P95 | predict 均值 | predict P95 | 框架侧开销 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, f, p in rows:
        print(f"| {name} | {f['mean']} ms | {f['p95']} ms | {p['mean']} ms | {p['p95']} ms | "
              f"{round(p['mean'] - f['mean'], 2)} ms |")


if __name__ == "__main__":
    main()
