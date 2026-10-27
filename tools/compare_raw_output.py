#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比三个后端在"同一张输入张量"下的原始输出，定位差异来源。

输出张量形状 (1, 18, 8400)：
    前 4 行  = 框（cx, cy, w, h）
    第 5-10 行 = 6 个类别的分数
    第 11-18 行 = 4 个关键点的 x, y
若原始输出几乎相同而最终检出数不同 → 差异在后处理（解码/NMS）；
若原始输出本身有可观测差异 → 差异在图计算（融合、算子实现、精度）。

用法：
  python tools/compare_raw_output.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--onnx", default="models/best.onnx")
    ap.add_argument("--torchscript", default="models/best.torchscript")
    ap.add_argument("--source", default="assets/d414.mp4")
    ap.add_argument("--frame", type=int, default=111, help="取第几帧（1 起）")
    ap.add_argument("--imgsz", type=int, default=640)
    return ap.parse_args()


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = (new - nh) // 2, new - nh - (new - nh) // 2
    left, right = (new - nw) // 2, new - nw - (new - nw) // 2
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到指定帧")

    img = letterbox(frame, args.imgsz)
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    print(f"[info] 第 {args.frame} 帧，输入张量 {x.shape}")

    import torch
    from ultralytics import YOLO

    # 1) PyTorch（融合后的模型，和 predict 内部一致）
    net = YOLO(args.pt).model.fuse().eval()
    with torch.no_grad():
        out_pt = net(torch.from_numpy(x))
    out_pt = (out_pt[0] if isinstance(out_pt, (tuple, list)) else out_pt).numpy()

    # 2) TorchScript
    ts = torch.jit.load(args.torchscript, map_location="cpu").eval()
    with torch.no_grad():
        out_ts = ts(torch.from_numpy(x))
    out_ts = (out_ts[0] if isinstance(out_ts, (tuple, list)) else out_ts).numpy()

    # 3) ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out_ox = sess.run(None, {name: x})[0]

    print(f"[info] 输出形状  pt={out_pt.shape}  ts={out_ts.shape}  onnx={out_ox.shape}")
    nc = 6
    seg = {"框 4 行": slice(0, 4), "类别 6 行": slice(4, 4 + nc), "关键点 8 行": slice(4 + nc, 4 + nc + 8)}
    print("\n| 对比 | 区段 | 最大绝对差 | 平均绝对差 | 完全相同的元素占比 |")
    print("| --- | --- | --- | --- | --- |")
    for tag, a, b in (("TorchScript", out_pt, out_ts), ("ONNX", out_pt, out_ox)):
        for seg_name, sl in seg.items():
            d = np.abs(a[:, sl] - b[:, sl])
            same = float((d == 0).mean()) * 100
            print(f"| .pt vs {tag} | {seg_name} | {d.max():.3e} | {d.mean():.3e} | {same:.1f}% |")

    # 类别分数：看"第一名与第二名差距"是否被削弱，这决定低置信度目标是否翻转
    def top2(scores):
        s = scores.max(axis=1)
        idx = scores.argmax(axis=1)
        return s, idx

    print("\n| 后端 | 类别分数最大值 | 达到 conf=0.25 的候选数 | 前两名分数差的均值 |")
    print("| --- | --- | --- | --- |")
    for tag, out in ((".pt", out_pt), ("TorchScript", out_ts), ("ONNX", out_ox)):
        cls_scores = out[0, 4:4 + nc, :]
        mx = cls_scores.max(axis=0)
        n_over = int((mx >= 0.25).sum())
        srt = np.sort(cls_scores, axis=0)
        margin = float((srt[-1] - srt[-2]).mean())
        print(f"| {tag} | {mx.max():.4f} | {n_over} | {margin:.5f} |")


if __name__ == "__main__":
    main()
