#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较 ultralytics 对不同后端送进网络的"预处理输入张量"是否一致。

为什么需要这一步：
    上一份工具（compare_raw_output.py）已经证明：把同一张张量分别喂给
    PyTorch / TorchScript / ONNX，输出几乎完全相同（TorchScript 甚至逐元素相等）。
    但端到端 predict() 的检出数却不同（387 vs 378），且在 conf=0.10 时差异依旧存在。
    那么差异只可能来自 ultralytics 框架侧：要么预处理不同，要么后处理不同。
    本脚本把每种模型内部的预处理器抓出来，对比它真正送进网络的那张张量。

用法：
  python tools/compare_preprocess.py --frame 111
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="models/best.pt")
    ap.add_argument("--onnx", default="models/best.onnx")
    ap.add_argument("--torchscript", default="models/best.torchscript")
    ap.add_argument("--source", default="assets/d414.mp4")
    ap.add_argument("--frame", type=int, default=111)
    ap.add_argument("--imgsz", type=int, default=640)
    return ap.parse_args()


def grab_preprocessed(model, frame, imgsz):
    """调用一次 predict 初始化 predictor，再用它自己的 preprocess 取得输入张量。"""
    model.predict(frame, imgsz=imgsz, conf=0.25, iou=0.7, verbose=False)
    pred = model.predictor
    out = pred.preprocess([frame.copy()])
    if isinstance(out, dict):
        t = out["img"]
    elif isinstance(out, (list, tuple)):
        t = out[0]
    else:
        t = out
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到指定帧")
    print(f"[info] 第 {args.frame} 帧 原图 {frame.shape}")

    from ultralytics import YOLO

    tensors = {}
    for tag, path in (("pt", args.pt), ("torchscript", args.torchscript), ("onnx", args.onnx)):
        m = YOLO(path)
        t = grab_preprocessed(m, frame, args.imgsz)
        tensors[tag] = t
        print(f"[info] {tag:12s} 预处理张量 shape={tuple(t.shape)} dtype={t.dtype} "
              f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.6f}")

    ref = tensors["pt"]
    print("\n| 对比 | shape 是否相同 | 最大绝对差 | 平均绝对差 | 完全相同元素占比 |")
    print("| --- | --- | --- | --- | --- |")
    for tag in ("torchscript", "onnx"):
        other = tensors[tag]
        if other.shape != ref.shape:
            print(f"| .pt vs {tag} | 否 {ref.shape} vs {other.shape} | - | - | - |")
            continue
        d = np.abs(ref - other)
        print(f"| .pt vs {tag} | 是 | {d.max():.3e} | {d.mean():.3e} | {float((d == 0).mean()) * 100:.2f}% |")

    # 若张量不同，进一步看差异是否集中在填充区域（letterbox 的灰边）
    for tag in ("torchscript", "onnx"):
        if tensors[tag].shape != ref.shape:
            continue
        diff = np.abs(ref - tensors[tag]).max(axis=(0, 1))     # (H, W)
        rows = np.where(diff.max(axis=1) > 1e-6)[0]
        cols = np.where(diff.max(axis=0) > 1e-6)[0]
        if len(rows):
            print(f"[{tag}] 差异行范围 {rows.min()}~{rows.max()}（共 {len(rows)} 行），"
                  f"列范围 {cols.min()}~{cols.max()}（共 {len(cols)} 列）")
        else:
            print(f"[{tag}] 预处理张量完全相同")


if __name__ == "__main__":
    main()
