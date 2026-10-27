#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多格式模型性能对比：同一数据、同一输入尺寸、同一阈值、同一台机器。

做法：
  1. 一次性把 N 帧解码到内存（避免读盘抖动影响计时）；
  2. 先预热 --warmup 帧（JIT/内存分配/onnxruntime 图优化都在这一阶段完成）；
  3. 重复 --repeats 轮，每轮对同样 N 帧计时，得到每帧延迟；
  4. 统计均值/中位/P95 与吞吐（FPS），并检查各格式的检测结果与 .pt 是否一致。

输出：results/benchmark.json、results/benchmark.md

示例：
  python benchmark.py --models models/best.pt models/best.onnx models/best.torchscript
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="多格式模型性能对比")
    ap.add_argument("--models", nargs="+",
                    default=["models/best.pt", "models/best.onnx", "models/best.torchscript"])
    ap.add_argument("--source", default="assets/d414.mp4")
    ap.add_argument("--frames", type=int, default=120, help="参与计时/比对的帧数")
    ap.add_argument("--repeats", type=int, default=3, help="重复轮数（取平均）")
    ap.add_argument("--warmup", type=int, default=10, help="预热帧数")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8,
                    help="固定 CPU 线程数，避免线程数漂移导致计时抖动；0 表示不干预")
    ap.add_argument("--rect", action="store_true",
                    help="允许 stride 对齐的最小填充（ultralytics 对 .pt 的默认行为）。"
                         "默认关闭，强制三种后端都吃 imgsz×imgsz 方形输入，才可比")
    ap.add_argument("--out-json", default="results/benchmark.json")
    ap.add_argument("--out-md", default="results/benchmark.md")
    return ap.parse_args()


def load_frames(path: str, count: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")
    frames = []
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("没有读到任何帧")
    return frames


def extract(res):
    """从 Results 里取出框、类别、置信度、关键点（numpy），用于一致性比对。"""
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,)), np.zeros((0, 4, 2))
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 4, 2))
    return boxes, cls, conf, kpts


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) -> (N,M) 两两 IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


MIN_IOU = 0.5          # 低于这个 IoU 视为"没匹配上"，单独统计，不混进误差均值


def greedy_match(rb, ob):
    """按 IoU 从大到小做一对一贪心匹配，返回 (pairs, 未匹配的 ref 数, 未匹配的 other 数)。"""
    M = iou_matrix(rb, ob)
    pairs, used_r, used_o = [], set(), set()
    if M.size:
        for idx in np.argsort(-M, axis=None):
            i, j = np.unravel_index(idx, M.shape)
            if M[i, j] < MIN_IOU:
                break
            if i in used_r or j in used_o:
                continue
            used_r.add(int(i)); used_o.add(int(j))
            pairs.append((int(i), int(j), float(M[i, j])))
    return pairs, len(rb) - len(pairs), len(ob) - len(pairs)


def compare(ref, other):
    """逐帧把 other 与 ref 比对：匹配上的算 IoU/置信度/关键点误差，没匹配上的单独计数。"""
    ious, conf_deltas, kpt_dists = [], [], []
    frames_count_mismatch = frames_unmatched = 0
    unmatched_ref_total = unmatched_other_total = 0
    class_flips = 0
    cls_ref, cls_other = {}, {}
    for r, o in zip(ref, other):
        rb, rc, rconf, rk = r
        ob, oc, oconf, ok_ = o
        for c in rc:
            cls_ref[int(c)] = cls_ref.get(int(c), 0) + 1
        for c in oc:
            cls_other[int(c)] = cls_other.get(int(c), 0) + 1
        if len(rb) != len(ob):
            frames_count_mismatch += 1
        pairs, un_r, un_o = greedy_match(rb, ob)
        unmatched_ref_total += un_r
        unmatched_other_total += un_o
        if un_r or un_o:
            frames_unmatched += 1
        for i, j, iou in pairs:
            ious.append(iou)
            conf_deltas.append(abs(float(rconf[i]) - float(oconf[j])))
            kpt_dists.append(float(np.linalg.norm(rk[i] - ok_[j], axis=1).mean()))
            if int(rc[i]) != int(oc[j]):
                class_flips += 1
    return {
        "detections_ref": int(sum(len(r[0]) for r in ref)),
        "detections_other": int(sum(len(o[0]) for o in other)),
        "class_counts_ref": dict(sorted(cls_ref.items())),
        "class_counts_other": dict(sorted(cls_other.items())),
        "frames_with_count_mismatch": frames_count_mismatch,
        "frames_with_unmatched": frames_unmatched,
        "unmatched_ref_boxes": unmatched_ref_total,
        "unmatched_other_boxes": unmatched_other_total,
        "matched_pairs": len(ious),
        "iou_mean": round(float(np.mean(ious)), 5) if ious else None,
        "iou_min": round(float(np.min(ious)), 5) if ious else None,
        "conf_delta_mean": round(float(np.mean(conf_deltas)), 5) if conf_deltas else None,
        "conf_delta_max": round(float(np.max(conf_deltas)), 5) if conf_deltas else None,
        "class_flips_in_matched": class_flips,
        "kpt_pixel_delta_mean": round(float(np.mean(kpt_dists)), 3) if kpt_dists else None,
        "kpt_pixel_delta_max": round(float(np.max(kpt_dists)), 3) if kpt_dists else None,
    }


def fmt_label(path: Path) -> str:
    name = path.name
    if name.endswith(".pt"):
        return "PyTorch (.pt)"
    if name.endswith(".onnx"):
        return "ONNX (.onnx)"
    if name.endswith(".torchscript"):
        return "TorchScript (.torchscript)"
    if name.endswith(".engine"):
        return "TensorRT (.engine)"
    return path.suffix or name


def pin_threads(n: int):
    """固定 torch 的 CPU 线程数；ORT 通过 OMP_NUM_THREADS 在进程启动时生效。"""
    if n <= 0:
        return None
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    import torch
    torch.set_num_threads(n)
    return torch.get_num_threads()


def main():
    args = parse_args()
    # 注意：OMP_NUM_THREADS 必须在 onnxruntime/ultralytics 初始化会话之前设置
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
    from ultralytics import YOLO

    frames = load_frames(args.source, args.frames)
    print(f"[info] 测试数据：{args.source} 取 {len(frames)} 帧；预热 {args.warmup} 帧；重复 {args.repeats} 轮")
    print(f"[info] 参数 imgsz={args.imgsz} conf={args.conf} iou={args.iou} "
          f"device={args.device} threads={args.threads} rect={args.rect}")

    env = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "device": args.device,
        "rect": args.rect,
        "threads_requested": args.threads,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "frames": len(frames),
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
        env["torch_threads_after_pin"] = pin_threads(args.threads)
    except Exception:
        pass
    try:
        import onnxruntime
        env["onnxruntime"] = onnxruntime.__version__
        env["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception:
        pass
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except Exception:
        pass

    results, reference = [], None
    for model_path in map(Path, args.models):
        if not model_path.is_file():
            print(f"[warn] 跳过（不存在）：{model_path}")
            continue
        label = fmt_label(model_path)
        size_bytes = model_path.stat().st_size
        print(f"\n[info] {label}  {size_bytes:,} 字节")

        t0 = time.perf_counter()
        model = YOLO(str(model_path))
        load_ms = (time.perf_counter() - t0) * 1000
        # ultralytics 加载模型时可能改动线程数，这里再固定一次
        pin_threads(args.threads)
        print(f"  加载 {load_ms:.1f} ms  torch_threads={__import__('torch').get_num_threads()}")

        # 预热
        for f in frames[: args.warmup]:
            model.predict(f, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          device=args.device, rect=args.rect, verbose=False)

        # 重复计时
        round_means, dets_per_frame = [], []
        all_res = []
        for r in range(args.repeats):
            lats = []
            for idx, f in enumerate(frames):
                t1 = time.perf_counter()
                res = model.predict(f, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    device=args.device, rect=args.rect, verbose=False)[0]
                lats.append((time.perf_counter() - t1) * 1000)
                if r == 0:
                    all_res.append(extract(res))
                    dets_per_frame.append(0 if res.boxes is None else len(res.boxes))
            round_means.append(statistics.mean(lats))
            print(f"  第 {r+1} 轮：均值 {round_means[-1]:.2f} ms/帧  "
                  f"中位 {statistics.median(lats):.2f}  P95 {np.percentile(lats, 95):.2f}  "
                  f"吞吐 {1000/round_means[-1]:.2f} FPS")

        all_lat = [m for m in round_means]
        mean_ms = statistics.mean(all_lat)
        entry = {
            "model": str(model_path),
            "format": label,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
            "load_ms": round(load_ms, 1),
            "latency_ms_mean": round(mean_ms, 2),
            "latency_ms_best_round": round(min(all_lat), 2),
            "latency_ms_worst_round": round(max(all_lat), 2),
            "latency_ms_std": round(statistics.pstdev(all_lat), 2) if len(all_lat) > 1 else 0.0,
            "fps": round(1000 / mean_ms, 2),
            "detections_total": int(sum(dets_per_frame)),
            "frames_with_detection": int(sum(1 for d in dets_per_frame if d > 0)),
            "detections_per_frame_mean": round(float(np.mean(dets_per_frame)), 3),
            "round_means_ms": [round(m, 2) for m in all_lat],
        }
        if reference is None:
            reference = all_res
            entry["consistency_vs_pt"] = None
        else:
            entry["consistency_vs_pt"] = compare(reference, all_res)
        results.append(entry)

    # ---- 输出 JSON ----
    payload = {"env": env, "results": results}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 输出 Markdown 表 ----
    lines = [
        "# 多格式性能对比（自动生成）",
        "",
        f"- 测试数据：`{args.source}` 前 {len(frames)} 帧（同一批帧，解码一次后复用）",
        f"- 参数：imgsz={args.imgsz}，conf={args.conf}，iou={args.iou}，device={args.device}",
        f"- 计时：预热 {args.warmup} 帧后重复 {args.repeats} 轮取平均，仅计 `model.predict()` 耗时",
        f"- 环境：{env['os']}，{env['cpu_count']} 线程，Python {env['python']}，"
        f"torch {env.get('torch','?')}，onnxruntime {env.get('onnxruntime','?')}",
        "",
        "| 模型格式 | 文件大小 | 推理耗时（均值） | 吞吐 | 检测目标数量 | 与 .pt 一致性 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in results:
        c = e.get("consistency_vs_pt")
        if c:
            cons = (f"IoU 均值 {c['iou_mean']}；未匹配 {c['unmatched_ref_boxes']}/{c['unmatched_other_boxes']} 个框；"
                    f"类别翻转 {c['class_flips_in_matched']}；关键点差 {c['kpt_pixel_delta_mean']} px")
        else:
            cons = "基准"
        lines.append(
            f"| {e['format']} | {e['size_mb']} MB | {e['latency_ms_mean']} ms/帧 | "
            f"{e['fps']} FPS | {e['detections_total']} | {cons} |")
    lines += [
        "",
        "> 检测目标数量为同一批帧上的累计检出个数。未匹配框 = 在 .pt 结果里找不到 IoU ≥ "
        f"{MIN_IOU} 的对应框（或反之），通常就是低置信度目标在导出后越过阈值造成的。",
        "",
    ]
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== 汇总 =====")
    for e in results:
        print(f"  {e['format']:<26s} {e['size_mb']:>6.2f} MB  "
              f"{e['latency_ms_mean']:>7.2f} ms/帧  {e['fps']:>6.2f} FPS  "
              f"检出 {e['detections_total']}")
    print(f"\n[info] 结果：{out_json} / {out_md}")


if __name__ == "__main__":
    main()
