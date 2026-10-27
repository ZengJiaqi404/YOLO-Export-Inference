#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用导出后的模型（.onnx / .torchscript）或原始 .pt 做推理与可视化。

同一套代码、同一套绘制逻辑、同一套预处理参数，保证不同格式之间可比。
推理时统计"只算模型推理"的耗时（不含读视频与绘制），并单独统计端到端耗时。

示例：
  python inference.py --model models/best.pt
  python inference.py --model models/best.onnx
  python inference.py --model models/best.torchscript --source assets/d414.mp4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

PALETTE = [
    (0, 255, 0), (255, 144, 30), (0, 215, 255), (255, 0, 255),
    (255, 255, 0), (60, 60, 255), (255, 0, 128), (0, 128, 255),
]

FORMAT_OF_SUFFIX = {
    ".pt": "PyTorch (.pt)",
    ".onnx": "ONNX (.onnx)",
    ".torchscript": "TorchScript (.torchscript)",
    ".engine": "TensorRT (.engine)",
    "_openvino_model": "OpenVINO",
}


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 多格式推理 + 可视化")
    ap.add_argument("--model", default="models/best.onnx", help="模型路径（.pt/.onnx/.torchscript）")
    ap.add_argument("--source", default="assets/d414.mp4", help="输入视频")
    ap.add_argument("--output", default=None, help="结果视频，默认 results/<模型名>_result.mp4")
    ap.add_argument("--report-json", default=None, help="统计报告，默认 results/<模型名>_report.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--device", default="cpu", help="cpu 或 0")
    ap.add_argument("--rect", action="store_true",
                    help="允许 stride 对齐的最小填充（ultralytics 对 .pt 的默认行为）。"
                         "默认关闭：强制补成 imgsz×imgsz 方形输入，保证各格式输入完全一致")
    ap.add_argument("--max-frames", type=int, default=0, help="0 表示整段")
    ap.add_argument("--warmup", type=int, default=5, help="预热帧数（不计入耗时统计）")
    ap.add_argument("--save-txt", action="store_true", help="同时输出逐帧检测明细")
    ap.add_argument("--no-hud", action="store_true")
    return ap.parse_args()


def model_format(path: Path) -> str:
    name = path.name
    for suffix, label in FORMAT_OF_SUFFIX.items():
        if name.endswith(suffix):
            return label
    return path.suffix


def instance_color(i: int):
    return PALETTE[i % len(PALETTE)]


def kpt_pairs(kpts_xy, kpts_conf):
    """关键点整理成 [x, y, conf or None]；本模型无关键点置信度时 conf 记 None。"""
    if not kpts_xy:
        return []
    confs = list(kpts_conf) if kpts_conf else [None] * len(kpts_xy)
    return [[round(float(p[0]), 1), round(float(p[1]), 1),
             None if c is None else round(float(c), 3)] for p, c in zip(kpts_xy, confs)]


def draw_detection(frame, xyxy, cls_id, conf, kpts_xy, kpts_conf, names, color,
                   line_width=2, kpt_radius=5, font_scale=0.6, skeleton="hull"):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

    label = f"{names.get(int(cls_id), str(cls_id))} {conf:.2f}"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_width - 1))
    ty = y1 - 6 if y1 - th - 6 > 0 else y2 + th + 8
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
    cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), max(1, line_width - 1), cv2.LINE_AA)

    pts = []
    for k in range(len(kpts_xy or [])):
        kx, ky = float(kpts_xy[k][0]), float(kpts_xy[k][1])
        pts.append(None if (kx <= 0 and ky <= 0) else (int(round(kx)), int(round(ky))))

    valid = [p for p in pts if p is not None]
    if skeleton == "hull" and len(valid) >= 3:
        hull = cv2.convexHull(np.array(valid, dtype=np.int32))
        cv2.polylines(frame, [hull], True, color, max(1, line_width - 1), cv2.LINE_AA)
    elif skeleton == "order" and len(valid) >= 2:
        for a, b in zip(valid, valid[1:] + valid[:1]):
            cv2.line(frame, a, b, color, max(1, line_width - 1), cv2.LINE_AA)

    for k, p in enumerate(pts):
        if p is None:
            continue
        cv2.circle(frame, p, kpt_radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, kpt_radius, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(k + 1), (p[0] + kpt_radius + 1, p[1] - kpt_radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(k + 1), (p[0] + kpt_radius + 1, p[1] - kpt_radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_hud(frame, lines):
    pad, lh, scale, thick = 8, 22, 0.55, 1
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines) + pad * 2
    height = lh * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def open_writer(path: Path, fps: float, size, codec="mp4v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if writer.isOpened():
        return writer, path
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    return writer, fallback


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"找不到模型：{model_path}")

    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"找不到输入视频：{src}")

    stem = model_path.name.split(".")[0] + "_" + model_path.suffix.lstrip(".")
    out_path = Path(args.output) if args.output else Path("results") / f"{stem}_result.mp4"
    report_path = Path(args.report_json) if args.report_json else Path("results") / f"{stem}_report.json"

    fmt = model_format(model_path)
    size_bytes = model_path.stat().st_size

    from ultralytics import YOLO

    t_load0 = time.perf_counter()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter() - t_load0) * 1000
    names = dict(model.names)
    print(f"[info] 模型 {model_path}（{fmt}，{size_bytes:,} 字节）")
    print(f"[info] 加载耗时 {load_ms:.1f} ms  任务={getattr(model,'task','?')}  类别={names}")
    print(f"[info] 参数 imgsz={args.imgsz} conf={args.conf} iou={args.iou} device={args.device}")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] 输入 {src.name} {width}x{height} {fps:.3f} FPS {total} 帧")

    writer, out_path = open_writer(out_path, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频：{out_path}")
    print(f"[info] 结果视频 -> {out_path}")

    limit = args.max_frames if args.max_frames > 0 else (total or 10 ** 9)
    lat_ms, det_total, frame_idx, det_frames, kpt_total = [], 0, 0, 0, 0
    conf_sum = 0.0
    cls_counts: dict[str, int] = {}
    txt_lines = []
    t_all0 = time.perf_counter()

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            max_det=args.max_det, device=args.device, rect=args.rect,
                            verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        if frame_idx > args.warmup:
            lat_ms.append(dt)

        n = 0 if res.boxes is None else len(res.boxes)
        if n:
            det_frames += 1
        for i in range(n):
            xyxy = res.boxes.xyxy[i].tolist()
            clsid = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            kxy = res.keypoints.xy[i].tolist() if res.keypoints is not None else None
            kcf = (res.keypoints.conf[i].tolist()
                   if (res.keypoints is not None and res.keypoints.conf is not None) else None)
            det_total += 1
            conf_sum += conf
            kpt_total += len(kxy or [])
            cname = names.get(clsid, str(clsid))
            cls_counts[cname] = cls_counts.get(cname, 0) + 1
            draw_detection(frame, xyxy, clsid, conf, kxy, kcf, names, instance_color(i))
            if args.save_txt:
                txt_lines.append(f"{frame_idx} {cname} {conf:.4f} "
                                 f"box={[round(v, 1) for v in xyxy]} kpts={kpt_pairs(kxy, kcf)}")

        if not args.no_hud:
            draw_hud(frame, [f"{model_path.name}  frame {frame_idx}  det {n}",
                             f"imgsz {args.imgsz}  conf {args.conf}",
                             f"infer {dt:.1f} ms"])
        writer.write(frame)
        if frame_idx % 60 == 0:
            print(f"  ... {frame_idx} 帧  累计检测 {det_total}")

    total_sec = time.perf_counter() - t_all0
    cap.release()
    writer.release()

    if args.save_txt:
        txt_path = report_path.parent / f"{stem}_detections.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        print(f"[info] 检测明细 -> {txt_path}")

    def pct(vals, q):
        return round(float(np.percentile(vals, q)), 2) if vals else None

    report = {
        "model": str(model_path),
        "format": fmt,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 2),
        "load_ms": round(load_ms, 1),
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "device": args.device,
        "source": str(src),
        "source_frames": total,
        "frames_processed": frame_idx,
        "frames_with_detection": det_frames,
        "detections_total": det_total,
        "keypoints_total": kpt_total,
        "keypoints_per_target": round(kpt_total / det_total, 2) if det_total else None,
        "mean_box_conf": round(conf_sum / det_total, 4) if det_total else None,
        "class_counts": dict(sorted(cls_counts.items())),
        "latency_ms_mean": round(float(statistics.mean(lat_ms)), 2) if lat_ms else None,
        "latency_ms_median": round(float(statistics.median(lat_ms)), 2) if lat_ms else None,
        "latency_ms_p95": pct(lat_ms, 95),
        "latency_ms_min": round(float(min(lat_ms)), 2) if lat_ms else None,
        "latency_ms_max": round(float(max(lat_ms)), 2) if lat_ms else None,
        "infer_fps": round(1000 / statistics.mean(lat_ms), 2) if lat_ms else None,
        "wall_sec": round(total_sec, 2),
        "output_video": str(out_path),
        "output_mb": round(Path(out_path).stat().st_size / 1024 / 1024, 2) if Path(out_path).is_file() else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 运行结果 =====")
    for k in ("format", "size_mb", "load_ms", "frames_processed", "frames_with_detection",
              "detections_total", "keypoints_total", "mean_box_conf", "class_counts",
              "latency_ms_mean", "latency_ms_median", "latency_ms_p95", "infer_fps",
              "wall_sec", "output_video"):
        print(f"  {k}: {report[k]}")
    print(f"[info] 报告 -> {report_path}")


if __name__ == "__main__":
    main()
