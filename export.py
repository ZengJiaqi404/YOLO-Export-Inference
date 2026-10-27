#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 PyTorch 格式的 YOLO 权重（.pt）导出为其他格式。

支持 ultralytics 的全部导出格式，本项目重点使用：
  - onnx        ：通用模型交换格式（onnxruntime 推理）
  - torchscript ：PyTorch 自己的部署格式（libtorch 推理）

示例：
  python export.py --weights models/best.pt --formats onnx torchscript
  python export.py --weights models/best.pt --formats onnx --imgsz 640 --opset 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

FORMAT_SUFFIX = {
    "onnx": ".onnx",
    "torchscript": ".torchscript",
    "engine": ".engine",
    "openvino_openvino_model": "",
    "openvino": "",
}


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 模型格式导出")
    ap.add_argument("--weights", default="models/best.pt", help="原始 .pt 权重")
    ap.add_argument("--formats", nargs="+", default=["onnx", "torchscript"],
                    help="要导出的格式，可多个")
    ap.add_argument("--imgsz", type=int, default=640, help="导出时的输入尺寸（需与推理时一致）")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset 版本")
    ap.add_argument("--simplify", action="store_true", default=True, help="ONNX 图简化（onnxslim）")
    ap.add_argument("--dynamic", action="store_true", help="ONNX 动态 batch/尺寸")
    ap.add_argument("--half", action="store_true", help="半精度（仅 GPU 有意义）")
    ap.add_argument("--report-json", default="results/export_report.json")
    return ap.parse_args()


def describe_onnx(path: Path) -> dict:
    """读取 ONNX 图的规模与算子集版本，用于说明"导出后到底变了什么"。"""
    import onnx

    m = onnx.load(str(path))
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    return {
        "onnx_nodes": len(m.graph.node),
        "onnx_initializers": len(m.graph.initializer),
        "onnx_opset": opsets,
        "onnx_inputs": [f"{i.name}{[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}"
                        for i in m.graph.input],
        "onnx_outputs": [o.name for o in m.graph.output],
    }


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"找不到权重：{weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    report = {
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "imgsz": args.imgsz,
        "task": getattr(model, "task", None),
        "names": {int(k): v for k, v in dict(model.names).items()},
        "kpt_shape": list(getattr(model.model, "kpt_shape", []) or []),
        "num_params": sum(p.numel() for p in model.model.parameters()),
        "exports": [],
    }
    print(f"[info] 源权重 {weights}  {weights.stat().st_size:,} 字节")
    print(f"[info] 任务={report['task']} 类别={report['names']} 关键点={report['kpt_shape']}")
    print(f"[info] 参数量={report['num_params']:,}")

    for fmt in args.formats:
        print(f"\n[info] 开始导出 format={fmt} imgsz={args.imgsz}")
        kwargs = {"format": fmt, "imgsz": args.imgsz}
        if fmt == "onnx":
            kwargs.update({"opset": args.opset, "simplify": args.simplify,
                           "dynamic": args.dynamic, "half": args.half})
        elif fmt in ("torchscript", "engine"):
            kwargs.update({"half": args.half})
        t0 = time.perf_counter()
        out = model.export(**kwargs)
        elapsed = time.perf_counter() - t0

        out_path = Path(out)
        if not out_path.exists():                      # 某些格式返回目录
            out_path = Path(out)
        size = out_path.stat().st_size if out_path.is_file() else sum(
            f.stat().st_size for f in out_path.rglob("*") if f.is_file())

        entry = {
            "format": fmt,
            "path": str(out_path),
            "bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
            "export_sec": round(elapsed, 2),
            "ratio_vs_pt": round(size / weights.stat().st_size, 2),
        }
        if fmt == "onnx":
            try:
                entry.update(describe_onnx(out_path))
            except Exception as e:                     # 描述失败不影响导出结果
                entry["describe_error"] = str(e)
        report["exports"].append(entry)
        print(f"[ok] {fmt} -> {out_path}  {size:,} 字节（{entry['size_mb']} MB，"
              f"为 .pt 的 {entry['ratio_vs_pt']}×）耗时 {elapsed:.2f}s")

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[info] 导出报告：{args.report_json}")


if __name__ == "__main__":
    main()
