"""One-shot CPU training driver (sandbox): train thermal + RGB detectors on the
simulator-rendered datasets, export ONNX, and record the measured metrics.

Uses the **P2 stride-4 head** (yolov8n-p2.yaml) — the same reason the
deployment thermal spec adds a P2 head: aerial survivors are 8-32 px targets
that a stride-8 head cannot resolve.  The synthetic sets are rendered at
15-45 m (near-pass / approach altitude) so targets are learnable on a 2-core
CPU at imgsz 256; the survey-altitude (35-70 m, 4-10 px) fine-tune needs the
640x512 GPU run in docs/colab/ (or scripts/train_detector.py --train --p2).

Run:  .venv/bin/python datasets/train_all.py
Outputs:
    runs/sar-thermal/weights/best.pt   -> models/yolov8n-thermal-sim.onnx
    runs/sar-rgb/weights/best.pt       -> models/yolov8n-rgb-sim.onnx
    datasets/train_results.json        -> measured val metrics
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch  # noqa: E402

torch.set_num_threads(2)

from ultralytics import YOLO  # noqa: E402

COMMON = dict(
    device="cpu", workers=0, imgsz=256, batch=4,
    scale=0.5, mosaic=1.0, close_mosaic=10, degrees=180.0,
    fliplr=0.5, flipud=0.5, translate=0.2, erasing=0.2,
    patience=30, cos_lr=True, plots=False,
)


def train_one(name: str, data_yaml: Path, epochs: int, channels: int) -> dict:
    t0 = time.time()
    model = YOLO("yolov8n-p2.yaml")
    overrides = dict(COMMON)
    overrides.update(project="runs", name=name, epochs=epochs, data=str(data_yaml))
    if channels == 1:
        overrides.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.4)
    else:
        overrides.update(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
    model.train(**overrides)
    # ultralytics auto-increments the run dir (sar-thermal -> sar-thermal3 ...)
    # when an old dir exists, so read the real save dir instead of assuming one.
    save_dir = Path(getattr(model.trainer, "save_dir", None)
                    or (ROOT / "runs" / name))
    best = save_dir / "weights" / "best.pt"
    metrics = model.val(data=str(data_yaml), imgsz=256, batch=8,
                        device="cpu", workers=0, plots=False, verbose=False)
    out = {
        "name": name,
        "model": "yolov8n-p2",
        "epochs": epochs,
        "imgsz": 256,
        "wall_s": round(time.time() - t0, 1),
        "best_pt": str(best),
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def export_onnx(best_pt: Path, out_onnx: Path, imgsz: int = 256) -> Path:
    model = YOLO(str(best_pt))
    produced = model.export(format="onnx", imgsz=imgsz, opset=12,
                            simplify=True, dynamic=False)
    produced = Path(produced)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out_onnx.resolve():
        shutil.copy(produced, out_onnx)
    print(f"exported -> {out_onnx} ({out_onnx.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return out_onnx


def main() -> int:
    results = {}
    # 1. thermal (LWIR) detector
    results["thermal"] = train_one(
        "sar-thermal", ROOT / "datasets/sim-thermal/data.yaml",
        epochs=30, channels=1)
    export_onnx(Path(results["thermal"]["best_pt"]),
                ROOT / "models/yolov8n-thermal-sim.onnx")
    # 2. RGB (phone camera) detector
    results["rgb"] = train_one(
        "sar-rgb", ROOT / "datasets/sim-rgb/data.yaml",
        epochs=30, channels=3)
    export_onnx(Path(results["rgb"]["best_pt"]),
                ROOT / "models/yolov8n-rgb-sim.onnx")

    out = ROOT / "datasets/train_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
