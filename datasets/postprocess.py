"""Post-training registration + smoke test for the sandbox CPU training run.

After `datasets/train_all.py` finishes, this script:

1. reads `datasets/train_results.json` (measured val metrics),
2. registers the two trained ONNX models in `models/registry.json` so
   `SAR_DETECTOR=auto|neural` picks them up via `best_for()`,
3. records their sha256 + file sizes,
4. smoke-tests each model through the real `OnnxDetectorEngine` (the same
   engine the aircraft runtime uses), reporting measured inference latency
   on this host.

Run:  .venv/bin/python datasets/postprocess.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

from sar.ai.export import write_registry_entry  # noqa: E402
from sar.ai.registry import ModelRegistry, default_registry  # noqa: E402

CLASSES = ["person", "person_group", "vehicle", "animal", "fire", "structure"]
CLASS_MAP = {c: c for c in CLASSES}

MODELS = [
    {
        "name": "yolov8n-thermal-sim",
        "task": "person_detection",
        "modality": "lwir",
        "architecture": "YOLOv8n-P2 (sim-rendered LWIR fine-tune)",
        "input_size": [256, 256],
        "classes": CLASSES,
        "class_map": CLASS_MAP,
        "file_name": "yolov8n-thermal-sim.onnx",
        "fmt": "onnx",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "normalise": "0-1",
        "channels": 3,   # LWIR frames are grayscale; ultralytics feeds a 3-ch stem
        "pretrained": False,
        "source": "train:datasets/train_all.py (sandbox CPU)",
        "trained_on": ["sim-thermal (800 frames / 625 boxes)"],
        "latency_ms": {},
        "notes": "Sandbox CPU fine-tune on simulator-rendered LWIR (15-45 m near-pass). "
                 "Proof-of-pipeline; fly the GPU-trained yolo11n-thermal-sar (640x512, P2) instead.",
    },
    {
        "name": "yolov8n-rgb-sim",
        "task": "person_detection",
        "modality": "rgb",
        "architecture": "YOLOv8n-P2 (sim-rendered RGB fine-tune)",
        "input_size": [256, 256],
        "classes": CLASSES,
        "class_map": CLASS_MAP,
        "file_name": "yolov8n-rgb-sim.onnx",
        "fmt": "onnx",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "normalise": "0-1",
        "channels": 3,
        "pretrained": False,
        "source": "train:datasets/train_all.py (sandbox CPU)",
        "trained_on": ["sim-rgb (500 frames / 400 boxes)"],
        "latency_ms": {},
        "notes": "Sandbox CPU fine-tune for the phone RGB camera path. Proof-of-pipeline; "
                 "fly the GPU-trained yolo11n-rgb / phone TFLite INT8 instead.",
    },
]


def register() -> None:
    reg = default_registry()
    for entry in MODELS:
        write_registry_entry(reg.models_dir, entry)
    # Drop the CI plumbing model so the trained checkpoints are what best_for()
    # actually selects (same ONNX format tier -> stable order favours it).
    reg_json = reg.models_dir / "registry.json"
    if reg_json.is_file():
        data = json.loads(reg_json.read_text())
        data["models"] = [m for m in data.get("models", [])
                          if m.get("name") != "synthetic-lwir"]
        reg_json.write_text(json.dumps(data, indent=2))
    (reg.models_dir / "synthetic-lwir.onnx").unlink(missing_ok=True)


def smoke(reg: ModelRegistry, name: str, channels: int) -> dict:
    from sar.ai.runtime import OnnxDetectorEngine

    path = reg.path(name)
    engine = OnnxDetectorEngine(
        str(path), input_hw=(256, 256), conf=0.25, iou=0.45,
        channels=channels, num_classes=len(CLASSES), warmup=3, budget_ms=125.0,
    )
    img = np.random.default_rng(0).integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8)
    if channels == 1:
        img = img[..., 0]
    lat = []
    for _ in range(10):
        t0 = time.perf_counter()
        boxes, scores, cls = engine.infer(img)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        "model": name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "sha256": reg.sha256(name),
        "provider": engine.active_providers[0],
        "detections": int(len(boxes)),
        "latency_ms": {
            "mean": round(float(np.mean(lat)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
        },
    }


def main() -> int:
    results_path = ROOT / "datasets/train_results.json"
    if not results_path.is_file():
        print("train_results.json not found - run datasets/train_all.py first",
              file=sys.stderr)
        return 2
    train_results = json.loads(results_path.read_text())

    register()
    # Build a FRESH registry: default_registry() caches the pre-register zoo.
    reg = ModelRegistry(default_registry().models_dir)

    report = {
        "train_results": train_results,
        "registered": MODELS,
        "smoke": [
            smoke(reg, "yolov8n-thermal-sim", 1),
            smoke(reg, "yolov8n-rgb-sim", 3),
        ],
    }
    out = ROOT / "datasets/postprocess.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\npostprocess report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
