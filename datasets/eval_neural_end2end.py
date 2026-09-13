"""End-to-end neural detector evaluation against simulator ground truth.

Renders survey frames from the reference scenario and runs the *trained*
ONNX models through the real runtime stack (``build_detector_stack``), then
scores person detections against the renderer's geometric truth.

This is the honest end-to-end number for the sandbox-trained checkpoints:
the same host, the same ONNX engine, the same decode/veto path the aircraft
would run. Run after ``datasets/postprocess.py`` has registered the models.

Run:  .venv/bin/python datasets/eval_neural_end2end.py \
        [--frames 40] [--alt-min 20 --alt-max 40] [--json artifacts/neural_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

from sar.ai.stack import build_detector_stack  # noqa: E402
from sar.sim.renderer import CameraRenderer  # noqa: E402
from sar.sim.scenario import build_reference_scenario  # noqa: E402
from sar.vehicle.dynamics import PlantState  # noqa: E402


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_modality(stack, world, spec, modality, n_frames, alt_min, alt_max, seed):
    cam = CameraRenderer(world, spec, seed=seed)
    rng = np.random.default_rng(seed)
    victims = list(world.victims)
    tp = fp = fn = 0
    lat = []
    for i in range(n_frames):
        if victims and rng.random() < 0.7:
            v = victims[int(rng.integers(0, len(victims)))]
            north = float(v.north + rng.normal(0, 6))
            east = float(v.east + rng.normal(0, 6))
        else:
            north = float(rng.uniform(0.1, 0.9) * world.north_m)
            east = float(rng.uniform(0.1, 0.9) * world.east_m)
        alt = float(rng.uniform(alt_min, alt_max))
        gz = float(np.nan_to_num(world.terrain_z(north, east)))
        st = PlantState(
            pos=np.array([north, east, -(gz + alt)]),
            euler=np.array([rng.normal(0, 0.04), rng.normal(0, 0.04),
                            rng.uniform(-np.pi, np.pi)]))
        frame = cam.render(st, t=float(i))
        t0 = time.perf_counter()
        dets = stack.detect([frame])
        lat.append((time.perf_counter() - t0) * 1000.0)
        dets = [d for d in dets if getattr(d, "modality", None) == modality]
        truths = []
        for t in frame.truth:
            if t.category == "victim" and getattr(t, "visible", True):
                u, v = t.pixel
                w, h = max(t.width_px, 1.0), max(t.height_px, 1.0)
                truths.append((u - w / 2, v - h / 2, u + w / 2, v + h / 2))
        matched = [False] * len(truths)
        for d in dets:
            box = d.bbox
            best_j, best_iou = -1, 0.0
            for j, tb in enumerate(truths):
                if matched[j]:
                    continue
                v = iou(box, tb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.3:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1
        fn += sum(0 if m else 1 for m in matched)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "modality": modality, "frames": n_frames,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "recall": round(recall, 4), "precision": round(precision, 4),
        "f1": round(2 * recall * precision / (recall + precision), 4)
        if (recall + precision) else 0.0,
        "latency_ms": {
            "mean": round(float(np.mean(lat)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--alt-min", type=float, default=20.0)
    ap.add_argument("--alt-max", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--mode", default="neural",
                    choices=("neural", "auto", "hybrid"))
    args = ap.parse_args()

    stack = build_detector_stack(args.mode)
    print("stack:", stack.stack_description)

    world, lwir_spec, rgb_spec = build_reference_scenario(
        "flood", seed=args.seed, smoke=True)
    out = {
        "stack": stack.to_dict(),
        "altitudes_m": [args.alt_min, args.alt_max],
        "frames_per_modality": args.frames,
    }
    for modality, spec in (("lwir", lwir_spec), ("rgb", rgb_spec)):
        res = run_modality(stack, world, spec, modality, args.frames,
                           args.alt_min, args.alt_max, args.seed)
        print(json.dumps(res, indent=2))
        out[modality] = res
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
