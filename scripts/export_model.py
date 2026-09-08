#!/usr/bin/env python3
"""Export a trained checkpoint to the format the aircraft actually runs.

    # workstation, with the `train` extra installed
    python3 scripts/export_model.py --weights runs/thermal/weights/best.pt \
                                    --out models/yolo11n-thermal-sar.onnx \
                                    --imgsz 640 512 --channels 1

    # prove the quantised model did not lose the small targets
    python3 scripts/export_model.py --validate \
        --reference models/yolo11n-thermal-sar.onnx \
        --candidate models/yolo11n-thermal-sar-int8.onnx \
        --scenario flood --frames 40

    # print the Qualcomm AI Hub command for the RB3 Gen 2
    python3 scripts/export_model.py --qnn-recipe

The validation mode is the one that matters.  It renders real frames from the
simulator's radiometric world model (so the images have the *statistics* of the
deployment domain: small targets, low contrast, thermal clutter), runs both
models through the identical deployment preprocessing, and reports how many of
the reference model's detections the candidate lost.  A quantisation that costs
more than 3% of recall does not fly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

QNN_RECIPE = """
Qualcomm RB3 Gen 2 (QCS6490) / RB5 (QCS8550) export
---------------------------------------------------
  pip install "qai-hub-models[yolov11_det]" qai-hub
  qai-hub configure --api_token <YOUR_TOKEN>

  # INT8 weights and activations, QNN model library for the Hexagon NPU
  python -m qai_hub_models.models.yolov11_det.export \\
      --quantize w8a8 \\
      --target-runtime qnn \\
      --chipset qualcomm-qcs6490-proxy \\
      --output-dir models/

  # if small-object recall regresses, keep activations at 16 bit:
  #   --quantize w8a16

Then:
  1. copy the produced .bin into models/
  2. python3 scripts/fetch_models.py --verify-all
  3. python3 scripts/export_model.py --validate --reference <fp32.onnx> \\
         --candidate <int8.onnx> --frames 60
  4. only then update configs/onboard.yaml to point at the QNN model
"""


def render_validation_frames(scenario: str, n: int, seed: int = 7):
    """Frames with deployment statistics, not COCO statistics.

    Half the frames are aimed at real survivors, half at random ground: a
    quantisation check run only on easy positives measures nothing, and one run
    only on background measures only the false-alarm side.
    """
    from sar.sim.renderer import CameraRenderer
    from sar.sim.scenario import build_reference_scenario
    from sar.vehicle.dynamics import PlantState

    world, lwir_spec, _rgb_spec = build_reference_scenario(scenario, seed=seed, smoke=True)
    cam = CameraRenderer(world, lwir_spec, seed=seed)
    rng = np.random.default_rng(seed)
    victims = list(world.victims)
    frames = []
    for i in range(n):
        alt = float(rng.uniform(35.0, 70.0))
        if victims and i % 2 == 0:
            v = victims[i // 2 % len(victims)]
            north, east = float(v.north), float(v.east)
        else:
            north = float(rng.uniform(0.15, 0.85) * world.north_m)
            east = float(rng.uniform(0.15, 0.85) * world.east_m)
        gz = float(np.nan_to_num(world.terrain_z(north, east)))
        st = PlantState(pos=np.array([north, east, -(gz + alt)]),
                        euler=np.array([0.0, 0.0, float(rng.uniform(-np.pi, np.pi))]))
        frames.append(np.asarray(cam.render(st, t=float(i)).image))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Export / validate SAR detector models")
    ap.add_argument("--weights", help="PyTorch .pt checkpoint to export")
    ap.add_argument("--out", help="destination .onnx path")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[640, 640],
                    metavar=("H", "W"))
    ap.add_argument("--opset", type=int, default=12,
                    help="12 keeps the Qualcomm QNN converter happy (default)")
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--reference", help="float reference model for --validate")
    ap.add_argument("--candidate", help="quantised model for --validate")
    ap.add_argument("--scenario", default="flood")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--json", help="write the validation report here")
    ap.add_argument("--qnn-recipe", action="store_true",
                    help="print the Qualcomm AI Hub export commands and exit")
    args = ap.parse_args()

    if args.qnn_recipe:
        print(QNN_RECIPE)
        return

    if args.validate:
        if not (args.reference and args.candidate):
            ap.error("--validate needs --reference and --candidate")
        from sar.ai.export import validate_onnx_against_reference
        print(f"rendering {args.frames} '{args.scenario}' LWIR frames...")
        frames = render_validation_frames(args.scenario, args.frames)
        report = validate_onnx_against_reference(
            args.candidate, args.reference, frames,
            input_hw=tuple(args.imgsz), channels=args.channels)
        print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
            print(f"report -> {args.json}")
        raise SystemExit(0 if report["verdict"] == "acceptable" else 1)

    if not (args.weights and args.out):
        ap.error("give --weights and --out, or --validate, or --qnn-recipe")

    from sar.ai.export import export_ultralytics_to_onnx
    out = export_ultralytics_to_onnx(args.weights, args.out,
                                     imgsz=tuple(args.imgsz), opset=args.opset)
    print(f"exported -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("next:  python3 scripts/fetch_models.py --verify-all")


if __name__ == "__main__":
    main()
