#!/usr/bin/env python3
"""Train the thermal/RGB survivor detector - including on data we can generate.

Three modes, in the order a team actually needs them:

**1. Synthesise a labelled dataset from the simulator.**::

    python3 scripts/train_detector.py --synthesize 4000 --scenario mixed \
                                      --out datasets/sim-thermal

The renderer is an energy-conserving radiometric model, so a survivor's
apparent temperature, sub-pixel area collapse and thermal contrast against the
ground are physically derived rather than pasted in.  That makes the synthetic
set genuinely useful for the hard part - *small, low-contrast targets* - which
is exactly the regime where public datasets are thinnest.  It is not a
replacement for real flights; it is what lets the network reach a sane starting
point before any are available, and what lets the training pipeline be tested
end to end today.

**2. Write dataset YAMLs for the public sets.**::

    python3 scripts/train_detector.py --dataset-yaml hit-uav --out datasets/

**3. Train.**::

    # thermal (LWIR) - the deployment detector
    python3 scripts/train_detector.py --train --data datasets/sim-thermal/data.yaml \
        --model yolo11n.pt --imgsz 640 --epochs 120 --p2

    # RGB (phone camera)
    python3 scripts/train_detector.py --train --data datasets/sard/data.yaml \
        --model yolo11n.pt --imgsz 640 --epochs 120

``--p2`` adds the stride-4 detection head.  For aerial SAR this is not a tuning
knob, it is the difference between a model that can represent an 8-pixel person
and one whose smallest anchor is already bigger than the target.  Upstream
Ultralytics does not ship a P2 variant of YOLO11, so the YAML is bundled at
``configs/models/yolo11n-p2.yaml`` and resolved automatically here.

Two flags that are easy to get wrong, and cost a whole GPU run each:

* ``--model`` may be a ``.pt`` (fine-tune COCO weights) or a ``.yaml`` (build
  the architecture and **train from scratch**).  On a 2k-image SAR dataset the
  difference is the whole result.  ``--p2`` needs an architecture, so passing
  ``yolo11n.pt --p2`` builds the P2 graph *and* keeps the ``.pt`` as the
  transfer source - see ``--load``.
* ``--channels`` selects the **augmentation regime**, not the tensor's channel
  count (the network is 3-channel either way; Ultralytics replicates a grayscale
  LWIR frame).  ``1`` disables hue/saturation jitter, which do not exist in
  LWIR; ``3`` enables it.  Omit it and the regime is inferred from the dataset,
  and the choice is printed.

Datasets worth the download (see docs/06_AI_MODELS_AND_DATASETS.md):
  HIT-UAV, AIResQ, SARD, HERIDAL, TinyPerson, SeaDronesSee, RGBTDronePerson,
  VisDrone (RGB aerial), RescueNet / FloodNet / xBD (hazard and damage).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

#: Class order used by the thermal SAR model.  Must match
#: ``sar/ai/registry.py::_THERMAL_SAR_CLASSES`` or the exported model will
#: decode to the wrong labels - a failure that looks like a bad detector.
CLASSES: Tuple[str, ...] = ("person", "person_group", "vehicle", "animal",
                            "fire", "structure")

TRUTH_TO_CLASS = {
    "person": "person", "person_group": "person_group", "victim": "person",
    "vehicle": "vehicle", "animal": "animal", "fire": "fire",
    "structure": "structure", "hot_rock": "structure",
}

PUBLIC_DATASETS: Dict[str, Dict[str, Any]] = {
    "hit-uav": {
        "url": "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset",
        "modality": "lwir", "images": 2898,
        "notes": "Aerial LWIR, 60-130 m AGL, person/car/bicycle. The closest "
                 "public match to our sensor geometry.",
    },
    "airesq": {
        "url": "https://www.nature.com/articles/s41597-026-07663-9",
        "modality": "lwir", "images": 0,
        "notes": "High-resolution airborne thermal SAR set; YOLOv8/YOLO11 "
                 "trained on it reach mAP50 ~0.55 on the paper's benchmark - "
                 "the best published aerial-thermal number we have found.",
    },
    "sard": {
        "url": "https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard",
        "modality": "rgb", "images": 1981,
        "notes": "Search-and-rescue postures (lying, sitting, crouching) in "
                 "non-urban terrain. Posture diversity is what it adds.",
    },
    "heridal": {
        "url": "http://ipsar.fesb.unist.hr/HERIDAL%20database.html",
        "modality": "rgb", "images": 68750,
        "notes": "Wilderness aerial RGB, tiny persons. Best published mAP 95.11%.",
    },
    "visdrone": {
        "url": "https://github.com/VisDrone/VisDrone-Dataset",
        "modality": "rgb", "images": 10209,
        "notes": "Aerial RGB with a genuine small-object distribution.",
    },
    "rescuenet": {
        "url": "https://www.classic.grss-ieee.org/community/technical-committees/rescuenet/",
        "modality": "rgb", "images": 4494,
        "notes": "Post-hurricane damage segmentation; the hazard-model source.",
    },
}


# --------------------------------------------------------------------------- #
# 1. synthetic dataset
# --------------------------------------------------------------------------- #
def synthesize(n_images: int, out_dir: Path, scenarios: List[str], *,
               modality: str = "lwir", seed: int = 11,
               alt_range: Tuple[float, float] = (30.0, 90.0),
               val_fraction: float = 0.15) -> Path:
    """Render labelled frames from the simulator into YOLO format."""
    from sar.sim.renderer import CameraRenderer
    from sar.sim.scenario import build_reference_scenario
    from sar.vehicle.dynamics import PlantState

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pillow is required to write the dataset: "
                         "pip install pillow") from exc

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    worlds = []
    for name in scenarios:
        world, lwir_spec, rgb_spec = build_reference_scenario(
            name, seed=int(rng.integers(0, 10_000)), smoke=True)
        spec = lwir_spec if modality == "lwir" else rgb_spec
        worlds.append((name, world, spec, CameraRenderer(world, spec, seed=seed)))

    n_written = 0
    stats = {"frames": 0, "boxes": 0, "empty_frames": 0,
             "by_class": {c: 0 for c in CLASSES},
             "box_px_histogram": {"<8": 0, "8-16": 0, "16-32": 0, ">32": 0}}
    while n_written < n_images:
        name, world, spec, cam = worlds[n_written % len(worlds)]
        alt = float(rng.uniform(*alt_range))
        # 60% of frames aimed near a survivor: an aerial SAR dataset that is
        # 99% empty ground trains a model that predicts "nothing" very well.
        if world.victims and rng.random() < 0.6:
            v = world.victims[int(rng.integers(0, len(world.victims)))]
            jitter = alt * 0.35
            north = float(v.north + rng.normal(0, jitter))
            east = float(v.east + rng.normal(0, jitter))
        else:
            north = float(rng.uniform(0.1, 0.9) * world.north_m)
            east = float(rng.uniform(0.1, 0.9) * world.east_m)
        gz = float(np.nan_to_num(world.terrain_z(north, east)))
        st = PlantState(
            pos=np.array([north, east, -(gz + alt)]),
            euler=np.array([float(rng.normal(0, 0.05)), float(rng.normal(0, 0.05)),
                            float(rng.uniform(-math.pi, math.pi))]))
        frame = cam.render(st, t=float(n_written))
        img = np.asarray(frame.image, dtype=np.float32)

        if modality == "lwir":
            from sar.ai.runtime import preprocess_lwir
            img8 = preprocess_lwir(img, mode="anchored").astype(np.uint8)
            pil = Image.fromarray(img8, mode="L")
        else:
            pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        h, w = img.shape[:2]
        lines: List[str] = []
        for t in frame.truth:
            if not getattr(t, "visible", True):
                continue
            cls_name = TRUTH_TO_CLASS.get(t.label) or TRUTH_TO_CLASS.get(t.category)
            if cls_name is None or cls_name not in CLASSES:
                continue
            cx, cy = t.pixel
            bw, bh = max(float(t.width_px), 2.0), max(float(t.height_px), 2.0)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            lines.append(f"{CLASSES.index(cls_name)} {cx / w:.6f} {cy / h:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
            stats["boxes"] += 1
            stats["by_class"][cls_name] += 1
            long_px = max(bw, bh)
            key = ("<8" if long_px < 8 else "8-16" if long_px < 16
                   else "16-32" if long_px < 32 else ">32")
            stats["box_px_histogram"][key] += 1

        split = "val" if (n_written % int(1 / max(val_fraction, 1e-6))) == 0 else "train"
        stem = f"{name}_{n_written:06d}"
        pil.save(out_dir / "images" / split / f"{stem}.png")
        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        if not lines:
            stats["empty_frames"] += 1
        stats["frames"] += 1
        n_written += 1

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"# Generated by scripts/train_detector.py --synthesize\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {list(CLASSES)}\n")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\ndataset -> {out_dir}\nyaml    -> {yaml_path}")
    print("\nNote the box_px_histogram: if almost everything is >32 px you have "
          "generated an easy dataset and the model will not transfer to survey "
          "altitude. Raise --alt-max.")
    return yaml_path


# --------------------------------------------------------------------------- #
# 2. public dataset descriptors
# --------------------------------------------------------------------------- #
def write_dataset_yaml(name: str, out_dir: Path) -> Path:
    if name not in PUBLIC_DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(PUBLIC_DATASETS)}")
    meta = PUBLIC_DATASETS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    path.write_text(
        f"# {name}: {meta['notes']}\n"
        f"# source: {meta['url']}\n"
        f"# Download and convert to YOLO format under {out_dir / name}\n"
        f"path: {(out_dir / name).resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: {list(CLASSES)}\n")
    print(f"wrote {path}\n  {meta['notes']}\n  {meta['url']}")
    return path


# --------------------------------------------------------------------------- #
# 3. training
# --------------------------------------------------------------------------- #
def infer_channels(data_yaml: Optional[str], explicit: Optional[int]) -> Tuple[int, str]:
    """Pick the augmentation regime: 1 = LWIR, 3 = RGB.  Return (channels, why).

    ``--channels`` used to default to ``1``, which silently trained RGB datasets
    with the LWIR augmentation - ``hsv_h=0, hsv_s=0``, i.e. no colour jitter at
    all.  Nothing in the log says so, and the symptom is a model that is fragile
    to time of day, which reads as "needs more epochs".  So the default is now
    *inferred* and the decision is printed.

    Inference order: an explicit flag wins; then this repo's own dataset table
    (authoritative - ``hit-uav`` has no thermal-looking substring); then path
    hints; then RGB with a loud warning.
    """
    if explicit is not None:
        return int(explicit), "explicit --channels"

    path = Path(data_yaml or "")
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)

    for key, meta in PUBLIC_DATASETS.items():
        if any(key in p for p in parts):
            ch = 1 if meta.get("modality") == "lwir" else 3
            return ch, f"{key!r} is a {meta.get('modality')} dataset"

    if any(h in joined for h in ("lwir", "thermal", "infrared")):
        return 1, "dataset path says thermal"
    if any(h in joined for h in ("rgb", "visible", "colour", "color")):
        return 3, "dataset path says RGB"

    return 3, ("UNRECOGNISED dataset - defaulting to RGB augmentation; "
               "pass --channels 1 if this is LWIR")


def p2_architecture(model: str) -> str:
    """``yolo11n.pt`` / ``yolo11n.yaml`` -> ``yolo11n-p2.yaml``.

    ``--p2`` is an architecture change, so it cannot be had from a stock ``.pt``:
    a stride-4 head has no weights in it.  The graph is built from the bundled
    P2 YAML and whatever COCO weights do fit are transferred across (``--load``).
    """
    stem = Path(model).stem
    return stem + ".yaml" if stem.endswith("-p2") else f"{stem}-p2.yaml"


def train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed (workstation only):\n"
              "  pip install -e '.[train]'")
        return 2
    model = YOLO(args.model)
    if args.load:
        # Without this a .yaml trains from random weights.  Ultralytics only
        # transfers weights that match by name and shape, so a P2 graph keeps the
        # whole COCO backbone+neck and starts its new stride-4 head fresh -
        # which is exactly what we want.
        src = resolve_model(args.load)
        model = model.load(src)
        print(f"transferring pretrained weights from {src}")
    elif Path(args.model).suffix == ".yaml":
        print(f"NOTE: {args.model} is an architecture file - training FROM "
              f"SCRATCH (random weights).  Pass --model yolo11n.pt to fine-tune, "
              f"or --load yolo11n.pt to transfer weights into a P2 graph.")
    # LWIR frames are single-band; ultralytics loads them as 3-channel
    # (grayscale replicated), so the stem stays 3-channel and only the colour
    # augmentation is disabled - hue/saturation do not exist in LWIR.
    overrides: Dict[str, Any] = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        # Aerial-thermal specifics, each for a reason:
        scale=0.5,        # scale jitter stands in for altitude variation
        mosaic=1.0,       # more small objects per image
        close_mosaic=15,  # ...but stop before the end so the model sees real layouts
        degrees=args.degrees,   # 180: a survivor seen from above has no canonical orientation
        fliplr=0.5, flipud=0.5 if args.degrees >= 90 else 0.0,
        hsv_v=0.4,        # apparent-temperature/exposure variation
        translate=0.2, erasing=0.2,
        patience=args.patience, cos_lr=True,
    )
    if args.channels == 1:
        overrides.update(hsv_h=0.0, hsv_s=0.0)       # no hue/saturation in LWIR
    else:
        overrides.update(hsv_h=0.015, hsv_s=0.7)
    print(json.dumps({k: str(v) for k, v in overrides.items()}, indent=2))
    model.train(**overrides)
    metrics = model.val()
    print(metrics)
    print("\nExport next:\n"
          f"  python3 scripts/export_model.py --weights {args.project}/{args.name}"
          f"/weights/best.pt --out models/yolo11n-thermal-sar.onnx "
          f"--imgsz {args.imgsz} {args.imgsz} --channels 3")
    return 0


def resolve_model(name: str) -> str:
    """Resolve a model YAML to something Ultralytics can load.

    Upstream ships no P2 variant of YOLO11, so the repo bundles one.  A bare
    name is first looked up in ``configs/models/``, then passed through
    unchanged (Ultralytics resolves its own zoo names, e.g. ``yolo11n.yaml``).
    """
    p = Path(name)
    if p.is_file():
        return str(p)
    local = Path(__file__).resolve().parents[1] / "configs" / "models" / p.name
    if local.is_file():
        return str(local)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="Train / synthesise data for the SAR detector")
    ap.add_argument("--synthesize", type=int, metavar="N",
                    help="render N labelled frames from the simulator")
    ap.add_argument("--scenario", default="mixed",
                    help="'mixed' or a comma-separated list of scenario presets")
    ap.add_argument("--modality", default="lwir", choices=("lwir", "rgb"))
    ap.add_argument("--alt-min", type=float, default=30.0)
    ap.add_argument("--alt-max", type=float, default=90.0)
    ap.add_argument("--out", default="datasets/sim-thermal")
    ap.add_argument("--seed", type=int, default=11)

    ap.add_argument("--dataset-yaml", help="write a YAML stub for a public dataset")
    ap.add_argument("--list-datasets", action="store_true")

    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", help="dataset YAML for training")
    ap.add_argument("--model", default="yolo11n.pt",
                    help=".pt = fine-tune those weights (the default, and what you "
                         "almost always want); .yaml = build the architecture and "
                         "train FROM SCRATCH on random weights")
    ap.add_argument("--load", metavar="WEIGHTS",
                    help="transfer weights from this .pt into the architecture "
                         "named by --model. Implied when --p2 is combined with a "
                         ".pt, because a stride-4 head cannot come from a stock "
                         "checkpoint - the graph is rebuilt and the weights that "
                         "do fit are carried over")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--channels", type=int, default=None,
                    help="augmentation regime: 1 = LWIR (no hue/sat jitter, which "
                         "do not exist in a single band); 3 = RGB. Default: "
                         "inferred from the dataset, and the choice is printed. "
                         "The trained model is 3-channel either way (ultralytics "
                         "loads grayscale as BGR-3).")
    ap.add_argument("--degrees", type=float, default=180.0,
                    help="rotation jitter. 180 suits nadir aerial, where a "
                         "survivor seen from above has no canonical orientation; "
                         "use 0 for oblique or ground-level RGB (parts of SARD), "
                         "where an upside-down person is not a real training "
                         "example. Vertical flip follows this setting.")
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stopping patience, in epochs")
    ap.add_argument("--project", default="runs/sar")
    ap.add_argument("--name", default="thermal")
    ap.add_argument("--p2", action="store_true",
                    help="use the stride-4 head variant (yolo11n-p2.yaml)")
    args = ap.parse_args()

    if args.list_datasets:
        for k, v in PUBLIC_DATASETS.items():
            print(f"  {k:<12} {v['modality']:<5} {v['images'] or '?':>6} images  {v['url']}")
            print(f"               {v['notes']}")
        return
    if args.dataset_yaml:
        write_dataset_yaml(args.dataset_yaml, Path(args.out))
        return
    if args.synthesize:
        scenarios = (["flood", "earthquake", "wildfire", "landslide"]
                     if args.scenario == "mixed" else args.scenario.split(","))
        synthesize(args.synthesize, Path(args.out), scenarios,
                   modality=args.modality, seed=args.seed,
                   alt_range=(args.alt_min, args.alt_max))
        return
    if args.train:
        if not args.data:
            ap.error("--train needs --data")
        if args.p2:
            # A stride-4 head is not in any stock checkpoint, so the graph has to
            # be rebuilt from the bundled P2 YAML - but keep the .pt the user
            # asked for as the transfer source rather than throwing it away.
            if args.model.lower().endswith(".pt") and not args.load:
                args.load = args.model
            args.model = p2_architecture(args.model)
        args.model = resolve_model(args.model)
        if args.p2 and not Path(args.model).is_file():
            raise SystemExit(
                f"--p2 needs a bundled P2 architecture, and {args.model!r} is not "
                f"one. Only configs/models/yolo11n-p2.yaml ships with this repo; "
                f"add a P2 YAML for that scale under configs/models/, or train "
                f"the 'n' scale.")
        args.channels, why = infer_channels(args.data, args.channels)
        if args.channels not in (1, 3):
            ap.error(f"--channels must be 1 (LWIR augmentation) or 3 (RGB), got "
                     f"{args.channels}")
        print(f"augmentation regime: --channels {args.channels}  ({why})")
        raise SystemExit(train(args))
    ap.print_help()


if __name__ == "__main__":
    main()
