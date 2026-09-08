#!/usr/bin/env python3
"""List, fetch and verify the detector weights this aircraft can fly.

Nothing in the runtime downloads anything by itself.  On a field network at
2 a.m. an implicit download is a hang at arm time, so acquiring weights is an
explicit, auditable step - this script.

    python3 scripts/fetch_models.py --list
    python3 scripts/fetch_models.py --model yolo11n-rgb-coco
    python3 scripts/fetch_models.py --verify-all
    python3 scripts/fetch_models.py --synthetic      # tiny CI/plumbing model

``--model yolo11n-rgb-coco`` exports the pretrained Ultralytics checkpoint to
ONNX locally (requires the ``train`` extra).  Models whose ``source`` starts
with ``train:`` cannot be fetched - they are produced by
``scripts/train_detector.py`` on your own data, and the script says so rather
than pretending a download exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# --- repo-root bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()

from sar.ai.registry import ModelFormat, ModelRegistry, default_registry  # noqa: E402


def cmd_list(reg: ModelRegistry) -> int:
    print(f"\nmodels dir: {reg.models_dir}\n")
    for m in reg.models():
        ok = "present" if reg.is_available(m.name) else "-"
        print(f"  {m.name:<28} {m.modality:<9} {m.fmt.value:<7} {ok:<8} {m.architecture}")
        print(f"    {m.notes.splitlines()[0] if m.notes else ''}")
        print(f"    source: {m.source or 'n/a'}")
        if m.latency_ms:
            lat = ", ".join(f"{k}={v:.0f} ms" for k, v in m.latency_ms.items())
            print(f"    latency: {lat}")
        print()
    missing = [m.name for m in reg.models() if not reg.is_available(m.name)]
    if missing:
        print("Nothing is broken if these are missing: the audited heuristic "
              "detector runs instead and the sortie report records which stack "
              "produced its numbers.\n")
    return 0


def cmd_verify(reg: ModelRegistry) -> int:
    bad = 0
    for m in reg.models():
        ok, msg = reg.verify(m.name)
        print(f"  {'PASS' if ok else 'FAIL'}  {m.name:<28} {msg}")
        bad += 0 if ok else 1
    return 1 if bad else 0


def cmd_fetch(reg: ModelRegistry, name: str) -> int:
    spec = reg.get(name)
    dest = reg.path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if reg.is_available(name):
        print(f"already present: {dest}")
        return 0
    src = spec.source
    if src.startswith("train:"):
        print(f"{name} is not downloadable.\n  {src}\n"
              "  See docs/06_AI_MODELS_AND_DATASETS.md for the dataset list and recipe.")
        return 2
    if src.startswith("qai_hub_models"):
        print(f"{name} is produced by the Qualcomm AI Hub exporter:\n  {src}\n"
              "  Run it on a workstation with a Qualcomm AI Hub token, then copy "
              f"the .bin into {reg.models_dir}.")
        return 2
    if src.startswith("ultralytics:"):
        ckpt = src.split(":", 1)[1]
        print(f"exporting {ckpt} -> {dest} (needs the 'train' extra)")
        try:
            from sar.ai.export import export_ultralytics_to_onnx
            export_ultralytics_to_onnx(ckpt, dest, imgsz=spec.input_size)
        except Exception as exc:
            print(f"  failed: {exc}")
            print("  pip install -e '.[train]'   # ultralytics + torch, workstation only")
            return 1
        print(f"  ok: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0
    if src.startswith("http"):
        print(f"downloading {src} -> {dest}")
        urllib.request.urlretrieve(src, dest)  # noqa: S310 - explicit operator action
        ok, msg = reg.verify(name)
        print(f"  {msg}")
        return 0 if ok else 1
    print(f"no fetch method for source {src!r}")
    return 2


def cmd_synthetic(reg: ModelRegistry) -> int:
    """Write the tiny plumbing model + a registry override that points at it."""
    from sar.ai.export import make_synthetic_detector_onnx, write_registry_entry
    path = reg.models_dir / "synthetic-lwir.onnx"
    make_synthetic_detector_onnx(path, input_hw=(128, 128), channels=1)
    entry = {
        "name": "synthetic-lwir", "task": "person_detection", "modality": "lwir",
        "architecture": "synthetic (CI plumbing only)", "input_size": [128, 128],
        "classes": ["person", "person_group", "vehicle", "animal", "fire", "structure"],
        "class_map": {"person": "person"}, "file_name": "synthetic-lwir.onnx",
        "fmt": "onnx", "conf_threshold": 0.2, "channels": 1, "normalise": "raw",
        "notes": "NOT a detector. Proves the neural path loads, preprocesses, "
                 "decodes and vetoes correctly without a licensed download.",
    }
    write_registry_entry(reg.models_dir, entry)
    print(f"wrote {path}\nwrote {reg.models_dir / 'registry.json'}\n"
          "Run:  SAR_DETECTOR=neural python3 scripts/run_rescue_simulation.py "
          "--duration 30\n(expect poor detections - it is a plumbing model, not a "
          "trained one)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/verify SAR detector weights")
    ap.add_argument("--list", action="store_true", help="show the model zoo")
    ap.add_argument("--model", help="fetch or export one model by name")
    ap.add_argument("--verify-all", action="store_true", help="checksum every model")
    ap.add_argument("--synthetic", action="store_true",
                    help="write the tiny CI plumbing model into models/")
    ap.add_argument("--json", action="store_true", help="registry summary as JSON")
    ap.add_argument("--models-dir", help="override the models directory")
    args = ap.parse_args()

    reg = ModelRegistry(Path(args.models_dir)) if args.models_dir else default_registry()

    if args.json:
        print(json.dumps(reg.summary(), indent=2))
        raise SystemExit(0)
    if args.synthetic:
        raise SystemExit(cmd_synthetic(reg))
    if args.verify_all:
        raise SystemExit(cmd_verify(reg))
    if args.model:
        raise SystemExit(cmd_fetch(reg, args.model))
    raise SystemExit(cmd_list(reg))


if __name__ == "__main__":
    main()
