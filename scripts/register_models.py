#!/usr/bin/env python3
"""Write ``models/registry.json`` for weights you trained elsewhere (Colab/Kaggle).

The notebook exports ``.onnx`` files but nothing registers them, and the built-in
zoo in ``sar/ai/registry.py`` only knows the *file names it was written with*.
An unregistered model is invisible to ``best_for()``, so ``SAR_DETECTOR=auto``
silently falls back to the heuristic detector - the failure looks like "the
neural model did not work".

    python3 scripts/register_models.py                 # scan models/, write registry.json
    python3 scripts/register_models.py --dry-run       # show what would be written
    python3 scripts/register_models.py --verify        # register, then checksum + load

Only two things in the entry actually matter at runtime:

* ``classes`` - ``sar/ai/backends.py`` passes ``num_classes=len(spec.classes)`` to
  the decoder, which expects a head of ``4 + nc``.  Get this wrong and every box
  decodes to the wrong label; it presents as a bad detector, not a bad config.
* ``file_name`` - what ``best_for()`` looks for on disk.

``input_size`` and ``channels`` are re-read from the ONNX graph by
``sar/ai/runtime.py`` ("trust the graph over the caller"), so they are recorded
for provenance rather than correctness.

No GPU, no torch, no network.  ``onnxruntime`` is used when present to read the
graph; without it the shapes come from ``--input-size`` / ``--nc``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Deliberately no scripts._bootstrap and no `import sar.ai`: both pull in numpy
# (via sar.ai.runtime) and the bootstrap gate hard-exits without it.  This
# script has to run on a laptop that has only the repo checked out and no venv,
# so registry.py - which is stdlib-only - is loaded directly by path.
ROOT = Path(__file__).resolve().parents[1]


def _load_registry_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sar_registry", ROOT / "sar" / "ai" / "registry.py")
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so the module must be
    # registered before it is executed.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_registry = _load_registry_module()
ModelFormat = _registry.ModelFormat
ModelSpec = _registry.ModelSpec
DEFAULT_MODELS_DIR = _registry.DEFAULT_MODELS_DIR

#: Must match scripts/train_detector.py::CLASSES and registry.py::_THERMAL_SAR_CLASSES.
PROJECT_CLASSES: Tuple[str, ...] = ("person", "person_group", "vehicle", "animal",
                                    "fire", "structure")

#: Weight formats the registry understands, by suffix.
FMT_BY_SUFFIX = {".onnx": ModelFormat.ONNX, ".tflite": ModelFormat.TFLITE,
                 ".bin": ModelFormat.QNN, ".pt": ModelFormat.TORCH}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_onnx_graph(path: Path) -> Optional[Dict[str, Any]]:
    """Return ``{input_hw, channels, nc, num_classes_source}`` or None."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:                                  # corrupt / unsupported
        print(f"  ! cannot open {path.name}: {exc}")
        return None
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    info: Dict[str, Any] = {"input_hw": None, "channels": None, "nc": None}
    if len(inp.shape) == 4:
        if all(isinstance(d, int) for d in inp.shape[2:]):
            info["input_hw"] = (int(inp.shape[2]), int(inp.shape[3]))
        if isinstance(inp.shape[1], int):
            info["channels"] = int(inp.shape[1])
    # YOLO head is (1, 4+nc, N) or (1, N, 4+nc); the head is the smaller axis.
    dims = [d for d in out.shape[1:] if isinstance(d, int)]
    if len(dims) == 2:
        head = min(dims)
        if head > 4:
            info["nc"] = head - 4
            info["output_shape"] = list(out.shape)
    return info


def guess_modality(name: str) -> str:
    low = name.lower()
    if "thermal" in low or "lwir" in low or "ir" == low.split("-")[-1]:
        return "lwir"
    return "rgb"


def classes_for(nc: Optional[int], override: Optional[List[str]]) -> Tuple[Tuple[str, ...], str]:
    """The class list to record, and where it came from."""
    if override:
        return tuple(override), "--classes"
    if nc == len(PROJECT_CLASSES):
        return PROJECT_CLASSES, "project CLASSES (6)"
    if nc == 1:
        return ("person",), "single-class dataset (SARD-style)"
    if nc is None:
        return PROJECT_CLASSES, "ASSUMED project CLASSES - verify with --nc"
    raise SystemExit(
        f"nc={nc} matches neither the project's 6 classes nor a 1-class person "
        f"dataset. Pass --classes explicitly, in the order your data.yaml lists "
        f"them - the decoder maps box class ids positionally.")


def build_entry(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    graph = read_onnx_graph(path) if path.suffix == ".onnx" else None
    if graph is None:
        graph = {}

    input_hw = tuple(args.input_size) if args.input_size else graph.get("input_hw") or (640, 640)
    channels = args.channels or graph.get("channels") or 3
    nc = args.nc if args.nc is not None else graph.get("nc")
    if args.nc is None and nc is None and path.suffix == ".onnx" and graph:
        print(f"  ! {path.name}: could not read nc from the output head "
              f"{graph.get('output_shape')}; assuming 6 project classes")

    override = args.classes.split(",") if args.classes else None
    classes, provenance = classes_for(nc, override)
    modality = args.modality or guess_modality(path.stem)
    name = args.name or path.stem

    entry: Dict[str, Any] = {
        "name": name,
        "task": "person_detection",
        "modality": modality,
        "architecture": args.architecture or f"YOLO11n ({'P2' if 'p2' in path.stem.lower() else 'stock'} head)",
        "input_size": [int(input_hw[0]), int(input_hw[1])],
        "classes": list(classes),
        "class_map": {c: c for c in classes},
        "file_name": path.name,
        "fmt": (FMT_BY_SUFFIX.get(path.suffix, ModelFormat.ONNX)).value,
        "conf_threshold": args.conf if args.conf is not None else (0.20 if modality == "lwir" else 0.25),
        "iou_threshold": 0.45,
        "normalise": "0-1",
        "channels": int(channels),
        "pretrained": False,
        "source": args.source or f"train:scripts/train_detector.py ({'GPU' if not args.source else ''})",
        "license": "AGPL-3.0 (Ultralytics); datasets under their own terms",
        "trained_on": args.trained_on.split(",") if args.trained_on else [],
        "latency_ms": {},
        "sha256": sha256(path),
        "notes": args.notes or f"Registered by scripts/register_models.py; classes from {provenance}.",
    }
    # Fail loudly now rather than at arm time: the entry must construct a ModelSpec.
    fields = {k: v for k, v in entry.items() if k in ModelSpec.__dataclass_fields__}
    fields["input_size"] = tuple(fields["input_size"])
    fields["classes"] = tuple(fields["classes"])
    fields["trained_on"] = tuple(fields["trained_on"])
    fields["fmt"] = ModelFormat(fields["fmt"])
    ModelSpec(**fields)
    return entry


def merge(models_dir: Path, entries: List[Dict[str, Any]]) -> Path:
    """Append/replace entries in models/registry.json (same semantics as
    sar.ai.export.write_registry_entry, without importing the ONNX path)."""
    path = models_dir / "registry.json"
    data: Dict[str, Any] = {"models": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except Exception:
            print(f"  ! {path} is not valid JSON; rewriting it")
            data = {"models": []}
    models = list(data.get("models", []))
    for entry in entries:
        models = [m for m in models if m.get("name") != entry.get("name")
                  and m.get("file_name") != entry.get("file_name")]
        models.append(entry)
    data["models"] = models
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    models_dir = Path(DEFAULT_MODELS_DIR)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default=str(models_dir), help="where the weights live")
    ap.add_argument("--file", action="append",
                    help="register this specific file (repeatable); default: scan the models dir")
    ap.add_argument("--name", help="registry name (default: file stem)")
    ap.add_argument("--modality", choices=("lwir", "rgb"), help="default: guessed from the name")
    ap.add_argument("--nc", type=int, help="class count; default: read from the ONNX head")
    ap.add_argument("--classes", help="comma-separated class names, in data.yaml order")
    ap.add_argument("--input-size", nargs=2, type=int, metavar=("H", "W"),
                    help="default: read from the ONNX graph")
    ap.add_argument("--channels", type=int, help="default: read from the ONNX graph")
    ap.add_argument("--conf", type=float, help="confidence threshold (default 0.20 lwir / 0.25 rgb)")
    ap.add_argument("--architecture", help="free-text, recorded in the sortie report")
    ap.add_argument("--trained-on", help="comma-separated dataset names, e.g. HIT-UAV,SARD")
    ap.add_argument("--source", help="provenance string")
    ap.add_argument("--notes", help="provenance string")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--verify", action="store_true", help="checksum every registered model after writing")
    args = ap.parse_args(argv)

    mdir = Path(args.models_dir)
    if not mdir.is_dir():
        raise SystemExit(f"no such models dir: {mdir}")

    if args.file:
        files = [Path(f) if Path(f).is_absolute() else mdir / Path(f).name for f in args.file]
    else:
        files = sorted(p for p in mdir.iterdir()
                       if p.suffix in FMT_BY_SUFFIX and p.suffix != ".pt")
        if len(files) > 1 and (args.name or args.nc or args.classes or args.modality):
            raise SystemExit("--name/--nc/--classes/--modality apply to one model; "
                             "use --file, or scan without them")

    if not files:
        print(f"nothing to register in {mdir}")
        print("  expected e.g. yolo11n-thermal-sar.onnx / yolo11n-rgb-sar.onnx")
        print("  (.pt checkpoints are training artefacts and are never flown)")
        return 1

    entries = []
    for path in files:
        if not path.is_file():
            print(f"  ! missing: {path}")
            continue
        entry = build_entry(path, args)
        entries.append(entry)
        print(f"  + {entry['name']:<28} {entry['modality']:<5} "
              f"{entry['fmt']:<7} input={tuple(entry['input_size'])} "
              f"ch={entry['channels']} nc={len(entry['classes'])} "
              f"{path.stat().st_size / 1e6:.1f} MB sha256={entry['sha256'][:16]}…")

    if not entries:
        return 1

    if args.dry_run:
        print(json.dumps({"models": entries}, indent=2))
        return 0

    out = merge(mdir, entries)
    print(f"wrote {out} ({len(entries)} model{'s' if len(entries) != 1 else ''})")

    if args.verify:
        reg = _registry.ModelRegistry(mdir)
        bad = 0
        for spec in reg.models():
            if not reg.is_available(spec.name):
                continue
            ok, msg = reg.verify(spec.name)
            print(f"  {'ok ' if ok else 'FAIL'} {msg}")
            bad += 0 if ok else 1
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
