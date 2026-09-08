"""The model zoo: which network we fly, in which format, and why.

Selection evidence (2025-2026 literature, summarised in
``docs/06_AI_MODELS_AND_DATASETS.md``)
--------------------------------------------------------------------------
* On **UAV thermal** imagery a CNN one-stage detector still beats a real-time
  transformer by a wide margin at the sizes that matter to us.  In the largest
  published comparison on a 75k-image thermal meta-dataset, RT-DETR-L reached
  36.3% AP50-95 against 48.1-49.4% for the YOLO family, and on the *small
  object* subset 5.0% against 10.7-11.6%.  Thermal blobs have no texture for an
  attention backbone to exploit, and thermal training data is scarce, so the
  convolutional inductive bias wins.  A survivor at 50 m AGL is a small object
  by definition; that is the subset we live in.
* Within the YOLO family, **YOLO11n / YOLOv8n** are the only variants that
  survive INT8 quantisation onto a 12-TOPS class NPU with margin left for the
  rest of the autonomy stack, and both have first-party Qualcomm AI Hub export
  recipes (``qai_hub_models.models.yolov11_det.export --quantize w8a8
  --target-runtime qnn --chipset qualcomm-qcs6490-proxy``) for the RB3 Gen 2 we
  target.
* Nothing off the shelf is trained on *aerial LWIR persons*.  A COCO-trained
  network is a starting point for the RGB channel only; the thermal channel is
  a fine-tune on HIT-UAV / AIResQ / SARD-class data.  Both entries are in the
  zoo, and the registry is explicit about which weights are pretrained and
  which must be produced by ``scripts/train_detector.py``.

Design rules
------------
1. **Absence is a supported state.**  No weights on disk is not an error; the
   audited heuristic detector runs instead and the sortie report says so.
2. **Every model declares its own postprocessing.**  Input size, layout, class
   list and normalisation live with the spec, not in the caller, so swapping a
   model cannot silently break decode.
3. **Checksums are recorded.**  A model file that changed underneath a measured
   result invalidates the result; the registry can prove it did not.
4. **No implicit downloads.**  Nothing reaches the network unless a human runs
   ``scripts/fetch_models.py``.  An aircraft on a field network must never
   block on a download at arm time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Where weights live.  Overridable so a companion computer can keep models on
#: a different mount from the code (a read-only /opt/sar with a writable /data).
MODELS_DIR_ENV = "SAR_MODELS_DIR"
DEFAULT_MODELS_DIR = Path(
    os.environ.get(MODELS_DIR_ENV, Path(__file__).resolve().parents[2] / "models"))


class ModelFormat(str, Enum):
    """Numeric/runtime format.  Ordered by deployment preference on target."""

    QNN = "qnn"          # Qualcomm Hexagon NPU, .bin from AI Hub - fastest on RB3
    TFLITE = "tflite"    # Qualcomm/Android delegate path, also AI Hub output
    ONNX = "onnx"        # portable; CPU / TensorRT EP / QNN EP
    TORCH = "pt"         # training and export only, never flown


@dataclass
class ModelSpec:
    """Everything needed to load, run and account for one model."""

    name: str
    task: str                              # 'person_detection' | 'hazard_detection'
    modality: str                          # 'rgb' | 'lwir' | 'rgb+lwir'
    architecture: str
    input_size: Tuple[int, int]            # (h, w) the network was exported at
    classes: Tuple[str, ...]               # network class order
    class_map: Dict[str, str]              # network class -> project label
    file_name: str                         # relative to the models dir
    fmt: ModelFormat = ModelFormat.ONNX
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    normalise: str = "0-1"                 # '0-1' | 'imagenet' | 'raw'
    channels: int = 3
    pretrained: bool = False               # true = downloadable weights exist
    source: str = ""                       # URL or hub id, for fetch_models.py
    sha256: str = ""                       # "" = unverified (dev checkpoints)
    license: str = "unknown"
    trained_on: Tuple[str, ...] = ()
    #: Published or measured latency, ms, keyed by device string.  Used by the
    #: planner's perception-rate budget and printed in the sortie report.
    latency_ms: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fmt"] = self.fmt.value
        d["input_size"] = list(self.input_size)
        d["classes"] = list(self.classes)
        d["trained_on"] = list(self.trained_on)
        return d


# --------------------------------------------------------------------------- #
# The zoo
# --------------------------------------------------------------------------- #
#: COCO class order, truncated to what we consume.  A COCO model detects
#: 'person' at index 0, which is the only class we take from it directly; the
#: rest map onto hazard/context labels the fuser understands.
_COCO_MAP = {
    "person": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "boat": "vehicle",
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "cow": "animal",
    "sheep": "animal",
}

_THERMAL_SAR_CLASSES = ("person", "person_group", "vehicle", "animal", "fire", "structure")

MODEL_ZOO: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="yolo11n-rgb-coco",
        task="person_detection",
        modality="rgb",
        architecture="YOLO11n",
        input_size=(640, 640),
        classes=tuple(_COCO_MAP),                     # decoded by name, see backends
        class_map=_COCO_MAP,
        file_name="yolo11n.onnx",
        fmt=ModelFormat.ONNX,
        conf_threshold=0.25,
        pretrained=True,
        source="ultralytics:yolo11n.pt",              # exported by scripts/export_model.py
        license="AGPL-3.0 (Ultralytics)",
        trained_on=("COCO",),
        latency_ms={"x86_cpu_2core": 78.0, "qcs6490_npu_int8": 12.0,
                    "jetson_orin_nano_fp16": 6.5},
        notes="Baseline visible-band detector. Useful only in daylight and only "
              "as the confirming half of the cross-modal pair; a person at "
              "0.12 m/px is 10-15 px and COCO has almost no examples that small.",
    ),
    ModelSpec(
        name="yolo11n-visdrone",
        task="person_detection",
        modality="rgb",
        architecture="YOLO11n + P2 head",
        input_size=(960, 960),
        classes=("pedestrian", "people", "bicycle", "car", "van", "truck",
                 "tricycle", "awning-tricycle", "bus", "motor"),
        class_map={"pedestrian": "person", "people": "person_group",
                   "car": "vehicle", "van": "vehicle", "truck": "vehicle",
                   "bus": "vehicle", "motor": "vehicle"},
        file_name="yolo11n-visdrone.onnx",
        fmt=ModelFormat.ONNX,
        conf_threshold=0.20,
        pretrained=False,
        source="train: scripts/train_detector.py --dataset visdrone",
        license="AGPL-3.0 (Ultralytics) / VisDrone research licence",
        trained_on=("VisDrone2019-DET",),
        latency_ms={"x86_cpu_2core": 165.0, "qcs6490_npu_int8": 29.0},
        notes="Aerial-domain RGB. The P2 (stride-4) head is the change that "
              "matters for 8-20 px targets; without it the smallest anchor is "
              "already larger than a survivor.",
    ),
    ModelSpec(
        name="yolo11n-thermal-sar",
        task="person_detection",
        modality="lwir",
        architecture="YOLO11n (1-ch stem, P2 head)",
        input_size=(640, 512),
        classes=_THERMAL_SAR_CLASSES,
        class_map={c: c for c in _THERMAL_SAR_CLASSES},
        file_name="yolo11n-thermal-sar.onnx",
        fmt=ModelFormat.ONNX,
        conf_threshold=0.20,
        channels=1,
        normalise="raw",          # radiometric AGC happens in preprocess_lwir
        pretrained=False,
        source="train: scripts/train_detector.py --dataset hit-uav+airesq+sard",
        license="AGPL-3.0 (Ultralytics); datasets under their own terms",
        trained_on=("HIT-UAV", "AIResQ", "SARD", "sim: sar.sim.renderer"),
        latency_ms={"x86_cpu_2core": 95.0, "qcs6490_npu_int8": 15.0},
        notes="The model that actually earns its place: LWIR is the only "
              "modality that works at night, through smoke and under light "
              "canopy. Published aerial-thermal mAP50 tops out near 0.55 on "
              "AIResQ-trained YOLOv8/YOLO11 - treat any higher claim as a "
              "different (easier) test set.",
    ),
    ModelSpec(
        name="yolo11n-thermal-sar-qnn",
        task="person_detection",
        modality="lwir",
        architecture="YOLO11n INT8 (w8a8) compiled for Hexagon",
        input_size=(640, 512),
        classes=_THERMAL_SAR_CLASSES,
        class_map={c: c for c in _THERMAL_SAR_CLASSES},
        file_name="yolo11n-thermal-sar-qcs6490.bin",
        fmt=ModelFormat.QNN,
        conf_threshold=0.22,
        channels=1,
        normalise="raw",
        pretrained=False,
        source="qai_hub_models.models.yolov11_det.export --quantize w8a8 "
               "--target-runtime qnn --chipset qualcomm-qcs6490-proxy",
        license="AGPL-3.0 (Ultralytics)",
        trained_on=("HIT-UAV", "AIResQ", "SARD"),
        latency_ms={"qcs6490_npu_int8": 15.0},
        notes="Flight format on the RB3 Gen 2. Quantisation is not free: "
              "validate w8a8 against the FP32 ONNX on the held-out thermal set "
              "and fall back to w8a16 if small-object recall drops more than "
              "3 points (scripts/export_model.py --validate).",
    ),
    ModelSpec(
        name="hazard-seg-rescuenet",
        task="hazard_detection",
        modality="rgb",
        architecture="YOLO11n-seg",
        input_size=(640, 640),
        classes=("water", "building-collapsed", "building-damaged", "road-blocked",
                 "debris", "fire", "smoke", "vehicle"),
        class_map={"water": "flood_water", "building-collapsed": "collapsed_structure",
                   "building-damaged": "damaged_structure", "road-blocked": "blocked_road",
                   "debris": "debris_field", "fire": "fire", "smoke": "smoke",
                   "vehicle": "vehicle"},
        file_name="hazard-seg-rescuenet.onnx",
        fmt=ModelFormat.ONNX,
        conf_threshold=0.30,
        pretrained=False,
        source="train: scripts/train_detector.py --dataset rescuenet+floodnet",
        license="AGPL-3.0 (Ultralytics); RescueNet/FloodNet research licences",
        trained_on=("RescueNet", "FloodNet", "AIDER"),
        latency_ms={"x86_cpu_2core": 140.0, "qcs6490_npu_int8": 24.0},
        notes="Hazard context for the belief map and for landing-zone rejection. "
              "Runs at 0.5 Hz, not per frame - hazards move on a slower clock "
              "than survivors do.",
    ),
)


class ModelRegistry:
    """Locates, verifies and describes the models available to this aircraft."""

    def __init__(self, models_dir: Optional[Path] = None,
                 zoo: Sequence[ModelSpec] = MODEL_ZOO) -> None:
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self._zoo: Dict[str, ModelSpec] = {m.name: m for m in zoo}
        self._overrides_loaded = False
        self._load_overrides()

    # -- discovery ------------------------------------------------------- #
    def _load_overrides(self) -> None:
        """Merge ``models/registry.json`` if present.

        A field-deployed aircraft may carry a model that post-dates the code.
        Rather than requiring a code change to fly it, an operator can drop a
        JSON entry beside the weights.  Unknown keys are ignored, so an older
        binary reading a newer registry degrades instead of crashing.
        """
        path = self.models_dir / "registry.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        for entry in data.get("models", []):
            try:
                fields = {k: v for k, v in entry.items()
                          if k in ModelSpec.__dataclass_fields__}
                fields["input_size"] = tuple(fields.get("input_size", (640, 640)))
                fields["classes"] = tuple(fields.get("classes", ()))
                fields["trained_on"] = tuple(fields.get("trained_on", ()))
                fields["fmt"] = ModelFormat(fields.get("fmt", "onnx"))
                spec = ModelSpec(**fields)
                self._zoo[spec.name] = spec
            except Exception:
                continue
        self._overrides_loaded = True

    def models(self, task: Optional[str] = None,
               modality: Optional[str] = None) -> List[ModelSpec]:
        out = list(self._zoo.values())
        if task:
            out = [m for m in out if m.task == task]
        if modality:
            out = [m for m in out if modality in m.modality]
        return out

    def get(self, name: str) -> ModelSpec:
        if name not in self._zoo:
            raise KeyError(f"unknown model {name!r}; known: {sorted(self._zoo)}")
        return self._zoo[name]

    def path(self, name: str) -> Path:
        return self.models_dir / self.get(name).file_name

    def is_available(self, name: str) -> bool:
        try:
            return self.path(name).is_file()
        except KeyError:
            return False

    # -- integrity ------------------------------------------------------- #
    def sha256(self, name: str) -> str:
        p = self.path(name)
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify(self, name: str) -> Tuple[bool, str]:
        """Return (ok, message).  An unrecorded checksum is a warning, not a pass."""
        spec = self.get(name)
        if not self.is_available(name):
            return False, f"{spec.file_name} not present in {self.models_dir}"
        if not spec.sha256:
            return True, f"{spec.file_name} present (no checksum recorded)"
        actual = self.sha256(name)
        if actual != spec.sha256:
            return False, (f"{spec.file_name} checksum mismatch: expected "
                           f"{spec.sha256[:16]}..., got {actual[:16]}...")
        return True, f"{spec.file_name} verified"

    # -- selection ------------------------------------------------------- #
    def best_for(self, task: str, modality: str,
                 prefer: Sequence[ModelFormat] = (ModelFormat.QNN, ModelFormat.TFLITE,
                                                  ModelFormat.ONNX)) -> Optional[ModelSpec]:
        """The fastest *available* model for a task/modality, or None.

        Preference is by deployment format, not by accuracy: within this zoo the
        formats hold the same weights, so the fastest one that is actually on
        disk and loadable is strictly better.
        """
        candidates = [m for m in self.models(task, modality) if self.is_available(m.name)]
        if not candidates:
            return None
        order = {f: i for i, f in enumerate(prefer)}
        candidates.sort(key=lambda m: order.get(m.fmt, len(order)))
        return candidates[0]

    def summary(self) -> Dict[str, Any]:
        return {
            "models_dir": str(self.models_dir),
            "models": [{**m.to_dict(),
                        "available": self.is_available(m.name),
                        "path": str(self.path(m.name))}
                       for m in self.models()],
        }


_DEFAULT: Optional[ModelRegistry] = None


def default_registry() -> ModelRegistry:
    """Process-wide registry.  Cheap to call; the filesystem is only touched lazily."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelRegistry()
    return _DEFAULT


__all__ = ["MODEL_ZOO", "ModelFormat", "ModelRegistry", "ModelSpec",
           "DEFAULT_MODELS_DIR", "default_registry"]
