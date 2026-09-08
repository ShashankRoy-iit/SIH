"""Export and packaging: PyTorch checkpoint -> ONNX -> INT8 -> Hexagon.

Three separate concerns live here, deliberately in one place so the deployment
path is readable end to end:

1. :func:`export_ultralytics_to_onnx` - the training-time export.  Needs
   ``ultralytics``/``torch``, which are never installed on the aircraft.
2. :func:`validate_onnx_against_reference` - the step teams skip and then
   discover in the field.  Quantisation is not free: INT8 (w8a8) can cost small
   object recall specifically, which is the recall we depend on.  This compares
   detections between two model files on the same images and reports the delta,
   so "we quantised it and it was fine" becomes a number.
3. :func:`make_synthetic_detector_onnx` - a tiny, valid, input-dependent ONNX
   detector used by the tests and by ``scripts/doctor.py``.  It exists so the
   *whole neural path* (session creation, provider selection, letterbox,
   decode, NMS, Detection construction, geometry veto) is exercised on every CI
   run without a 6 MB download or a licence question.  It is not a model of
   anything; it is a proof that the plumbing carries water.

Qualcomm path (RB3 Gen 2 / QCS6490, or RB5 / QCS8550)
-----------------------------------------------------
::

    pip install "qai-hub-models[yolov11_det]" qai-hub
    qai-hub configure --api_token <token>

    python -m qai_hub_models.models.yolov11_det.export \
        --quantize w8a8 \
        --target-runtime qnn \
        --chipset qualcomm-qcs6490-proxy \
        --output-dir models/

That emits a ``.bin`` QNN model library which the registry entry
``yolo11n-thermal-sar-qnn`` points at.  Validate it with
``scripts/export_model.py --validate`` before it flies; if w8a8 costs more than
~3 points of small-object recall, re-export with ``--quantize w8a16``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("sar.ai.export")


# --------------------------------------------------------------------------- #
# Training-time export
# --------------------------------------------------------------------------- #
def export_ultralytics_to_onnx(weights: str | Path, out_path: str | Path, *,
                               imgsz: Tuple[int, int] = (640, 640),
                               opset: int = 12, simplify: bool = True,
                               half: bool = False, dynamic: bool = False) -> Path:
    """Export an Ultralytics checkpoint to ONNX for the flight runtime.

    ``opset=12`` on purpose: it is the highest opset the Qualcomm QNN converter
    accepts without op fallbacks that silently move layers back to the CPU, and
    a model that runs half on the NPU and half on the CPU is slower than one
    that runs entirely on the CPU.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - training-only dependency
        raise RuntimeError(
            "ultralytics is not installed. This is an export-time dependency "
            "only: `pip install -e '.[train]'` on a workstation, never on the "
            "aircraft.") from exc
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    produced = model.export(format="onnx", imgsz=list(imgsz), opset=opset,
                            simplify=simplify, half=half, dynamic=dynamic)
    produced = Path(produced)
    if produced.resolve() != out_path.resolve():
        produced.replace(out_path)
    log.info("exported %s -> %s", weights, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Quantisation validation
# --------------------------------------------------------------------------- #
def validate_onnx_against_reference(candidate: str | Path, reference: str | Path,
                                    images: Sequence[np.ndarray], *,
                                    input_hw: Tuple[int, int] = (640, 640),
                                    channels: int = 3, conf: float = 0.25,
                                    iou_match: float = 0.5) -> Dict[str, Any]:
    """Compare a quantised model against its float reference on real frames.

    Reports, per image and in aggregate: detections found by both, by the
    reference only (**recall lost to quantisation** - the number that matters),
    and by the candidate only (new false alarms).  Deliberately measured on the
    *deployment* preprocessing path, because a mismatch there is the other
    common cause of "it was fine in PyTorch".
    """
    from sar.ai.runtime import OnnxDetectorEngine, nms_numpy  # local: optional dep

    eng_ref = OnnxDetectorEngine(reference, input_hw=input_hw, conf=conf,
                                 channels=channels, warmup=1)
    eng_cand = OnnxDetectorEngine(candidate, input_hw=input_hw, conf=conf,
                                  channels=channels, warmup=1)
    matched = lost = gained = 0
    per_image: List[Dict[str, Any]] = []
    for i, img in enumerate(images):
        rb, rs, _ = eng_ref.infer(img)
        cb, cs, _ = eng_cand.infer(img)
        used = set()
        m = 0
        for j in range(len(rb)):
            best, best_iou = -1, 0.0
            for k in range(len(cb)):
                if k in used:
                    continue
                v = _iou(rb[j], cb[k])
                if v > best_iou:
                    best, best_iou = k, v
            if best >= 0 and best_iou >= iou_match:
                used.add(best)
                m += 1
        matched += m
        lost += len(rb) - m
        gained += len(cb) - len(used)
        per_image.append({"i": i, "reference": int(len(rb)), "candidate": int(len(cb)),
                          "matched": int(m)})
    total_ref = matched + lost
    return {
        "reference": str(reference), "candidate": str(candidate),
        "images": len(images), "matched": matched,
        "recall_lost": lost, "new_detections": gained,
        "recall_retained": (matched / total_ref) if total_ref else 1.0,
        "reference_latency": eng_ref.stats.to_dict(),
        "candidate_latency": eng_cand.stats.to_dict(),
        "per_image": per_image,
        "verdict": ("acceptable" if total_ref and matched / total_ref >= 0.97
                    else "REGRESSION - do not fly this quantisation"),
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ub = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    denom = ua + ub - inter
    return float(inter / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Synthetic model for CI and for `doctor`
# --------------------------------------------------------------------------- #
def make_synthetic_detector_onnx(out_path: str | Path, *,
                                 input_hw: Tuple[int, int] = (128, 128),
                                 channels: int = 1,
                                 boxes_xywh: Optional[Sequence[Sequence[float]]] = None,
                                 n_classes: int = 6,
                                 class_id: int = 0) -> Path:
    """Write a minimal but *valid* YOLO-style ONNX detector.

    The head is constant except for a brightness term taken from the input
    tensor, so a test can prove that preprocessing actually reached the network
    (a model that ignores its input cannot catch a broken letterbox).

    Output layout is ``(1, 4 + n_classes, N)`` - the modern Ultralytics export -
    with boxes in ``cxcywh`` pixels of the network input frame.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the `onnx` package is required to build the synthetic "
                           "model: pip install onnx") from exc

    h, w = input_hw
    if boxes_xywh is None:
        boxes_xywh = [[w * 0.5, h * 0.5, 12.0, 20.0],
                      [w * 0.25, h * 0.75, 10.0, 16.0]]
    n = len(boxes_xywh)
    box_arr = np.asarray(boxes_xywh, dtype=np.float32).T.reshape(1, 4, n)
    cls_arr = np.zeros((1, n_classes, n), dtype=np.float32)
    cls_arr[0, class_id, :] = 1.0        # scaled by mean input brightness below

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                        [1, channels, h, w])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                        [1, 4 + n_classes, n])
    # score = clip(3 * mean(input), 0.02, 0.98): a dark frame yields nothing,
    # a frame with signal in it yields the fixed boxes.  That asymmetry is what
    # makes the test able to distinguish "ran the model" from "ignored it".
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["brightness"], keepdims=0),
        helper.make_node("Mul", ["brightness", "gain_k"], ["scaled"]),
        helper.make_node("Clip", ["scaled", "lo", "hi"], ["gain"]),
        helper.make_node("Mul", ["cls_base", "gain"], ["cls_scores"]),
        helper.make_node("Concat", ["box_const", "cls_scores"], ["output0"], axis=1),
    ]
    initialisers = [
        numpy_helper.from_array(box_arr, "box_const"),
        numpy_helper.from_array(cls_arr, "cls_base"),
        numpy_helper.from_array(np.array(3.0, dtype=np.float32), "gain_k"),
        numpy_helper.from_array(np.array(0.02, dtype=np.float32), "lo"),
        numpy_helper.from_array(np.array(0.98, dtype=np.float32), "hi"),
    ]
    graph = helper.make_graph(nodes, "synthetic_sar_detector", [inp], [out],
                              initializer=initialisers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="sar.ai.export")
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def write_registry_entry(models_dir: str | Path, entry: Dict[str, Any]) -> Path:
    """Append/replace one model in ``models/registry.json`` (operator override)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "registry.json"
    data: Dict[str, Any] = {"models": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    models = [m for m in data.get("models", []) if m.get("name") != entry.get("name")]
    models.append(entry)
    data["models"] = models
    path.write_text(json.dumps(data, indent=2))
    return path


__all__ = ["export_ultralytics_to_onnx", "make_synthetic_detector_onnx",
           "validate_onnx_against_reference", "write_registry_entry"]
