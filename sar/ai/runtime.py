"""Inference runtime: one API over CPU, NPU and NVIDIA, with honest timing.

What this module guarantees to the rest of the stack
----------------------------------------------------
* **The same numbers everywhere.**  Preprocessing (letterbox, thermal AGC) and
  decode (anchor-free YOLO head, NMS) live here, not in the backend and not in
  the accelerator wrapper, so a detection produced on a laptop and one produced
  on the aircraft's NPU differ only by quantisation - which is measurable
  instead of mysterious.
* **A latency budget that is enforced, not hoped for.**  Every call records its
  own wall time; :class:`InferenceStats` reports mean/p50/p95 and how many
  cycles blew the budget.  The perception rate feeds the coverage planner's
  ground-speed calculation, so a model that is 3x slower than assumed does not
  quietly turn into 3x less coverage - it shows up as a deadline miss count in
  the sortie report.
* **Failure is a value, not an exception.**  Inference that throws returns an
  empty detection list and increments a fault counter.  A sortie over a flooded
  basin does not end because one frame tripped a shape assertion.

Thermal preprocessing is the part people get wrong
--------------------------------------------------
An LWIR frame is radiometric: 16-bit counts that map to temperature.  Feeding
it to a network trained on 8-bit imagery requires an automatic gain control
step, and the naive min-max version destroys exactly the signal we need - one
hot rock in frame compresses a 3 K human-to-ground contrast into two grey
levels.  :func:`preprocess_lwir` uses a robust percentile stretch with a floor
on the mapped span, and optionally a **physiology-anchored** window (23-40.5 C)
that keeps human-band contrast constant across frames.  Constant mapping across
frames also makes the network's confidence comparable across frames, which is
what :mod:`sar.ai.calibration` then relies on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("sar.ai.runtime")

#: Provider preference on each platform, most specific first.  onnxruntime
#: silently falls back to CPU for any provider it does not have, which is how a
#: "GPU deployment" ends up running at 3 fps without anyone noticing - so the
#: chosen list and the *actually active* list are both reported.
PROVIDER_PREFERENCE: Tuple[str, ...] = (
    "QNNExecutionProvider",        # Qualcomm Hexagon NPU (RB3/RB5, QCS6490/8550)
    "TensorrtExecutionProvider",   # Jetson
    "CUDAExecutionProvider",       # discrete NVIDIA
    "CoreMLExecutionProvider",     # macOS dev machines
    "XnnpackExecutionProvider",    # ARM CPU, still much faster than plain CPU EP
    "CPUExecutionProvider",
)

#: Human skin/clothing apparent-temperature band used by the radiometric
#: anchored stretch.  Same numbers as sar.perception.thermal, restated here so
#: the runtime has no import cycle into perception.
HUMAN_BAND_C: Tuple[float, float] = (23.0, 40.5)


def available_providers() -> List[str]:
    """Providers onnxruntime reports in this process (empty if not installed)."""
    try:
        import onnxruntime as ort
    except Exception:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return []


def select_providers(preferred: Sequence[str] = PROVIDER_PREFERENCE) -> List[str]:
    """Intersect our preference order with what this host actually offers."""
    have = set(available_providers())
    chosen = [p for p in preferred if p in have]
    return chosen or ["CPUExecutionProvider"]


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, out_hw: Tuple[int, int],
              pad_value: float = 114.0) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize preserving aspect ratio and pad to ``out_hw``.

    Returns ``(canvas, scale, (pad_x, pad_y))`` so boxes can be mapped back
    exactly.  Getting the inverse transform wrong is the classic cause of
    detections that are right in the letterboxed frame and 20 px off in the
    original - which, projected to the ground from 50 m, is a 6 m geotag error
    that looks like a navigation problem.
    """
    h, w = img.shape[:2]
    oh, ow = out_hw
    scale = min(oh / max(h, 1), ow / max(w, 1))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = _resize(img, (nh, nw))
    if img.ndim == 3:
        canvas = np.full((oh, ow, img.shape[2]), pad_value, dtype=np.float32)
    else:
        canvas = np.full((oh, ow), pad_value, dtype=np.float32)
    pad_y, pad_x = (oh - nh) // 2, (ow - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, float(scale), (pad_x, pad_y)


def _resize(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize with no hard OpenCV dependency."""
    h, w = img.shape[:2]
    nh, nw = out_hw
    if (h, w) == (nh, nw):
        return img.astype(np.float32)
    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (nw, nh),
                          interpolation=cv2.INTER_LINEAR)
    except Exception:
        pass
    try:
        from scipy import ndimage as ndi
        zoom = (nh / h, nw / w) + ((1.0,) if img.ndim == 3 else ())
        return ndi.zoom(img.astype(np.float32), zoom, order=1)
    except Exception:  # pragma: no cover - numpy nearest-neighbour fallback
        yi = (np.arange(nh) * h / nh).astype(int).clip(0, h - 1)
        xi = (np.arange(nw) * w / nw).astype(int).clip(0, w - 1)
        return img.astype(np.float32)[yi][:, xi]


def preprocess_lwir(frame_c: np.ndarray, *, mode: str = "anchored",
                    band: Tuple[float, float] = HUMAN_BAND_C,
                    low_pct: float = 1.0, high_pct: float = 99.5,
                    min_span_c: float = 6.0) -> np.ndarray:
    """Radiometric LWIR (degrees C) -> 8-bit-equivalent float image in 0..255.

    ``mode='anchored'`` maps a fixed window around the human thermal band, with
    the window recentred on the scene median so a 5 C morning and a 40 C
    afternoon both present a person at similar grey level.  This is what makes
    network confidence comparable frame to frame.

    ``mode='percentile'`` is the conventional robust stretch; it adapts better
    to scenes with no people in them and is the right choice for hazard
    classes, but its output level for a given body temperature depends on what
    else is in frame.
    """
    x = np.asarray(frame_c, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    if mode == "anchored":
        med = float(np.median(finite))
        lo = min(band[0], med - 3.0)
        hi = max(band[1], med + 3.0)
    else:
        lo = float(np.percentile(finite, low_pct))
        hi = float(np.percentile(finite, high_pct))
    if hi - lo < min_span_c:
        centre = 0.5 * (hi + lo)
        lo, hi = centre - min_span_c / 2, centre + min_span_c / 2
    out = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0) * 255.0


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@dataclass
class InferenceStats:
    """Latency accounting.  Reported in every sortie so claims are checkable."""

    budget_ms: float = 125.0
    samples: List[float] = field(default_factory=list)
    faults: int = 0
    keep: int = 512

    def record(self, ms: float) -> None:
        self.samples.append(float(ms))
        if len(self.samples) > self.keep:
            del self.samples[: len(self.samples) - self.keep]

    @property
    def n(self) -> int:
        return len(self.samples)

    def _pct(self, q: float) -> float:
        return float(np.percentile(self.samples, q)) if self.samples else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        return self._pct(95)

    @property
    def deadline_misses(self) -> int:
        return int(sum(1 for s in self.samples if s > self.budget_ms))

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "mean_ms": round(self.mean_ms, 2),
                "p50_ms": round(self._pct(50), 2), "p95_ms": round(self.p95_ms, 2),
                "max_ms": round(max(self.samples), 2) if self.samples else 0.0,
                "budget_ms": self.budget_ms, "deadline_misses": self.deadline_misses,
                "faults": self.faults}


# --------------------------------------------------------------------------- #
# ONNX Runtime engine
# --------------------------------------------------------------------------- #
class OnnxDetectorEngine:
    """Load an anchor-free YOLO ONNX export and run it on the best provider.

    Supports the two head layouts Ultralytics emits:

    * ``(1, 4 + nc, N)``  - the modern export (YOLOv8/YOLO11), transposed here;
    * ``(1, N, 4 + nc)``  - the same after ``--nms=False --simplify`` variants.

    End-to-end exports that already include NMS (``(1, N, 6)``: xyxy, conf, cls)
    are detected and passed through.
    """

    def __init__(self, model_path: str | Path, *, input_hw: Tuple[int, int] = (640, 640),
                 conf: float = 0.25, iou: float = 0.45, channels: int = 3,
                 providers: Optional[Sequence[str]] = None,
                 num_classes: Optional[int] = None,
                 budget_ms: float = 125.0, warmup: int = 2) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised only without ORT
            raise RuntimeError(
                "onnxruntime is not installed. `pip install -r requirements-ai.txt` "
                "or use the heuristic backend (sar.ai.build_detector_stack('heuristic'))."
            ) from exc

        self.model_path = str(model_path)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"model weights not found: {self.model_path}. "
                "Run `python scripts/fetch_models.py --list` to see what is expected.")

        self.requested_providers = list(providers or select_providers())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # One inference thread by default: the companion computer runs the
        # control loop too, and an inference that steals every core turns a
        # 2 Hz perception rate into a jittery 0.7 Hz.
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(self.model_path, sess_options=opts,
                                            providers=self.requested_providers)
        self.active_providers = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Trust the graph over the caller where the graph is concrete.
        if len(shape) == 4 and all(isinstance(s, int) for s in shape[2:]):
            input_hw = (int(shape[2]), int(shape[3]))
        if len(shape) == 4 and isinstance(shape[1], int):
            channels = int(shape[1])
        self.input_hw = input_hw
        self.channels = channels
        self.conf, self.iou = float(conf), float(iou)
        #: Number of classes the head predicts.  Optional, but supplying it
        #: removes the only ambiguity in decode: with an (A, B) output we must
        #: know which axis is "predictions" and which is "4 + classes", and the
        #: usual `A < B` guess is wrong whenever the image contains few anchors
        #: (small inputs, end-to-end exports, unit tests).
        self.num_classes = int(num_classes) if num_classes else None
        self.stats = InferenceStats(budget_ms=budget_ms)

        if self.active_providers[:1] != self.requested_providers[:1]:
            log.warning("provider fallback: requested %s, active %s",
                        self.requested_providers, self.active_providers)
        for _ in range(max(0, warmup)):
            self._warm()

    # ------------------------------------------------------------------ #
    def _warm(self) -> None:
        blank = np.zeros((1, self.channels, *self.input_hw), dtype=np.float32)
        try:
            self.session.run(None, {self.input_name: blank})
        except Exception as exc:  # pragma: no cover
            log.warning("warmup failed: %r", exc)

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one image.  Returns ``(boxes_xyxy, scores, class_ids)`` in *image* pixels.

        ``img`` is HxW (single channel, already AGC'd to 0..255) or HxWx3.
        """
        t0 = time.perf_counter()
        try:
            tensor, scale, pad = self._prepare(img)
            raw = self.session.run(None, {self.input_name: tensor})[0]
            boxes, scores, cls = self._decode(np.asarray(raw))
            boxes = self._unletterbox(boxes, scale, pad, img.shape[:2])
            keep = nms_numpy(boxes, scores, self.iou)
            out = boxes[keep], scores[keep], cls[keep]
        except Exception as exc:
            self.stats.faults += 1
            log.warning("inference fault (%d total): %r", self.stats.faults, exc)
            out = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                   np.zeros((0,), np.int32))
        finally:
            self.stats.record((time.perf_counter() - t0) * 1000.0)
        return out

    # ------------------------------------------------------------------ #
    def _prepare(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        x = np.asarray(img, dtype=np.float32)
        if self.channels == 3 and x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        elif self.channels == 1 and x.ndim == 3:
            x = x.mean(axis=-1)
        canvas, scale, pad = letterbox(x, self.input_hw)
        if canvas.ndim == 2:
            canvas = canvas[..., None]
        tensor = canvas.transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor, dtype=np.float32), scale, pad

    def _decode(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = np.squeeze(raw)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))
        # End-to-end export with NMS already applied: (N, 6) xyxy conf cls
        if preds.shape[1] == 6 and preds.shape[0] != 6:
            boxes = preds[:, :4].astype(np.float32)
            scores = preds[:, 4].astype(np.float32)
            cls = preds[:, 5].astype(np.int32)
            keep = scores >= self.conf
            return boxes[keep], scores[keep], cls[keep]
        # Raw head: (4+nc, N) or (N, 4+nc)
        preds = self._orient(preds)
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        if class_scores.size == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))
        cls = class_scores.argmax(axis=1).astype(np.int32)
        scores = class_scores.max(axis=1).astype(np.float32)
        keep = scores >= self.conf
        boxes_cxcywh, scores, cls = boxes_cxcywh[keep], scores[keep], cls[keep]
        cx, cy, w, h = boxes_cxcywh.T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                         axis=1).astype(np.float32)
        return boxes, scores, cls

    def _orient(self, preds: np.ndarray) -> np.ndarray:
        """Return predictions as (N, 4 + nc), whichever way round they arrived."""
        if self.num_classes:
            head = 4 + self.num_classes
            if preds.shape[1] == head:
                return preds
            if preds.shape[0] == head:
                return preds.T
        return preds.T if preds.shape[0] < preds.shape[1] else preds

    @staticmethod
    def _unletterbox(boxes: np.ndarray, scale: float, pad: Tuple[int, int],
                     orig_hw: Tuple[int, int]) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        px, py = pad
        out = boxes.copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - px) / max(scale, 1e-9)
        out[:, [1, 3]] = (out[:, [1, 3]] - py) / max(scale, 1e-9)
        h, w = orig_hw
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
        return out

    def describe(self) -> Dict[str, Any]:
        return {"model": self.model_path, "input_hw": list(self.input_hw),
                "channels": self.channels, "conf": self.conf, "iou": self.iou,
                "requested_providers": self.requested_providers,
                "active_providers": self.active_providers,
                "latency": self.stats.to_dict()}


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS.  Pure numpy so it runs identically wherever the model does."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(xx1 - xx0, 0) * np.maximum(yy1 - yy0, 0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


__all__ = ["HUMAN_BAND_C", "InferenceStats", "OnnxDetectorEngine",
           "PROVIDER_PREFERENCE", "available_providers", "letterbox",
           "nms_numpy", "preprocess_lwir", "select_providers"]
