"""Neural detector backends that satisfy the same contract as the heuristic one.

A backend takes one frame and returns
:class:`sar.perception.detector.Detection` objects.  That is the whole
interface, and it is why the ensemble, the fuser, the tracker, the geo-tagger
and the mission runner do not know or care whether a network was involved.

Three things happen here that a bare YOLO wrapper does not do, and each exists
because of a specific failure seen in aerial SAR:

1. **Geometry veto.**  The aircraft knows its ground sample distance.  A
   "person" whose box implies a 4 m tall body, or a 0.2 m one, is not a person
   regardless of what the network scored it.  Rejecting on projected extent
   removes the dominant false-alarm class (rooftop furniture, vehicle bonnets)
   at zero inference cost, and it is the same rule the heuristic backend uses,
   so precision does not depend on which backend is loaded.

2. **Radiometry attached to thermal detections.**  The network gives a box; the
   physics gives peak/mean/background apparent temperature inside that box.
   Downstream, ``sar.perception.thermal`` needs those to run the physiology
   model and produce a triage clock, and ``fusion`` needs them for the
   cross-modal gate.  A detection without radiometry silently degrades triage
   to a constant, so the backend measures it even though the network did not.

3. **Calibrated confidence.**  A softmax/objectness score is not a probability
   of detection.  Search theory composes probabilities
   (``1 - Prod(1 - p_i)``), so an uncalibrated score corrupts the coverage map -
   it does not merely mislabel a box.  The optional
   :class:`~sar.ai.calibration.ConfidenceCalibrator` maps raw scores through a
   fitted logistic before they leave the backend.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.ai.calibration import ConfidenceCalibrator
from sar.ai.registry import ModelSpec, default_registry
from sar.ai.runtime import OnnxDetectorEngine, preprocess_lwir
from sar.perception.detector import (EXTENT_RATIO_MAX, HUMAN_APPARENT_AREA_M2,
                                     PERSON_LABELS, Detection)

log = logging.getLogger("sar.ai.backends")

#: A standing adult is ~1.75 m tall and ~0.5 m wide; prone, the long axis is
#: still ~1.75 m.  Anything whose *larger* ground dimension is outside this
#: window is not a human body.  Wide, because posture and thermal bloom both
#: distort the apparent extent, and a missed survivor costs more than a decoy.
HUMAN_EXTENT_M: Tuple[float, float] = (0.35, 3.2)


class NeuralDetectorBackend:
    """ONNX detector for one modality, wrapped to the project's Detection type."""

    def __init__(self, spec: ModelSpec, *, model_path: Optional[str] = None,
                 engine: Optional[Any] = None,
                 calibrator: Optional[ConfidenceCalibrator] = None,
                 enforce_geometry: bool = True,
                 budget_ms: float = 125.0) -> None:
        self.spec = spec
        self.name = f"neural:{spec.name}"
        self.modalities: Tuple[str, ...] = tuple(spec.modality.split("+"))
        self.enforce_geometry = enforce_geometry
        self.calibrator = calibrator
        self._class_names = list(spec.classes)
        if engine is not None:
            self.engine = engine            # injected: tests and QNN target
        else:
            path = model_path or str(default_registry().path(spec.name))
            self.engine = OnnxDetectorEngine(
                path, input_hw=spec.input_size, conf=spec.conf_threshold,
                iou=spec.iou_threshold, channels=spec.channels,
                num_classes=len(spec.classes), budget_ms=budget_ms)

    # ------------------------------------------------------------------ #
    def detect(self, frame: Any) -> List[Detection]:
        img = np.asarray(getattr(frame, "image", frame))
        kind = str(getattr(frame, "kind", self.modalities[0]))
        if kind not in self.modalities:
            return []
        gsd = float(getattr(frame, "gsd_m", 0.0) or 0.0)
        prepared = self._prepare_for_network(img, kind)
        boxes, scores, class_ids = self.engine.infer(prepared)

        dets: List[Detection] = []
        for box, score, cid in zip(boxes, scores, class_ids):
            label = self._label(int(cid))
            if label is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            w_px, h_px = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
            if w_px < 1.0 or h_px < 1.0:
                continue
            if (self.enforce_geometry and label in PERSON_LABELS
                    and not self._plausible_human(w_px, h_px, gsd)):
                continue
            det = Detection(
                label=label,
                score=self._calibrate(float(score), label, w_px * h_px, gsd),
                bbox=(x0, y0, x1, y1),
                modality=kind,
                frame_t=float(getattr(frame, "t", 0.0)),
                centroid=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                area_px=w_px * h_px,
                polarity="n/a",
                aspect=float(w_px / max(h_px, 1e-6)),
                elongation=float(max(w_px, h_px) / max(min(w_px, h_px), 1e-6)),
                scale_px=float(math.hypot(w_px, h_px) / 2.0),
                backend=self.name,
                attributes={"stage": "neural", "model": self.spec.name,
                            "raw_score": float(score), "gsd_m": gsd},
            )
            if kind == "lwir":
                self._attach_radiometry(det, img)
            dets.append(det)
        return dets

    # ------------------------------------------------------------------ #
    def _prepare_for_network(self, img: np.ndarray, kind: str) -> np.ndarray:
        if kind == "lwir":
            # Radiometric counts in degrees C -> stable 0..255 via anchored AGC.
            return preprocess_lwir(img, mode="anchored")
        x = np.asarray(img, dtype=np.float32)
        if x.max() <= 1.5:      # some renderers hand back 0..1 RGB
            x = x * 255.0
        return x

    def _label(self, class_id: int) -> Optional[str]:
        if class_id < 0 or class_id >= len(self._class_names):
            return None
        raw = self._class_names[class_id]
        return self.spec.class_map.get(raw)

    @staticmethod
    def _plausible_human(w_px: float, h_px: float, gsd_m: float) -> bool:
        """Reject boxes whose implied ground extent is not a human body."""
        if gsd_m <= 0:
            return True             # no geometry available: do not guess
        long_m = max(w_px, h_px) * gsd_m
        short_m = max(min(w_px, h_px) * gsd_m, 1e-6)
        if not (HUMAN_EXTENT_M[0] <= long_m <= HUMAN_EXTENT_M[1]):
            return False
        return (long_m / short_m) <= EXTENT_RATIO_MAX

    def _calibrate(self, raw: float, label: str, area_px: float, gsd_m: float) -> float:
        if self.calibrator is not None:
            return float(self.calibrator.apply(raw, label=label,
                                               area_px=area_px, gsd_m=gsd_m))
        # Without a fitted calibrator, apply a conservative shrink toward the
        # prior: an uncalibrated network score is usually over-confident on
        # small objects, which is exactly our regime.  This is deliberately mild
        # and is replaced the moment a calibration file exists.
        n_equiv = math.sqrt(max(area_px, 1.0))
        small_penalty = float(np.clip(n_equiv / 8.0, 0.55, 1.0))
        return float(np.clip(raw * small_penalty, 0.0, 1.0))

    @staticmethod
    def _attach_radiometry(det: Detection, thermal_c: np.ndarray) -> None:
        """Measure apparent temperature inside the box and in an annulus around it."""
        h, w = thermal_c.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in det.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
        core = thermal_c[y0:y1, x0:x1]
        if core.size == 0:
            return
        pad = max(4, int(0.75 * max(x1 - x0, y1 - y0)))
        ay0, ay1 = max(0, y0 - pad), min(h, y1 + pad)
        ax0, ax1 = max(0, x0 - pad), min(w, x1 + pad)
        ring = thermal_c[ay0:ay1, ax0:ax1].copy().astype(np.float32)
        ring[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] = np.nan
        bg = float(np.nanmedian(ring)) if np.isfinite(ring).any() else float(np.median(core))
        det.peak_temp_c = float(np.nanmax(core))
        det.mean_temp_c = float(np.nanmean(core))
        det.background_temp_c = bg
        sigma = float(np.nanstd(ring)) if np.isfinite(ring).any() else 1.0
        det.contrast_k = float((det.peak_temp_c - bg) / max(sigma, 0.25))
        det.polarity = "hot" if det.peak_temp_c >= bg else "cold"
        det.attributes["apparent_area_m2"] = HUMAN_APPARENT_AREA_M2

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        d = {"backend": self.name, "spec": self.spec.to_dict()}
        if hasattr(self.engine, "describe"):
            d["engine"] = self.engine.describe()
        return d


class ThermalNeuralBackend(NeuralDetectorBackend):
    """LWIR-only neural backend with the radiometric class gate still applied.

    The network is allowed to propose; physics disposes.  A detection whose
    measured apparent temperature sits outside the plausible human band *and*
    whose contrast is weak is demoted rather than deleted - demoted, because at
    110 m AGL sub-pixel averaging legitimately drags a survivor's apparent
    temperature toward the background (measured in
    ``artifacts/subpixel_radiometry.json``), so a hard band veto at altitude
    would delete real people.
    """

    def __init__(self, *args: Any, band_c: Tuple[float, float] = (23.0, 40.5),
                 demote_factor: float = 0.55, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.band_c = band_c
        self.demote_factor = demote_factor
        self.name = f"neural_lwir:{self.spec.name}"
        self.modalities = ("lwir",)

    def detect(self, frame: Any) -> List[Detection]:
        dets = super().detect(frame)
        for d in dets:
            if d.label not in PERSON_LABELS or d.peak_temp_c is None:
                continue
            in_band = self.band_c[0] <= d.peak_temp_c <= self.band_c[1]
            strong = d.contrast_k >= 3.0
            if not in_band and not strong:
                d.score *= self.demote_factor
                d.attributes["radiometric_gate"] = "demoted_out_of_band"
            elif in_band:
                d.attributes["radiometric_gate"] = "in_human_band"
        return [d for d in dets if d.score >= 0.05]


__all__ = ["HUMAN_EXTENT_M", "NeuralDetectorBackend", "ThermalNeuralBackend"]
