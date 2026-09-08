"""Choosing the detector stack: neural where it is real, heuristic where it is not.

One function, :func:`build_detector_stack`, decides what runs.  Everything else
in the project calls it and does not branch on availability itself - that is the
point.  The same call in the simulator, on the laptop and on the aircraft yields
the strongest configuration that host can honestly support, and *says so*, so a
sortie report can never be ambiguous about what produced its numbers.

Modes
-----
``auto`` (default)
    Neural backends for every modality with weights present and a working
    runtime; heuristic for the rest.  This is the intended flight setting.
``hybrid``
    Force both - neural *and* heuristic on the same modality, fused by the
    ensemble.  Highest recall and the slowest; used for evaluation runs where
    a missed survivor costs more than the extra 80 ms.
``heuristic``
    No networks at all.  Deterministic, ~90 fps on two CPU cores, and the
    configuration every committed artifact in this repo was produced with.
``neural``
    Networks only.  Fails loudly if weights are missing, because silently
    falling back would make a benchmark meaningless.

Why the heuristic path is not a placeholder
-------------------------------------------
It is the reference implementation and the fallback the aircraft actually needs.
Weights go missing, an NPU driver mismatches after an OS update, a quantised
model regresses on the day. The heuristic detector's decisions are auditable
line by line, its confidence is already calibrated against search theory, and it
needs no accelerator - so it is what keeps a sortie useful when the clever half
is unavailable.  A system whose only detector is a downloaded artifact is a
system with a single point of failure that nobody flight-tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sar.ai.calibration import ConfidenceCalibrator
from sar.ai.registry import ModelRegistry, ModelSpec, default_registry
from sar.ai.runtime import available_providers, select_providers
from sar.perception.detector import (DetectorEnsemble, RgbHazardDetector,
                                     ThermalAnomalyDetector)

log = logging.getLogger("sar.ai.stack")

VALID_MODES = ("auto", "hybrid", "heuristic", "neural")


@dataclass
class DetectorStack:
    """The ensemble plus a provenance record of how it was assembled."""

    ensemble: DetectorEnsemble
    mode: str
    backends: List[str] = field(default_factory=list)
    neural_models: List[Dict[str, Any]] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    calibration: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # The stack is usable anywhere a DetectorEnsemble is expected.
    def detect(self, frames: Any, rgb_quality: Optional[float] = None):
        return self.ensemble.detect(frames, rgb_quality=rgb_quality)

    @property
    def name(self) -> str:
        return f"stack:{self.mode}"

    @property
    def stack_description(self) -> str:
        parts = [f"mode={self.mode}", "backends=" + "+".join(self.backends)]
        if self.neural_models:
            parts.append("models=" + ",".join(m["name"] for m in self.neural_models))
        if self.providers:
            parts.append("ep=" + self.providers[0])
        if self.calibration:
            parts.append(f"calibration={self.calibration}")
        return "  ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "backends": self.backends,
                "neural_models": self.neural_models, "providers": self.providers,
                "calibration": self.calibration, "notes": self.notes}


def build_detector_stack(mode: str = "auto", *,
                         registry: Optional[ModelRegistry] = None,
                         calibration_path: Optional[str | Path] = None,
                         budget_ms: float = 125.0,
                         ensemble_kwargs: Optional[Dict[str, Any]] = None,
                         heuristic_kwargs: Optional[Dict[str, Any]] = None,
                         ) -> DetectorStack:
    """Assemble the detector stack for this host.  See module docstring."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    reg = registry or default_registry()
    ensemble_kwargs = dict(ensemble_kwargs or {})
    heuristic_kwargs = dict(heuristic_kwargs or {})
    notes: List[str] = []
    backends: List[Any] = []
    names: List[str] = []
    neural_models: List[Dict[str, Any]] = []

    calibration_path = Path(calibration_path or (reg.models_dir / "calibration.json"))
    calibrator = ConfidenceCalibrator.load_if_present(calibration_path)
    if calibrator is None and mode in ("auto", "neural", "hybrid"):
        notes.append(
            f"no calibration file at {calibration_path}; neural scores are shrunk "
            "conservatively rather than calibrated (see sar/ai/calibration.py)")

    want_neural = mode in ("auto", "neural", "hybrid")
    providers: List[str] = []
    if want_neural:
        providers = select_providers()
        for modality in ("lwir", "rgb"):
            spec = reg.best_for("person_detection", modality)
            if spec is None:
                notes.append(f"no {modality} weights available in {reg.models_dir}")
                continue
            backend = _try_neural(spec, calibrator, budget_ms, notes)
            if backend is not None:
                backends.append(backend)
                names.append(backend.name)
                neural_models.append({"name": spec.name, "modality": modality,
                                      "format": spec.fmt.value,
                                      "input_size": list(spec.input_size)})

    if mode == "neural" and not backends:
        raise RuntimeError(
            f"mode='neural' but no usable neural backend was loaded from "
            f"{reg.models_dir}. Run `python scripts/fetch_models.py --list`, or "
            f"use mode='auto' to fall back to the heuristic detector.")

    have_lwir = any(getattr(b, "modalities", ()) and "lwir" in b.modalities
                    for b in backends)
    have_rgb = any(getattr(b, "modalities", ()) and "rgb" in b.modalities
                   for b in backends)

    add_heuristic_lwir = mode in ("heuristic", "hybrid") or (mode == "auto" and not have_lwir)
    add_heuristic_rgb = mode in ("heuristic", "hybrid") or (mode == "auto" and not have_rgb)

    if add_heuristic_lwir:
        backends.append(ThermalAnomalyDetector(**heuristic_kwargs.get("lwir", {})))
        names.append("heuristic_lwir")
    if add_heuristic_rgb:
        # The RGB heuristic also carries hazard classes (fire, water, debris,
        # powerlines), which no person-detection network provides, so it stays
        # in the stack in hybrid/auto even when an RGB network is loaded.
        backends.append(RgbHazardDetector(**heuristic_kwargs.get("rgb", {})))
        names.append("heuristic_rgb")
    elif mode == "auto" and have_rgb:
        backends.append(RgbHazardDetector(**heuristic_kwargs.get("rgb", {})))
        names.append("heuristic_rgb(hazards)")
        notes.append("RGB network handles people; heuristic RGB kept for hazard classes")

    ensemble = DetectorEnsemble(backends, **ensemble_kwargs)
    stack = DetectorStack(
        ensemble=ensemble, mode=mode, backends=names,
        neural_models=neural_models, providers=providers,
        calibration=(str(calibration_path) if calibrator else None), notes=notes)
    log.info("detector stack: %s", stack.stack_description)
    for n in notes:
        log.info("  note: %s", n)
    return stack


def _try_neural(spec: ModelSpec, calibrator: Optional[ConfidenceCalibrator],
                budget_ms: float, notes: List[str]) -> Optional[Any]:
    """Load one neural backend, downgrading to None with a reason on failure."""
    from sar.ai.backends import NeuralDetectorBackend, ThermalNeuralBackend
    cls = ThermalNeuralBackend if spec.modality == "lwir" else NeuralDetectorBackend
    try:
        return cls(spec, calibrator=calibrator, budget_ms=budget_ms)
    except Exception as exc:            # missing runtime, bad file, wrong opset
        notes.append(f"{spec.name} not loaded: {type(exc).__name__}: {exc}")
        log.warning("neural backend %s unavailable: %r", spec.name, exc)
        return None


def describe_ai_environment(registry: Optional[ModelRegistry] = None) -> Dict[str, Any]:
    """What the AI layer can do on this host - printed by scripts/doctor.py and
    embedded in every sortie report so results are attributable."""
    reg = registry or default_registry()
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
    except Exception:
        ort_version = None
    return {
        "onnxruntime": ort_version,
        "providers_available": available_providers(),
        "providers_selected": select_providers() if ort_version else [],
        "models_dir": str(reg.models_dir),
        "models": [{"name": m.name, "modality": m.modality, "format": m.fmt.value,
                    "available": reg.is_available(m.name), "pretrained": m.pretrained}
                   for m in reg.models()],
        "calibration": (reg.models_dir / "calibration.json").is_file(),
    }


__all__ = ["DetectorStack", "VALID_MODES", "build_detector_stack",
           "describe_ai_environment"]
