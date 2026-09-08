"""On-device AI: model zoo, inference runtime, calibration, and backend wiring.

Why this package exists separately from :mod:`sar.perception`
-------------------------------------------------------------
``sar.perception`` owns the *physics* of finding a person from the air: the
geometry-aware aperture, radiometric class decision, cross-modal gating,
tracking, geo-tagging.  Those rules are auditable and run anywhere.

:mod:`sar.ai` owns the *learned* half: which network, in which numeric format,
on which accelerator, with which calibration, and how its raw confidences are
turned into probabilities that can be composed with search theory.  Keeping it
apart matters operationally - the aircraft must still fly a useful sortie when
the NPU is unavailable, the weights are missing, or the model was quantised into
uselessness, and that is only testable if the boundary is explicit.

The contract in both directions is one method::

    backend.detect(frame) -> List[sar.perception.detector.Detection]

so a neural backend and the heuristic backend are interchangeable inside
:class:`~sar.perception.detector.DetectorEnsemble`, and the mission code above
them cannot tell which is running.

Entry point
-----------
::

    from sar.ai import build_detector_stack

    detector = build_detector_stack("auto")   # neural if weights exist, else heuristic
    print(detector.stack_description)

See ``docs/06_AI_MODELS_AND_DATASETS.md`` for the model selection evidence, the
training recipe and the Qualcomm AI Hub export path.
"""

from sar.ai.registry import (MODEL_ZOO, ModelFormat, ModelRegistry, ModelSpec,
                             default_registry)
from sar.ai.runtime import (InferenceStats, OnnxDetectorEngine, available_providers,
                            letterbox, preprocess_lwir, select_providers)
from sar.ai.backends import NeuralDetectorBackend, ThermalNeuralBackend
from sar.ai.calibration import (ConfidenceCalibrator, expected_calibration_error,
                                fit_platt_scaling)
from sar.ai.stack import DetectorStack, build_detector_stack, describe_ai_environment

__all__ = [
    "MODEL_ZOO",
    "ModelFormat",
    "ModelRegistry",
    "ModelSpec",
    "default_registry",
    "InferenceStats",
    "OnnxDetectorEngine",
    "available_providers",
    "letterbox",
    "preprocess_lwir",
    "select_providers",
    "NeuralDetectorBackend",
    "ThermalNeuralBackend",
    "ConfidenceCalibrator",
    "expected_calibration_error",
    "fit_platt_scaling",
    "DetectorStack",
    "build_detector_stack",
    "describe_ai_environment",
]
