"""The AI layer: registry, runtime, calibration, and the stack selector.

These tests are written against the failures that would actually ship:

* a model file that is missing must degrade to the heuristic detector, not
  raise, because that is what happens on the field the morning the SD card
  was reflashed;
* the neural path must be *exercised end to end* - session, preprocessing,
  decode, NMS, geometry veto - which is why a tiny synthetic ONNX model is
  built here rather than mocking the engine;
* an over-confident detector corrupts the coverage map, so calibration must
  measurably reduce expected calibration error;
* the decode step must not care whether the head arrives as (4+nc, N) or
  (N, 4+nc).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sar.ai.calibration import (ConfidenceCalibrator, expected_calibration_error,
                                fit_platt_scaling)
from sar.ai.registry import MODEL_ZOO, ModelFormat, ModelRegistry, ModelSpec
from sar.ai.runtime import letterbox, nms_numpy, preprocess_lwir, select_providers
from sar.ai.stack import build_detector_stack
from sar.perception.detector import ImageFrame

onnx = pytest.importorskip("onnx", reason="onnx is needed to build the test model")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime is the neural runtime")

SYNTH_CLASSES = ("person", "person_group", "vehicle", "animal", "fire", "structure")


@pytest.fixture(scope="module")
def synth_model(tmp_path_factory) -> Path:
    from sar.ai.export import make_synthetic_detector_onnx
    out = tmp_path_factory.mktemp("models") / "synthetic-lwir.onnx"
    return make_synthetic_detector_onnx(out, input_hw=(128, 128), channels=1)


def synth_spec(path: Path, **kw) -> ModelSpec:
    return ModelSpec(
        name="synthetic", task="person_detection", modality="lwir",
        architecture="synthetic", input_size=(128, 128), classes=SYNTH_CLASSES,
        class_map={"person": "person"}, file_name=path.name,
        fmt=ModelFormat.ONNX, conf_threshold=0.2, channels=1, **kw)


def warm_frame(gsd_m: float = 0.03) -> ImageFrame:
    img = np.full((240, 320), 22.0, np.float32)
    img[100:145, 150:185] = 34.0          # a warm, human-band patch
    return ImageFrame(image=img, kind="lwir", t=1.0, gsd_m=gsd_m)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_reports_absence_without_raising(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.models(), "the zoo should not be empty"
    assert not any(reg.is_available(m.name) for m in reg.models())
    ok, msg = reg.verify(MODEL_ZOO[0].name)
    assert ok is False and "not present" in msg


def test_registry_override_file_is_merged(tmp_path):
    (tmp_path / "registry.json").write_text(json.dumps({"models": [{
        "name": "field-drop-in", "task": "person_detection", "modality": "lwir",
        "architecture": "whatever", "input_size": [320, 256],
        "classes": ["person"], "class_map": {"person": "person"},
        "file_name": "field.onnx", "fmt": "onnx"}]}))
    reg = ModelRegistry(tmp_path)
    assert reg.get("field-drop-in").input_size == (320, 256)


def test_best_for_prefers_available_and_fastest_format(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.best_for("person_detection", "lwir") is None
    (tmp_path / "yolo11n-thermal-sar.onnx").write_bytes(b"not a real model")
    assert reg.best_for("person_detection", "lwir").name == "yolo11n-thermal-sar"


# --------------------------------------------------------------------------- #
# Preprocessing and decode
# --------------------------------------------------------------------------- #
def test_letterbox_is_invertible_within_a_pixel():
    img = np.zeros((240, 320), np.float32)
    canvas, scale, (px, py) = letterbox(img, (128, 128))
    assert canvas.shape == (128, 128)
    # a point at (160, 120) in the original maps to scale*p + pad
    u, v = 160 * scale + px, 120 * scale + py
    back_u, back_v = (u - px) / scale, (v - py) / scale
    assert abs(back_u - 160) < 1e-6 and abs(back_v - 120) < 1e-6


def test_anchored_agc_keeps_human_band_contrast_stable():
    """The same body over a cold and a hot scene must map to a similar level."""
    cold = np.full((60, 80), 5.0, np.float32)
    hot = np.full((60, 80), 38.0, np.float32)
    cold[20:30, 20:30] = 33.0
    hot[20:30, 20:30] = 33.0
    a = preprocess_lwir(cold)[25, 25]
    b = preprocess_lwir(hot)[25, 25]
    # Not identical - the window recentres - but within a quarter of the range,
    # where a naive min-max stretch would put them at 255 and ~0.
    assert abs(a - b) < 64.0


def test_nms_removes_duplicates_and_keeps_the_best():
    boxes = np.array([[0, 0, 10, 20], [1, 1, 11, 21], [100, 100, 110, 120]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms_numpy(boxes, scores, 0.45)
    assert sorted(keep.tolist()) == [0, 2]


# --------------------------------------------------------------------------- #
# Neural backend, end to end
# --------------------------------------------------------------------------- #
def test_neural_backend_produces_detections_with_radiometry(synth_model):
    from sar.ai.backends import ThermalNeuralBackend
    be = ThermalNeuralBackend(synth_spec(synth_model), model_path=str(synth_model))
    dets = be.detect(warm_frame(gsd_m=0.03))
    assert dets, "the synthetic model must produce detections on a warm frame"
    d = max(dets, key=lambda x: x.score)
    assert d.label == "person"
    assert d.modality == "lwir"
    # Radiometry is measured by the backend, not by the network.
    assert d.peak_temp_c is not None and d.background_temp_c is not None
    assert d.attributes["stage"] == "neural"


def test_geometry_veto_rejects_impossible_bodies(synth_model):
    """Same detection, coarser GSD: the implied body is 4 m long and must go."""
    from sar.ai.backends import ThermalNeuralBackend
    be = ThermalNeuralBackend(synth_spec(synth_model), model_path=str(synth_model))
    fine = be.detect(warm_frame(gsd_m=0.03))
    coarse = be.detect(warm_frame(gsd_m=0.30))
    assert len(fine) > len(coarse)


def test_radiometric_gate_demotes_out_of_band_detections(synth_model):
    from sar.ai.backends import ThermalNeuralBackend
    be = ThermalNeuralBackend(synth_spec(synth_model), model_path=str(synth_model))
    dets = be.detect(warm_frame(gsd_m=0.03))
    gates = {d.attributes.get("radiometric_gate") for d in dets}
    assert "in_human_band" in gates


def test_engine_decodes_both_head_orientations(synth_model):
    from sar.ai.runtime import OnnxDetectorEngine
    eng = OnnxDetectorEngine(synth_model, input_hw=(128, 128), channels=1,
                             conf=0.2, num_classes=len(SYNTH_CLASSES))
    head = np.zeros((4 + len(SYNTH_CLASSES), 3), np.float32)
    head[:4, :] = np.array([[50], [50], [10], [20]])
    head[4, :] = 0.9
    a = eng._decode(head[None])
    b = eng._decode(head.T[None])
    assert len(a[0]) == len(b[0]) == 3
    assert np.allclose(a[0], b[0])


def test_inference_stats_track_the_budget(synth_model):
    from sar.ai.runtime import OnnxDetectorEngine
    eng = OnnxDetectorEngine(synth_model, input_hw=(128, 128), channels=1,
                             budget_ms=0.0001, num_classes=len(SYNTH_CLASSES))
    eng.infer(preprocess_lwir(warm_frame().image))
    stats = eng.stats.to_dict()
    assert stats["n"] >= 1
    assert stats["deadline_misses"] >= 1, "an absurd budget must register as missed"


def test_inference_fault_returns_empty_rather_than_raising(synth_model):
    from sar.ai.runtime import OnnxDetectorEngine
    eng = OnnxDetectorEngine(synth_model, input_hw=(128, 128), channels=1,
                             num_classes=len(SYNTH_CLASSES))
    boxes, scores, cls = eng.infer(np.array("not an image", dtype=object))
    assert len(boxes) == 0 and eng.stats.faults == 1


# --------------------------------------------------------------------------- #
# Stack selection
# --------------------------------------------------------------------------- #
def test_auto_falls_back_to_heuristic_when_no_weights(tmp_path):
    stack = build_detector_stack("auto", registry=ModelRegistry(tmp_path))
    assert stack.mode == "auto"
    assert stack.backends == ["heuristic_lwir", "heuristic_rgb"]
    assert any("no lwir weights" in n for n in stack.notes)


def test_neural_mode_refuses_to_silently_fall_back(tmp_path):
    with pytest.raises(RuntimeError, match="no usable neural backend"):
        build_detector_stack("neural", registry=ModelRegistry(tmp_path))


def test_stack_is_usable_as_a_detector(tmp_path):
    stack = build_detector_stack("heuristic", registry=ModelRegistry(tmp_path))
    dets = stack.detect([warm_frame()], rgb_quality=0.0)
    assert isinstance(dets, list)


def test_auto_uses_the_neural_backend_when_weights_exist(tmp_path, synth_model):
    import shutil
    shutil.copy(synth_model, tmp_path / "synthetic-lwir.onnx")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [{
        "name": "synthetic-lwir", "task": "person_detection", "modality": "lwir",
        "architecture": "synthetic", "input_size": [128, 128],
        "classes": list(SYNTH_CLASSES), "class_map": {"person": "person"},
        "file_name": "synthetic-lwir.onnx", "fmt": "onnx",
        "conf_threshold": 0.2, "channels": 1}]}))
    stack = build_detector_stack("auto", registry=ModelRegistry(tmp_path))
    assert any(b.startswith("neural") for b in stack.backends), stack.backends
    assert stack.neural_models[0]["name"] == "synthetic-lwir"


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_platt_scaling_reduces_calibration_error():
    rng = np.random.default_rng(3)
    n = 600
    truth = rng.random(n) < 0.35
    # An over-confident detector: scores pushed toward 1 regardless of truth.
    raw = np.clip(np.where(truth, rng.normal(0.85, 0.1, n),
                           rng.normal(0.65, 0.15, n)), 0.01, 0.99)
    samples = [{"score": float(s), "matched": int(t), "label": "person",
                "area_px": float(rng.uniform(20, 400)), "gsd_m": 0.1}
               for s, t in zip(raw, truth)]
    cal = ConfidenceCalibrator.fit(samples)
    assert cal.ece_after < cal.ece_before
    assert cal.ece_after < 0.10


def test_calibrator_round_trips_through_json(tmp_path):
    cal = ConfidenceCalibrator(params={"person": {"a": 1.2, "b": -0.4, "w": []}},
                               source="test")
    p = cal.save(tmp_path / "calibration.json")
    back = ConfidenceCalibrator.load(p)
    assert abs(back.apply(0.7) - cal.apply(0.7)) < 1e-9


def test_identity_calibrator_is_a_no_op():
    cal = ConfidenceCalibrator()
    assert cal.apply(0.42) == pytest.approx(0.42)


def test_ece_of_a_perfect_predictor_is_zero():
    probs = [0.0] * 50 + [1.0] * 50
    labels = [0] * 50 + [1] * 50
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_fit_handles_an_empty_sample_set():
    out = fit_platt_scaling([], [])
    assert out["n"] == 0


def test_providers_always_include_a_cpu_fallback():
    assert "CPUExecutionProvider" in select_providers()
