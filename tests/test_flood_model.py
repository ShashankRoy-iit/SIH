"""FloodDetector behaviour: gates, context, fusion, and temporal memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sar.ai.flood import (FloodDetector, TemporalFilter, flood_context,
                          geometry_gate, glint_test, structure_edge_score,
                          water_mask)
from sar.perception.detector import Detection, ImageFrame


def test_geometry_gate_accepts_body_rejects_speck_and_roof():
    gsd = 0.12  # 50 m class; expected body ~43 px
    assert geometry_gate(gsd, 43.0)[0] is True
    assert geometry_gate(gsd, 2.0)[0] is False      # hot speck
    assert geometry_gate(gsd, 400.0)[0] is False    # roof / vehicle
    assert geometry_gate(gsd, 43.0, extent_ratio=9.0)[0] is False  # embedded
    assert geometry_gate(0.0, 1.0)[0] is True       # unknown GSD passes


def test_water_mask_finds_smooth_blue_dark():
    rgb = np.full((60, 60, 3), (105, 100, 92), np.uint8)
    rgb[30:, :] = (78, 90, 104)
    wm = water_mask(rgb)
    assert wm[40, 30] and not wm[10, 10]


def test_flood_context_verdicts():
    water = np.full((60, 60, 3), (78, 90, 104), np.uint8)
    assert flood_context(water, 30, 30)["verdict"] == "water"
    roof = np.full((60, 60, 3), (165, 155, 145), np.uint8)
    assert flood_context(roof, 30, 30)["verdict"] == "roof"
    mud = np.full((60, 60, 3), (105, 100, 92), np.uint8)
    assert flood_context(mud, 30, 30)["verdict"] == "ground"


def test_structure_edge_flags_roof_outline_not_clothing():
    rgb = np.full((120, 120, 3), (108, 102, 94), np.uint8)
    rgb[40:80, 40:80] = (150, 142, 132)  # roof slab with long straight edges
    assert structure_edge_score(rgb, 40, 60)["on_structure"] is True
    plain = np.full((120, 120, 3), (108, 102, 94), np.uint8)
    assert structure_edge_score(plain, 60, 60)["on_structure"] is False


def test_glint_flicker_vs_stable():
    assert glint_test([3.0, 3.1, 2.9, 3.0])[0] is False
    assert glint_test([0.5])[0] is False  # too short: never veto
    is_glint, _ = glint_test([4.0, 0.3, 4.2, 0.2, 3.9, 0.3])
    assert is_glint is True


def _shaded_patch(rgb, y0, y1, x0, x1, color=(205, 70, 55)):
    """Paint clothing with a lit centre (flat plateaus peak at corners)."""
    py, px = np.mgrid[0:y1 - y0, 0:x1 - x0]
    cy, cx = (y1 - y0 - 1) / 2.0, (x1 - x0 - 1) / 2.0
    rr = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    shade = 1.0 - 0.18 * np.clip(rr / max(max(cx, cy), 1), 0, 1)
    rgb[y0:y1, x0:x1] = (np.asarray(color, np.float32)
                         * shade[..., None]).astype(np.uint8)


def _person(u, v, score=0.8, contrast=8.0, modality="lwir"):
    return Detection(label="person", score=score, bbox=(u - 2, v - 2, u + 2, v + 2),
                     modality=modality, centroid=(u, v), area_px=16.0,
                     contrast_k=contrast)


def test_temporal_confirms_after_3_px_frames():
    tf = TemporalFilter(min_frames=3)
    for t in (0.0, 0.5, 1.0):
        out = tf.update([_person(50, 50)], t)
    assert out[0].attributes.get("confirmed") is True


def test_temporal_vetoes_glints():
    tf = TemporalFilter(min_frames=3)
    for t, c in ((0.0, 4.0), (0.5, 0.2), (1.0, 4.2), (1.5, 0.2)):
        out = tf.update([_person(50, 50, contrast=c)], t)
    assert "veto" in out[0].attributes


def test_temporal_world_frame_survives_camera_motion():
    tf = TemporalFilter(min_frames=3, radius_m=3.0)
    # Camera slides 3 m/frame; survivor fixed at world (100, 50).
    locs = [(100.0, 50.0)] * 3
    for t, (n, e) in zip((0.0, 0.5, 1.0), locs):
        out = tf.update([_person(50 + t * 40, 50)], t, loc=lambda d, n=n, e=e: (n, e))
    assert out[0].attributes.get("confirmed") is True


def test_flood_detector_finds_static_body_and_confirms():
    fd = FloodDetector()
    lwir = np.full((120, 160), 14.0, np.float32)
    lwir[60:66, 80:86] = 33.0
    rgb = np.full((240, 320, 3), (105, 100, 92), np.uint8)
    _shaded_patch(rgb, 118, 134, 158, 174)  # clothing cue (body-sized 16x16)
    dets = []
    for t in range(4):
        dets = fd.detect([ImageFrame(image=lwir, kind="lwir", t=float(t), gsd_m=0.084),
                          ImageFrame(image=rgb, kind="rgb", t=float(t), gsd_m=0.042)])
    persons = [d for d in dets if d.is_person]
    assert persons, "a clear static body must be detected"
    best = max(persons, key=lambda d: d.score)
    assert best.attributes.get("confirmed") is True
    assert best.attributes.get("cross_modal") == "agree"


def test_flood_detector_night_keeps_lwir_only():
    fd = FloodDetector()
    lwir = np.full((120, 160), 14.0, np.float32)
    lwir[60:66, 80:86] = 33.0
    night = np.full((240, 320, 3), (8, 8, 10), np.uint8)  # night: RGB blind
    dets = []
    for t in range(4):
        dets = fd.detect([ImageFrame(image=lwir, kind="lwir", t=float(t), gsd_m=0.084),
                          ImageFrame(image=night, kind="rgb", t=float(t), gsd_m=0.042)])
    persons = [d for d in dets if d.is_person and "veto" not in d.attributes]
    assert persons, "night must not delete thermal detections"
    # Dark flat night ground reads as water-like (safe: the in-water path also
    # never penalises); either way RGB-blindness must not delete the detection.
    assert any(d.attributes.get("cross_modal") in ("lwir_only_rgb_blind",
                                                  "lwir_only_in_water") for d in persons)


def test_flood_detector_mismatched_resolutions_fuse():
    # Phone RGB 1280x720 + Lepton 160x120: fusion must still pair (the
    # cross-resolution scale bug this file refuses to inherit).
    fd = FloodDetector()
    lwir = np.full((120, 160), 14.0, np.float32)
    lwir[60:66, 80:86] = 33.0
    rgb = np.full((720, 1280, 3), (105, 100, 92), np.uint8)
    _shaded_patch(rgb, 356, 376, 636, 656)  # 20x20 clothing at phone resolution
    dets = []
    for t in range(4):
        dets = fd.detect([ImageFrame(image=lwir, kind="lwir", t=float(t), gsd_m=0.084),
                          ImageFrame(image=rgb, kind="rgb", t=float(t), gsd_m=0.046)])
    persons = [d for d in dets if d.is_person]
    assert any(d.attributes.get("cross_modal") == "agree" for d in persons)
