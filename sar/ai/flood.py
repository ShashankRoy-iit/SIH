"""The flood detector: the best model for this problem statement, on evidence.

``FloodDetector`` is a physics-gated neural-heuristic ensemble tuned for one
job: finding people in flood water from a small UAV, day and night, with a
false-alarm rate a human operator can survive.  Every stage exists because
the kill-critique in ``docs/MODEL_CRITIQUE.md`` murdered the alternative:

1. **Thermal triage in kelvin** (inherited from ``ThermalAnomalyDetector``):
   human band on the *peak* pixel, bi-directional (cold-body-on-hot-roof).
2. **Geometry gate**: projected extent must be human at the *live* GSD, and
   the warm-context extent ratio must not scream "roof/vehicle".
3. **Flood-context reasoning**: an RGB water mask + flatness prior decides
   whether a candidate sits *in water* (confirm by descent, never penalise
   for missing RGB clothing cue) or *on a roof* (demand agreement).
4. **Glint suppression**: single-frame chromatic specks with no thermal
   partner and flickering contrast are water glints, not survivors.
5. **Conditional fusion**: RGB disagreement penalises only when RGB is
   usable (``rgb_quality >= 0.30``); at night LWIR-only is expected.
6. **Temporal consistency**: ≥3 frames, geo-consistent position, plausible
   thermal persistence; flicker vetoes.
7. **Optional neural head**: any ``DetectorBackend`` (YOLO TFLite/ONNX on
   the phone NPU) fused as one more vote behind the same gates — the
   network rejects *geometry* (roof shapes), the physics rejects
   *impossibility*, and only together do they hold precision.

Every rejection carries ``attributes["veto"]`` with a reason code, so the
sortie report can say *what* was suppressed and *why* — a detector whose
false alarms vanish silently cannot be trusted or improved.

The class honours the ``DetectorBackend`` contract (``detect(frame)``) and
also accepts a ``(lwir, rgb)`` pair like the ensemble does.  Pure numpy +
scipy: runs on a laptop, in CI, and — via the TFLite head — on the phone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.perception.detector import (Detection, DetectorEnsemble,
                                     ImageFrame, RgbHazardDetector,
                                     ThermalAnomalyDetector, aperture_radius_px,
                                     nms, rgb_frame_quality)

try:
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Reason codes (every veto is logged, never silent)
# --------------------------------------------------------------------------- #
VETO_GEOMETRY = "geometry:extent-not-human"
VETO_CONTEXT_ROOF = "context:roof-no-agreement"
VETO_GLINT = "glint:single-frame-flicker"
VETO_TEMPORAL = "temporal:needs-confirmation"
VETO_COLD_CLUTTER = "radiometry:below-human-band"


# --------------------------------------------------------------------------- #
# Stage helpers (pure functions — unit-tested directly)
# --------------------------------------------------------------------------- #
def geometry_gate(gsd_m: float, area_px: float, extent_ratio: float = 1.0,
                  max_extent_ratio: float = 6.5) -> Tuple[bool, str]:
    """Is this blob human-sized at this ground resolution?

    Expected body footprint: 0.62 m^2 apparent area.  Accept 8%..260% of it:
    the low end keeps chest-deep survivors (~15% visible), the high end keeps
    spreading/person-group cases; outside is speck or structure.
    """
    if not gsd_m or gsd_m <= 0:
        return True, "unknown-gsd:pass"
    expected = 0.62 / (gsd_m * gsd_m)
    frac = area_px / max(expected, 1e-6)
    if frac < 0.08:
        return False, f"{VETO_GEOMETRY}:too-small:{frac:.2f}"
    if frac > 2.6:
        return False, f"{VETO_GEOMETRY}:too-big:{frac:.2f}"
    if extent_ratio > max_extent_ratio:
        return False, f"{VETO_GEOMETRY}:embedded:{extent_ratio:.1f}"
    return True, f"ok:{frac:.2f}"


def water_mask(rgb: np.ndarray) -> np.ndarray:
    """Flood-water mask: dark, blue-cyan, and spatially smooth.

    Same discriminator as the hazard detector (smoothness separates water
    from shadowed rubble), factored out so the context stage and the LiDAR
    water flag share one definition.
    """
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        return np.zeros(img.shape[:2], dtype=bool)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = img.max(axis=-1) - img.min(axis=-1)
    if _HAVE_SCIPY:
        m = ndi.uniform_filter(lum, size=7, mode="nearest")
        std = np.sqrt(np.maximum(
            ndi.uniform_filter(lum * lum, size=7, mode="nearest") - m * m, 0.0))
    else:  # pragma: no cover
        std = np.zeros_like(lum)
    return ((b >= r - 6) & (lum < 118) & (chroma < 40) & (std < 4.2))


def flood_context(rgb: np.ndarray, u: float, v: float,
                  radius_px: float = 9.0) -> Dict[str, Any]:
    """What is around this image point: water, roof, or ground?

    Returns fractions in a disk around (u, v): ``water_frac`` (flood mask),
    ``bright_frac`` (sunlit roof/structure proxy: bright + low chroma), and
    the verdict used by fusion: 'water' | 'roof' | 'ground'.
    """
    h, w = np.asarray(rgb).shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    if not disk.any():
        return {"water_frac": 0.0, "bright_frac": 0.0, "verdict": "ground"}
    wm = water_mask(rgb)
    water_frac = float(wm[disk].mean())
    img = np.asarray(rgb, dtype=np.float32)
    lum = img[..., :3].mean(axis=-1)
    chroma = img[..., :3].max(axis=-1) - img[..., :3].min(axis=-1)
    bright_frac = float(((lum > 150) & (chroma < 45))[disk].mean())
    if water_frac > 0.45:
        verdict = "water"
    elif bright_frac > 0.55:
        verdict = "roof"
    else:
        verdict = "ground"
    return {"water_frac": round(water_frac, 3),
            "bright_frac": round(bright_frac, 3), "verdict": verdict}


def structure_edge_score(rgb: np.ndarray, u: float, v: float,
                         radius_px: float = 18.0) -> Dict[str, Any]:
    """How much long straight structure surrounds this image point?

    Roof corners and wall edges — the dominant daylight false alarm — sit at
    the intersection of long straight RGB edges.  A body on open ground does
    not.  Returns ``edge_frac`` (fraction of the disk that is a strong edge)
    and ``straight`` (whether those edge pixels share one orientation, i.e. a
    straight line rather than texture).  Bodies in rubble can score moderate
    edge_frac but low straightness; roof outlines score high on both.
    """
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim != 3:
        return {"edge_frac": 0.0, "straight": False, "on_structure": False}
    h, w = img.shape[:2]
    lum = img[..., :3].mean(axis=-1)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    if not disk.any():
        return {"edge_frac": 0.0, "straight": False, "on_structure": False}
    edges = disk & (mag > 14.0)
    edge_frac = float(edges.sum() / max(disk.sum(), 1))
    straight = False
    if edges.sum() >= 12:
        ang = np.arctan2(gy[edges], gx[edges]) % np.pi  # orientation, undirected
        # Circular concentration of doubled angles: 1.0 = one straight line.
        # A corner (two perpendicular lines) scores ~0 here, so corners are
        # caught by raw edge length instead: a roof outline crossing the disk
        # puts 25+ colinear-ish pixels in it; texture never does.
        c = np.abs(np.exp(2j * ang).mean())
        straight = bool(c > 0.55 or edges.sum() >= 25)
    # Single-pixel roof outlines give edge_frac ~0.03-0.06 in an 18px disk,
    # so the threshold sits below that (a first cut at 0.16 never fired).
    on_structure = bool(edge_frac > 0.015 and straight)
    return {"edge_frac": round(edge_frac, 3), "straight": straight,
            "on_structure": on_structure}


def glint_test(contrast_series: Sequence[float]) -> Tuple[bool, str]:
    """Is this contrast history a water glint?  Glints flicker: high variance
    relative to the mean across consecutive frames.  Bodies persist."""
    x = np.asarray(list(contrast_series), dtype=float)
    if x.size < 2:
        return False, "too-short"
    mean = float(x.mean())
    if mean <= 0.05:
        return False, "no-signal"
    cv = float(x.std() / mean)  # coefficient of variation
    if cv > 0.85:
        return True, f"flicker:cv={cv:.2f}"
    return False, f"stable:cv={cv:.2f}"


# --------------------------------------------------------------------------- #
# Temporal filter
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    u: float
    v: float
    frames: int = 1
    contrasts: List[float] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    first_t: float = 0.0
    last_t: float = 0.0
    label: str = "person"


class TemporalFilter:
    """Cross-frame confirmation with geo-consistency + flicker veto.

    A candidate becomes *confirmed* after ``min_frames`` sightings whose
    positions stay within the association gate and whose contrast history is
    not a glint.  Until then it is reported with ``needs_confirmation`` —
    visible to the operator and eligible for a confirmation descent, but not
    yet an alert.

    Association space matters: with a moving camera, a static survivor slides
    ~20 px/frame through the image, so pixel association never links and
    nothing ever confirms (found by the flood sim: recall 0.0 with 8
    dets/frame).  Pass ``loc`` — a callable mapping a detection to world
    (north, east) metres — and association happens in metres with
    ``radius_m``.  Without it (static bench camera), pixels + ``radius_px``.
    """

    def __init__(self, min_frames: int = 3, radius_px: float = 9.0,
                 radius_m: float = 3.0, max_gap_s: float = 3.0) -> None:
        self.min_frames = min_frames
        self.radius_px = radius_px
        self.radius_m = radius_m
        self.max_gap_s = max_gap_s
        self.tracks: List[_Track] = []
        self.confirmed_total = 0

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, dets: List[Detection], t: float,
               loc: Any = None) -> List[Detection]:
        # Age out stale tracks.
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_gap_s]
        gate = self.radius_m if loc is not None else self.radius_px

        def pos(d: Detection):
            return loc(d) if loc is not None else (d.u, d.v)

        out: List[Detection] = []
        for d in dets:
            if not d.is_person:
                out.append(d)  # hazards pass through unfiltered
                continue
            x, y = pos(d)
            best = None
            best_dist = gate
            for tr in self.tracks:
                dist = math.hypot(x - tr.u, y - tr.v)
                if dist < best_dist:
                    best, best_dist = tr, dist
            if best is None:
                best = _Track(u=x, v=y, first_t=t, last_t=t,
                              contrasts=[d.contrast_k], scores=[d.score],
                              label=d.label)
                self.tracks.append(best)
            else:
                # Running centroid; keeps slow drift (floating survivor) tracked.
                best.u = 0.7 * best.u + 0.3 * x
                best.v = 0.7 * best.v + 0.3 * y
                best.frames += 1
                best.last_t = t
                best.contrasts.append(d.contrast_k)
                best.contrasts = best.contrasts[-8:]
                best.scores.append(d.score)
            is_glint, why = glint_test(best.contrasts)
            d.attributes["track_frames"] = best.frames
            d.attributes["temporal_cv"] = why
            if is_glint:
                d.attributes["veto"] = VETO_GLINT
                d.score = min(d.score, 0.10)
                d.attributes["needs_confirmation"] = True
            elif best.frames >= self.min_frames:
                d.attributes["confirmed"] = True
                d.score = float(min(0.99, d.score + 0.06))
                self.confirmed_total += 1
            else:
                d.attributes["veto"] = VETO_TEMPORAL
                d.attributes["needs_confirmation"] = True
            out.append(d)
        return out


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class FloodDetector:
    """Physics-gated flood SAR ensemble.  See module docstring for the why."""

    name = "flood"

    def __init__(self,
                 neural: Any = None,
                 min_frames: int = 3,
                 rgb_blind_threshold: float = 0.30,
                 agreement_bonus: float = 0.24,
                 disagreement_penalty: float = 0.16,
                 cross_modal_radius_px: float = 7.0,
                 max_extent_ratio: float = 6.5,
                 extent_flag_ratio: float = 4.5,
                 lwir_kwargs: Optional[Dict[str, Any]] = None,
                 rgb_kwargs: Optional[Dict[str, Any]] = None) -> None:
        # A wider peak budget than the audited baseline (64 vs 40): warm roofs
        # crowd genuine survivors out of a 40-peak budget, and the flood
        # stack's downstream vetoes (geometry/context/temporal) make extra
        # peaks cheap — precision is protected downstream, so recall is
        # bought upstream.  Override via lwir_kwargs={"max_peaks": N}.
        _lw = dict(lwir_kwargs or {})
        _lw.setdefault("max_peaks", 64)
        self.lwir = ThermalAnomalyDetector(**_lw)
        self.rgb = RgbHazardDetector(**(rgb_kwargs or {}))
        self.neural = neural
        self.modalities = ("lwir", "rgb")
        self.max_extent_ratio = max_extent_ratio
        self.extent_flag_ratio = extent_flag_ratio
        self.temporal = TemporalFilter(min_frames=min_frames)
        self.rgb_blind_threshold = rgb_blind_threshold
        self.agreement_bonus = agreement_bonus
        self.disagreement_penalty = disagreement_penalty
        self.radius = cross_modal_radius_px
        self.last_stats: Dict[str, Any] = {}
        self.last_rgb_quality = 0.0

    # -- main entry ----------------------------------------------------- #
    def detect(self, frames: Any, rgb_quality: Optional[float] = None,
               world_fn: Any = None) -> List[Detection]:
        """Detect over a frame or (lwir, rgb) pair.

        ``world_fn`` maps a detection to world (north, east) metres for
        motion-robust temporal association.  Omit it only for a static
        camera (bench/eval); a moving camera without it confirms nothing.
        """
        flist = list(frames) if isinstance(frames, (list, tuple)) else [frames]
        by_kind: Dict[str, Any] = {}
        for f in flist:
            k = str(getattr(f, "kind", "") or getattr(f, "modality", ""))
            if k:
                by_kind[k] = f

        lwir_dets: List[Detection] = []
        rgb_dets: List[Detection] = []
        if "lwir" in by_kind:
            try:
                lwir_dets = self.lwir.detect(by_kind["lwir"])
            except Exception as exc:
                self.last_stats["error_lwir"] = repr(exc)
        if "rgb" in by_kind:
            try:
                rgb_dets = self.rgb.detect(by_kind["rgb"])
            except Exception as exc:
                self.last_stats["error_rgb"] = repr(exc)
        neural_dets: List[Detection] = []
        if self.neural is not None:
            for k, fr in by_kind.items():
                try:
                    if k in tuple(getattr(self.neural, "modalities", (k,))):
                        neural_dets.extend(self.neural.detect(fr))
                except Exception as exc:
                    self.last_stats["error_neural"] = repr(exc)

        if rgb_quality is None:
            rgb_quality = (rgb_frame_quality(by_kind["rgb"]) if "rgb" in by_kind else 0.0)
        self.last_rgb_quality = float(np.clip(rgb_quality, 0.0, 1.0))
        rgb_usable = self.last_rgb_quality >= self.rgb_blind_threshold

        rgb_img = np.asarray(by_kind["rgb"].image) if "rgb" in by_kind else None
        gsd = float(getattr(by_kind.get("lwir", object()), "gsd_m", 0.0) or 0.0)
        # Camera resolution scales: RGB is usually 2-8x the LWIR size, so an
        # LWIR centroid must be scaled UP for RGB context and RGB centroids
        # scaled DOWN for pairing.  See the stage-3 note.
        _lw_shape0 = (np.asarray(by_kind["lwir"].image).shape[:2]
                      if "lwir" in by_kind else (240, 320))
        _rgb_shape0 = rgb_img.shape[:2] if rgb_img is not None else _lw_shape0
        _up = _rgb_shape0[1] / max(_lw_shape0[1], 1)
        _up_v = _rgb_shape0[0] / max(_lw_shape0[0], 1)

        # Stage 1+2: flood context FIRST, then the context-aware geometry gate.
        # Order matters: a survivor on a sun-heated roof is *supposed* to have
        # warm surroundings, so the embedded-extent veto must not fire there —
        # the roof-flag path (stage 3) handles roofs visibly instead.  Size
        # gates (too small / too big) still apply everywhere: a speck on a
        # roof is still a speck.
        gated: List[Detection] = []
        for d in lwir_dets:
            if d.is_person and rgb_img is not None:
                ctx = flood_context(rgb_img, d.u * _up, d.v * _up_v,
                                    radius_px=9.0 * _up)
                d.attributes["context"] = ctx["verdict"]
                d.attributes["water_frac"] = ctx["water_frac"]
                if ctx["verdict"] == "water":
                    # In-water: no clothing cue is EXPECTED (head/shoulders
                    # only).  Never penalise; flag for confirmation descent.
                    d.attributes["needs_confirmation"] = True
                    d.attributes["in_water"] = True
                elif ctx["verdict"] == "roof":
                    d.attributes["on_roof"] = True
            if d.is_person:
                max_ext = (1e9 if d.attributes.get("context") == "roof"
                           else self.max_extent_ratio)
                ext = float(d.attributes.get("extent_ratio", 1.0))
                ok, why = geometry_gate(gsd, d.area_px, ext, max_ext)
                d.attributes["geometry"] = why
                if not ok:
                    d.attributes["veto"] = why
                    d.score = min(d.score, 0.12)
                elif (self.extent_flag_ratio < ext <= self.max_extent_ratio
                        and d.attributes.get("context") != "roof"):
                    # Grey zone: warm surroundings, but not egregious.  A
                    # survivor sheltering against a warm wall lives here, so
                    # flag + down-weight + confirm — never hide.
                    d.attributes["extent_flag"] = round(ext, 2)
                    d.attributes["needs_confirmation"] = True
                    d.score = float(max(0.05, d.score - 0.15))
            gated.append(d)

        # Stage 3: conditional cross-modal fusion.
        # NOTE: the two cameras have different resolutions (phone RGB 1280x720
        # vs Lepton 160x120), so pairing happens in LWIR-pixel space with the
        # RGB centroids scaled down.  Comparing raw pixels across mismatched
        # sizes silently disables all fusion — a real bug this file refuses to
        # inherit (see tech.md §8).
        lw_shape = np.asarray(by_kind["lwir"].image).shape[:2] if "lwir" in by_kind else (240, 320)
        rgb_shape = np.asarray(by_kind["rgb"].image).shape[:2] if "rgb" in by_kind else lw_shape
        su = lw_shape[1] / max(rgb_shape[1], 1)
        sv = lw_shape[0] / max(rgb_shape[0], 1)

        def pair_dist(a: Detection, b: Detection) -> float:
            return math.hypot(a.u - b.u * su, a.v - b.v * sv)

        persons_l = [d for d in gated if d.is_person]
        persons_r = [d for d in rgb_dets if d.is_person]
        consumed = set()
        for d in persons_l:
            partners = [o for o in persons_r if pair_dist(d, o) <= self.radius]
            if partners:
                best = max(partners, key=lambda o: o.score)
                d.score = float(np.clip(d.score + self.agreement_bonus * best.score,
                                        0, 0.995))
                d.modality = "fused"
                d.attributes["cross_modal"] = "agree"
                d.attributes.pop("veto", None)
                consumed.add(id(best))
            elif rgb_usable and d.attributes.get("context") == "roof":
                # Warm + human-sized + on a bright roof + invisible in good
                # light: usually a heated roof patch — BUT a survivor on a
                # roof looks identical until the confirmation pass, so this
                # is flagged, down-weighted, and queued for a descent, NEVER
                # hidden.  Hiding roof candidates is how roof survivors die.
                d.attributes["roof_flag"] = VETO_CONTEXT_ROOF
                d.attributes["cross_modal"] = "lwir_only_on_roof"
                d.attributes["needs_confirmation"] = True
                d.score = float(max(0.05, d.score - 2.0 * self.disagreement_penalty))
            elif rgb_usable and not d.attributes.get("in_water"):
                d.attributes["cross_modal"] = "lwir_only"
                d.attributes["needs_confirmation"] = True
                d.score = float(max(0.05, d.score - self.disagreement_penalty
                                        * self.last_rgb_quality))
                # Warm, human-sized, invisible in good light, sitting on a
                # long straight RGB edge: a roof corner or wall edge until
                # proven otherwise.  Down-weighted hard and flagged — the
                # confirmation descent (RGB identity at 22 m) is the appeal
                # court, and hiding is still forbidden.
                if rgb_img is not None:
                    es = structure_edge_score(rgb_img, d.u / max(su, 1e-6),
                                              d.v / max(sv, 1e-6),
                                              radius_px=9.0 / max(su, 1e-6))
                    d.attributes["edge_frac"] = es["edge_frac"]
                    if es["on_structure"]:
                        d.attributes["on_structure_edge"] = True
                        d.attributes["needs_confirmation"] = True
                        d.score = float(max(0.05, d.score - 0.30))
            else:
                d.attributes["cross_modal"] = ("lwir_only_in_water"
                                               if d.attributes.get("in_water")
                                               else "lwir_only_rgb_blind")

        # Stage 4: neural vote (if a head is loaded) — agreement only helps,
        # disagreement with physics never deletes (the network is the junior).
        for nd in neural_dets:
            if not getattr(nd, "is_person", False):
                gated.append(nd)
                continue
            matched = [d for d in persons_l
                       if math.hypot(d.u - nd.u, d.v - nd.v) <= self.radius + 3.0
                       or pair_dist(d, nd) <= self.radius + 3.0]
            if matched:
                best = max(matched, key=lambda d: d.score)
                best.score = float(np.clip(best.score + 0.10 * nd.score, 0, 0.995))
                best.attributes["neural_agree"] = round(nd.score, 3)
            # A lone neural box with no thermal support is kept, marked, and
            # scored down — it may be a debris-covered survivor visible only
            # in RGB, but it may equally be a person on a poster.
            nd.attributes["cross_modal"] = "neural_only"
            nd.score = float(max(0.05, nd.score - 0.10))
            gated.append(nd)

        # Hazards + unmatched RGB persons pass through.
        for d in rgb_dets:
            if id(d) not in consumed and not (d.is_person and d.attributes.get("consumed_by")):
                if d.is_person:
                    d.attributes["cross_modal"] = "rgb_only"
                    d.score = float(max(0.05, d.score - 0.6 * self.disagreement_penalty))
                gated.append(d)

        # Stage 5: temporal confirmation (world-frame when available).
        t = float(getattr(by_kind.get("lwir", by_kind.get("rgb")), "t", 0.0))
        out = self.temporal.update(gated, t, loc=world_fn)
        self.last_stats.update({"lwir": len(lwir_dets), "rgb": len(rgb_dets),
                                "neural": len(neural_dets),
                                "rgb_quality": round(self.last_rgb_quality, 3),
                                "out": len(out)})
        return nms(out, 0.30, center_thresh=2.0)

    def reset(self) -> None:
        self.temporal.reset()


def build_flood_detector(mode: str = "auto", **kwargs: Any) -> FloodDetector:
    """Build the flood stack.  ``mode`` mirrors ``build_detector_stack``:

    * ``auto``/``flood`` — neural head attached when weights exist, else pure
      physics (always works, zero downloads).
    * ``heuristic`` — physics only, deterministic.
    * ``neural`` — require the neural head (raises without weights).
    """
    neural = None
    if mode in ("auto", "flood", "hybrid", "neural"):
        try:
            from sar.ai.registry import default_registry
            from sar.ai.backends import NeuralDetectorBackend, ThermalNeuralBackend
            reg = default_registry()
            spec = reg.best_for("person_detection", "lwir") or reg.best_for(
                "person_detection", "rgb")
            if spec is not None:
                cls = (ThermalNeuralBackend if spec.modality == "lwir"
                       else NeuralDetectorBackend)
                neural = cls(spec)
        except Exception as exc:
            if mode == "neural":
                raise RuntimeError(f"mode='neural' but no head loaded: {exc}")
            neural = None
    if mode == "neural" and neural is None:
        raise RuntimeError("mode='neural' but no weights found in models/")
    return FloodDetector(neural=neural, **kwargs)


__all__ = ["FloodDetector", "TemporalFilter", "build_flood_detector",
           "geometry_gate", "water_mask", "flood_context", "structure_edge_score",
           "glint_test",
           "VETO_GEOMETRY", "VETO_CONTEXT_ROOF", "VETO_GLINT", "VETO_TEMPORAL",
           "VETO_COLD_CLUTTER"]
