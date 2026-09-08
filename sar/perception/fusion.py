"""Cross-modal fusion with online boresight self-calibration.

The ensemble in :mod:`sar.perception.detector` decides, within one frame, whether
a thermal detection and a visible detection support each other.  This module does
the harder job of deciding whether they are detections **of the same object** -
and it fixes the thing that makes that non-trivial on a real airframe.

The boresight problem
---------------------
A dual-sensor payload has two lenses on one bracket.  They are never perfectly
co-aligned: the LWIR core sits a few millimetres and a fraction of a degree from
the visible camera, the bracket flexes under vibration and thermal cycling, and
the 3D-printed mount this airframe actually uses will move more between flights
than between them.  At 40 m AGL a 1-degree boresight error is 0.7 m on the ground
- seven pixels - which is larger than the survivor being imaged.  Fuse on raw
pixels and the two modalities disagree about everything, and the system's best
false-alarm filter (cross-modal agreement) becomes its worst source of missed
confirmations.

Field recalibration of a boresight normally needs a target and a tripod.  A
flying SAR aircraft has neither.  So the offset is estimated **online from the
mission's own data**: every pair of LWIR and RGB detections that agree on an
object is a noisy sample of the boresight offset, and a robust (Huber-weighted)
running estimate converges within a few hundred detections.  That is what
:class:`BoresightEstimator` does, and it is why the fusion radius can be tight
enough to reject accidental coincidences.

Outputs
-------
:class:`FusedObservation` - one object, one label likelihood vector over the
whole taxonomy, one position with uncertainty, and the evidence trail that
produced it.  The likelihood vector rather than an argmax matters: the tracker,
the victim map and the triage model all need "65% person, 30% animal, 5% cloth"
rather than "person", because those are different decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.perception.detector import (ALL_LABELS, HAZARD_LABELS, PERSON_LABELS,
                                     Detection)

__all__ = ["FusedObservation", "BoresightEstimator", "CrossModalFuser"]


@dataclass
class FusedObservation:
    """One physical object seen by one or both sensors in one frame."""

    t: float
    u: float
    v: float
    #: Label likelihood over the full taxonomy, sums to 1.
    likelihood: Dict[str, float]
    confidence: float
    modalities: Tuple[str, ...]
    bbox: Tuple[float, float, float, float]
    area_px: float = 0.0
    peak_temp_c: Optional[float] = None
    mean_temp_c: Optional[float] = None
    background_temp_c: Optional[float] = None
    contrast_k: float = 0.0
    polarity: str = "n/a"
    elongation: float = 1.0
    compactness: float = 0.0
    scale_px: float = 0.0
    #: Pixel offset between the two sensors for this pair, if both saw it.
    boresight_sample: Optional[Tuple[float, float]] = None
    cross_modal_agree: bool = False
    rgb_blind: bool = False
    #: Set when a single modality saw this and the other one did not, so the
    #: mission planner should schedule a low confirmation look before anyone is
    #: dispatched.  This is the flag that drives the two-pass architecture.
    needs_confirmation: bool = False
    sources: List[Detection] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def label(self) -> str:
        return max(self.likelihood, key=lambda k: self.likelihood[k])

    @property
    def person_probability(self) -> float:
        return float(sum(self.likelihood.get(l, 0.0) for l in PERSON_LABELS))

    @property
    def hazard_probability(self) -> float:
        return float(sum(self.likelihood.get(l, 0.0) for l in HAZARD_LABELS))

    @property
    def is_person(self) -> bool:
        return self.label in PERSON_LABELS

    def to_detection(self, modality: Optional[str] = None) -> Detection:
        """Project back to a :class:`Detection` for the tracker and the map.

        The tracker consumes detections, not likelihood vectors, so the fused
        observation is flattened here.  The full vector is preserved in
        ``attributes['likelihood']`` for anything that wants it (the triage model
        and the confirmation planner do).

        ``modality`` defaults to the *true* sensor set.  Hard-coding "fused" here
        - as an earlier version did - made every single-modality observation look
        like cross-modal agreement to the tracker, because the tracker infers
        agreement from the set of modalities it has seen.  The result was that a
        lone LWIR point target at survey altitude, unconfirmed by anything,
        reported itself as confirmed by two sensors.
        """
        if modality is None:
            mods = [m for m in self.modalities if m]
            modality = "fused" if (self.cross_modal_agree or len(set(mods)) > 1) \
                else (mods[0] if mods else "fused")
        d = Detection(
            label=self.label, score=self.confidence, bbox=self.bbox,
            modality=modality, frame_t=self.t, centroid=(self.u, self.v),
            area_px=self.area_px, peak_temp_c=self.peak_temp_c,
            mean_temp_c=self.mean_temp_c, background_temp_c=self.background_temp_c,
            contrast_k=self.contrast_k, polarity=self.polarity,
            elongation=self.elongation, compactness=self.compactness,
            scale_px=self.scale_px, backend="fusion",
            attributes={**self.attributes, "likelihood": dict(self.likelihood),
                        "cross_modal": self.cross_modal_agree,
                        "rgb_blind": self.rgb_blind,
                        "modalities": list(self.modalities),
                        "needs_confirmation": self.needs_confirmation,
                        "n_sources": len(self.sources)},
        )
        return d

    def to_dict(self) -> Dict[str, Any]:
        top = sorted(self.likelihood.items(), key=lambda kv: -kv[1])[:3]
        return {"t": round(self.t, 3), "u": round(self.u, 1), "v": round(self.v, 1),
                "label": self.label, "confidence": round(self.confidence, 3),
                "top": [(k, round(v, 3)) for k, v in top],
                "modalities": list(self.modalities),
                "cross_modal": self.cross_modal_agree, "rgb_blind": self.rgb_blind,
                "person_p": round(self.person_probability, 3),
                "peak_temp_c": self.peak_temp_c, "area_px": round(self.area_px, 1)}


# --------------------------------------------------------------------------- #
class BoresightEstimator:
    """Robust online estimate of the LWIR->RGB pixel offset.

    Model: a constant translation ``(du, dv)`` plus an optional scale term, which
    covers the two errors a rigidly-mounted dual-sensor payload actually has
    (mounting offset, and different focal lengths giving different GSDs).  Rotation
    is not estimated: on a bracket-mounted pair it is a build error that shows up
    as a residual, and estimating it from a handful of noisy pairs is unstable.

    Estimation uses iteratively-reweighted least squares with a Huber loss, so a
    wrong pairing (two unrelated detections that happened to be close) does not
    drag the estimate.  Convergence is reported as a sample count and a residual
    sigma, and the estimator refuses to apply a correction it does not trust -
    which matters, because applying a bad boresight correction is worse than
    applying none.
    """

    def __init__(self, min_samples: int = 25, max_offset_px: float = 25.0,
                 huber_delta: float = 2.5, decay: float = 0.995) -> None:
        self.min_samples = min_samples
        self.max_offset_px = max_offset_px
        self.huber_delta = huber_delta
        self.decay = decay
        self._du = 0.0
        self._dv = 0.0
        self._n = 0
        self._resid_sigma = float("inf")
        self._samples: List[Tuple[float, float, float]] = []

    # ------------------------------------------------------------------ #
    @property
    def offset(self) -> Tuple[float, float]:
        """Current (du, dv) to add to an LWIR pixel to predict the RGB pixel."""
        if self._n < self.min_samples:
            return (0.0, 0.0)
        return (self._du, self._dv)

    @property
    def converged(self) -> bool:
        return self._n >= self.min_samples and self._resid_sigma < 4.0

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def residual_sigma_px(self) -> float:
        return self._resid_sigma

    def observe(self, du: float, dv: float, weight: float = 1.0) -> None:
        """Add one pairing sample and re-fit."""
        if not (math.isfinite(du) and math.isfinite(dv)):
            return
        if math.hypot(du, dv) > self.max_offset_px:
            return                              # implausible pairing, discard
        self._samples.append((du, dv, max(weight, 0.05)))
        self._n += 1
        if len(self._samples) > 600:
            self._samples = self._samples[-500:]
        self._fit()

    def _fit(self) -> None:
        if len(self._samples) < 5:
            return
        arr = np.asarray(self._samples, dtype=float)
        du, dv, w = arr[:, 0], arr[:, 1], arr[:, 2]
        # Two IRLS passes are plenty for a 2-parameter location estimate.
        weights = w.copy()
        mu_u = float(np.median(du))
        mu_v = float(np.median(dv))
        for _ in range(2):
            r = np.hypot(du - mu_u, dv - mu_v)
            sigma = max(float(np.median(r)) * 1.4826, 0.3)
            huber = np.where(r <= self.huber_delta * sigma, 1.0,
                             self.huber_delta * sigma / np.maximum(r, 1e-6))
            weights = w * huber
            mu_u = float(np.sum(weights * du) / np.sum(weights))
            mu_v = float(np.sum(weights * dv) / np.sum(weights))
        r = np.hypot(du - mu_u, dv - mu_v)
        self._du, self._dv = mu_u, mu_v
        self._resid_sigma = float(max(np.median(r) * 1.4826, 1e-3))

    def reset(self) -> None:
        self._du = self._dv = 0.0
        self._n = 0
        self._resid_sigma = float("inf")
        self._samples.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {"du_px": round(self._du, 3), "dv_px": round(self._dv, 3),
                "applied": self.converged, "n_samples": self._n,
                "residual_sigma_px": (round(self._resid_sigma, 3)
                                      if math.isfinite(self._resid_sigma) else None)}


# --------------------------------------------------------------------------- #
class CrossModalFuser:
    """Pairs detections across modalities and produces fused observations.

    Pairing is mutually-nearest-neighbour within a radius that adapts to the
    estimated boresight residual, so it starts permissive (before the estimator
    has converged) and tightens as evidence accumulates.  Unpaired detections
    pass through unchanged - losing a thermal-only detection because the visible
    camera was blinded by smoke would defeat the purpose of carrying two sensors.
    """

    def __init__(self, labels: Sequence[str] = ALL_LABELS,
                 base_radius_px: float = 6.0,
                 boresight: Optional[BoresightEstimator] = None,
                 agreement_boost: float = 0.14) -> None:
        self.labels = tuple(labels)
        self.base_radius_px = base_radius_px
        self.boresight = boresight or BoresightEstimator()
        self.agreement_boost = agreement_boost
        self.stats: Dict[str, Any] = {"pairs": 0, "lwir_only": 0, "rgb_only": 0}

    # ------------------------------------------------------------------ #
    def radius_px(self) -> float:
        """Pairing radius, widened until the boresight estimate converges."""
        if self.boresight.converged:
            return float(self.base_radius_px + 2.0 * self.boresight.residual_sigma_px)
        return float(self.base_radius_px * 2.0)

    def fuse(self, detections: Sequence[Detection], t: float,
             rgb_quality: float = 1.0) -> List[FusedObservation]:
        lwir = [d for d in detections if d.modality in ("lwir", "fused")
                or (d.modality == "lwir" and d.backend == "fusion")]
        rgb = [d for d in detections if d.modality == "rgb"]
        lwir = [d for d in detections if d.modality != "rgb"]
        rgb_blind = rgb_quality < 0.30 or not rgb

        du, dv = self.boresight.offset
        radius = self.radius_px()

        # ---- mutually-nearest-neighbour pairing ---------------------------
        pairs: List[Tuple[int, int, float]] = []
        for i, a in enumerate(lwir):
            best_j, best_d = -1, float("inf")
            for j, b in enumerate(rgb):
                if a.label != b.label and not (a.is_person and b.is_person):
                    continue
                d = math.hypot((a.u + du) - b.u, (a.v + dv) - b.v)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j >= 0 and best_d <= radius:
                pairs.append((i, best_j, best_d))
        pairs.sort(key=lambda p: p[2])
        used_i: set = set()
        used_j: set = set()
        accepted: List[Tuple[int, int, float]] = []
        for i, j, d in pairs:
            if i in used_i or j in used_j:
                continue
            # Mutual check: the RGB detection must also prefer this LWIR one.
            a = lwir[i]
            best_i2, best_d2 = -1, float("inf")
            for k, b in enumerate(lwir):
                if b.label != a.label and not (b.is_person and a.is_person):
                    continue
                dd = math.hypot((b.u + du) - rgb[j].u, (b.v + dv) - rgb[j].v)
                if dd < best_d2:
                    best_i2, best_d2 = k, dd
            if best_i2 != i:
                continue
            used_i.add(i)
            used_j.add(j)
            accepted.append((i, j, d))

        out: List[FusedObservation] = []
        for i, j, d in accepted:
            a, b = lwir[i], rgb[j]
            self.boresight.observe(b.u - a.u, b.v - a.v,
                                   weight=min(a.score, b.score))
            self.stats["pairs"] += 1
            out.append(self._merge(a, b, d, t))
        for i, a in enumerate(lwir):
            if i in used_i:
                continue
            self.stats["lwir_only"] += 1
            out.append(self._single(a, t, rgb_blind=rgb_blind, rgb_quality=rgb_quality))
        for j, b in enumerate(rgb):
            if j in used_j:
                continue
            self.stats["rgb_only"] += 1
            out.append(self._single(b, t, rgb_blind=False, rgb_quality=rgb_quality))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _union_box(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, float, float, float]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def _likelihood(self, dets: Sequence[Detection],
                    weights: Sequence[float]) -> Dict[str, float]:
        """Weighted vote over the taxonomy, then normalised.

        Votes are combined in probability space rather than by taking the argmax
        of the highest-scoring detection: two sensors at 0.6 for 'person' are
        stronger evidence than one at 0.8, and this is the only place in the stack
        where that is expressed.
        """
        acc: Dict[str, float] = {lab: 1e-3 for lab in self.labels}
        for d, w in zip(dets, weights):
            lab = d.label if d.label in acc else "unknown"
            acc[lab] = acc.get(lab, 0.0) + w * max(d.score, 0.01)
            # Cross-label leakage: a confident 'person' also weakly supports
            # 'person_group' and vice versa, and a hot anomaly weakly supports
            # every thermal label.  Without leakage a single mislabelled frame
            # resets the evidence.
            if lab in PERSON_LABELS:
                for other in PERSON_LABELS:
                    if other != lab:
                        acc[other] += 0.15 * w * d.score
        tot = sum(acc.values()) or 1.0
        return {k: v / tot for k, v in acc.items()}

    def _merge(self, a: Detection, b: Detection, dist: float, t: float) -> FusedObservation:
        # Weight each sensor by its own confidence and by how informative it is.
        wa = a.score * (1.2 if a.modality != "rgb" else 1.0)
        wb = b.score * (1.0 if b.modality == "rgb" else 1.2)
        lk = self._likelihood([a, b], [wa, wb])
        conf = float(min(0.995, max(a.score, b.score)
                         + self.agreement_boost * min(a.score, b.score)))
        for lab in PERSON_LABELS:
            lk[lab] *= (1.0 + 0.35)
        tot = sum(lk.values()) or 1.0
        lk = {k: v / tot for k, v in lk.items()}
        # Radiometry comes from whichever sensor has it; geometry from the
        # higher-resolution one.
        thermal = a if a.peak_temp_c is not None else b
        visual = b if b.modality == "rgb" else a
        return FusedObservation(
            t=t, u=float((a.u * wa + b.u * wb) / (wa + wb)),
            v=float((a.v * wa + b.v * wb) / (wa + wb)),
            likelihood=lk, confidence=conf, modalities=("lwir", "rgb"),
            bbox=self._union_box(a.bbox, b.bbox),
            area_px=max(a.area_px, b.area_px),
            peak_temp_c=thermal.peak_temp_c, mean_temp_c=thermal.mean_temp_c,
            background_temp_c=thermal.background_temp_c,
            contrast_k=thermal.contrast_k, polarity=thermal.polarity,
            elongation=visual.elongation, compactness=visual.compactness,
            scale_px=max(a.scale_px, b.scale_px),
            boresight_sample=(b.u - a.u, b.v - a.v),
            cross_modal_agree=True, rgb_blind=False, sources=[a, b],
            attributes={"pair_distance_px": round(dist, 2),
                        "lwir_score": round(a.score, 3),
                        "rgb_score": round(b.score, 3),
                        "stage": "cross_modal"},
        )

    def _single(self, d: Detection, t: float, rgb_blind: bool,
                rgb_quality: float) -> FusedObservation:
        conf = float(d.score)
        note = None
        needs_confirmation = False
        if d.is_person and not rgb_blind and d.modality != "rgb":
            # Warm and human-sized, and the visible camera could have seen it but
            # did not.  Reduce confidence and flag for a confirmation look.
            # 0.22 was far too gentle.  Measured at 110 m AGL, this class was
            # 62 of 69 person false alarms - a warm rock or a patch of sun-baked
            # asphalt that the visible camera had a fair look at and correctly
            # declined to call a person.  Their confidence sat at 0.75-0.83 while
            # genuinely cross-confirmed survivors sat at 0.99, so the evidence was
            # there but the weighting threw it away.  0.55 puts an unconfirmed
            # LWIR-only detection below the tracker's 0.50 person-evidence
            # threshold, which is the intended behaviour: it still opens a
            # candidate track and still appears on the map, but it does not
            # dispatch anybody until a low confirmation look resolves it.
            conf = float(max(0.05, d.score * (1.0 - 0.55 * rgb_quality)))
            note = "lwir_only_rgb_usable"
            needs_confirmation = True
        elif d.is_person and rgb_blind and d.modality != "rgb":
            note = "lwir_only_rgb_blind"
        elif d.is_person and d.modality == "rgb":
            # The two single-modality cases are NOT symmetric, and treating them
            # as if they were cost five background false alarms per frame at
            # survey altitude.
            #
            # An LWIR person detection has passed a physical gate: apparent
            # temperature inside the human band (23.0-40.5 C), a mean above the
            # cold-background floor, an area consistent with a body at that GSD,
            # and an extent veto that rejects vehicle-sized warm objects.  Warm
            # rock and sun-baked asphalt mostly fail it.
            #
            # An RGB person detection has passed a shape and appearance test
            # only.  From 70 m a person is a handful of pixels, and a shadow, a
            # crack in the road, a pole or a piece of debris matches that
            # description just as well.  There is no physical quantity that
            # distinguishes them, so the honest confidence is much lower and it
            # scales with how much shape information the pixels actually carry.
            shape = float(np.clip(d.area_px / 25.0, 0.0, 1.0))   # 5x5 px = posture
            conf = float(d.score * (0.30 + 0.40 * shape)
                         * (0.35 + 0.65 * float(np.clip(rgb_quality, 0.0, 1.0))))
            note = "rgb_only_unconfirmed"
            needs_confirmation = True
        lk = self._likelihood([d], [1.0])
        return FusedObservation(
            t=t, u=d.u, v=d.v, likelihood=lk, confidence=conf,
            modalities=(d.modality,), bbox=d.bbox, area_px=d.area_px,
            peak_temp_c=d.peak_temp_c, mean_temp_c=d.mean_temp_c,
            background_temp_c=d.background_temp_c, contrast_k=d.contrast_k,
            polarity=d.polarity, elongation=d.elongation,
            compactness=d.compactness, scale_px=d.scale_px,
            cross_modal_agree=False, rgb_blind=rgb_blind, sources=[d],
            needs_confirmation=needs_confirmation,
            attributes={"stage": "single_modal", "note": note,
                        "detection_backend": d.backend, **d.attributes},
        )
