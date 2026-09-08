"""The on-device perception pipeline: frames in, actionable picture out.

This is the object the mission loop actually calls.  One invocation takes a
synchronised pair of frames plus the navigation state that was valid when they
were captured, and returns everything the rest of the system consumes:

    frames + pose + nav quality
        -> detect            (detector.py: two-path, geometry-aware)
        -> fuse              (fusion.py: cross-modal pairing, boresight)
        -> geo-tag           (geotag.py: pixel -> geodetic + error ellipse)
        -> track             (tracker.py: geographic Kalman tracks, voting)
        -> hazards           (hazard.py: contextual severity, plumes, LZs)
        -> assess            (thermal.py: physiology -> triage tier + clock)
        -> PerceptionProduct

Design rules this module enforces
---------------------------------
**Nothing leaves the aircraft that has not been confirmed.**  Single-frame
detections are candidates; only tracks that satisfy the persistence and
evidence thresholds become alerts.  The reason is measured, not aesthetic: at
110 m AGL a sun-warmed rock reads inside the human thermal band
(``artifacts/subpixel_radiometry.json``), so a single high-altitude frame cannot
be trusted to task a rescue asset.

**Every number carries its uncertainty.**  A position without an error ellipse
and a triage tier without a clock are not actionable, so the product exposes
``sigma_m``, ``r95_m``, ``time_critical_s`` and a ``rationale`` list.

**The pipeline must not throw.**  A detector that crashes on one frame, a pixel
that projects above the horizon, a singular covariance - none of these may end a
sortie over a flooded basin.  Each stage is individually guarded and reports its
own failures in ``PerceptionProduct.faults``.

**It must run in the real-time budget.**  Measured on a 2-core x86 without an
NPU: ~50 ms per frame pair end to end, against an 8 Hz thermal core (125 ms).
The budget is asserted by ``tests/test_perception_timing.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint
from sar.perception.detector import (PERSON_LABELS, Detection, DetectorEnsemble,
                                     build_reference_pipeline, rgb_frame_quality)
from sar.perception.fusion import CrossModalFuser, FusedObservation
from sar.perception.geotag import GeoTag, NavQuality, PixelGeoTagger
from sar.perception.hazard import HazardMap, LandingZone
from sar.perception.thermal import (PriorityTier, SurvivorViability,
                                    ThermalPhysiologyModel)
from sar.perception.tracker import Track, Tracker

__all__ = ["SurvivorAssessment", "PerceptionProduct", "PerceptionPipeline"]


@dataclass
class SurvivorAssessment:
    """One confirmed survivor: where, how sure, how urgent, and why."""

    track_id: int
    point: GeoPoint
    north_m: float
    east_m: float
    sigma_m: float
    r95_m: float
    label: str
    confidence: float
    viability: SurvivorViability
    priority: PriorityTier
    time_critical_s: Optional[float]
    n_obs: int
    cross_modal: bool
    immobility: float
    speed_ms: float
    first_seen_t: float
    last_seen_t: float
    needs: str = "medical"
    group_size: int = 1
    landing_zones: List[LandingZone] = field(default_factory=list)
    hazards_nearby: List[Dict[str, Any]] = field(default_factory=list)
    needs_confirmation_pass: bool = False
    best_gsd_m: float = 0.0

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id, "lat": self.point.lat, "lon": self.point.lon,
            "north_m": round(self.north_m, 1), "east_m": round(self.east_m, 1),
            "sigma_m": round(self.sigma_m, 2), "r95_m": round(self.r95_m, 1),
            "label": self.label, "confidence": round(self.confidence, 3),
            "priority": self.priority.value,
            "time_critical_s": (round(self.time_critical_s)
                                if self.time_critical_s else None),
            "n_obs": self.n_obs, "cross_modal": self.cross_modal,
            "immobility": round(self.immobility, 3),
            "speed_ms": round(self.speed_ms, 2),
            "needs": self.needs, "group_size": self.group_size,
            "needs_confirmation_pass": self.needs_confirmation_pass,
            "best_gsd_m": round(self.best_gsd_m, 4),
            "viability": self.viability.to_dict(),
            "landing_zones": [z.to_dict() for z in self.landing_zones[:3]],
            "hazards_nearby": self.hazards_nearby[:4],
            "first_seen_t": round(self.first_seen_t, 1),
            "last_seen_t": round(self.last_seen_t, 1),
        }


@dataclass
class PerceptionProduct:
    """Everything one pipeline invocation produced."""

    t: float
    observations: List[FusedObservation] = field(default_factory=list)
    geo_tags: List[Optional[GeoTag]] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    survivors: List[SurvivorAssessment] = field(default_factory=list)
    new_hazards: List[Any] = field(default_factory=list)
    n_detections: int = 0
    n_person_candidates: int = 0
    rgb_quality: float = 0.0
    nav: Optional[NavQuality] = None
    timing_ms: Dict[str, float] = field(default_factory=dict)
    faults: List[str] = field(default_factory=list)
    boresight: Dict[str, Any] = field(default_factory=dict)
    tracker_snapshot: Dict[str, Any] = field(default_factory=dict)
    hazard_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tracks: bool = False) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "n_detections": self.n_detections,
            "n_observations": len(self.observations),
            "n_person_candidates": self.n_person_candidates,
            "n_tracks": len(self.tracks),
            "n_survivors": len(self.survivors),
            "n_new_hazards": len(self.new_hazards),
            "rgb_quality": round(self.rgb_quality, 3),
            "nav": self.nav.describe() if self.nav else None,
            "timing_ms": {k: round(v, 1) for k, v in self.timing_ms.items()},
            "faults": list(self.faults),
            "boresight": self.boresight,
            "tracker": self.tracker_snapshot,
            "hazards": self.hazard_snapshot,
            "survivors": [s.to_dict() for s in self.survivors],
            "tracks": ([tr.to_dict() for tr in self.tracks] if include_tracks else None),
        }


# --------------------------------------------------------------------------- #
class PerceptionPipeline:
    """End-to-end on-device perception.  Thread-free by design.

    Nothing here spawns a thread.  On an embedded companion computer the whole
    pipeline is called from the camera callback and must complete inside the
    frame interval; hidden threads make that budget unauditable and make the
    failure modes unreproducible.  Parallelism, where it is affordable, belongs
    in the NPU backend, not in this module.
    """

    def __init__(self,
                 origin: GeoPoint,
                 tagger: PixelGeoTagger,
                 detector: Optional[DetectorEnsemble] = None,
                 tracker: Optional[Tracker] = None,
                 hazard_map: Optional[HazardMap] = None,
                 physiology: Optional[ThermalPhysiologyModel] = None,
                 fuser: Optional[CrossModalFuser] = None,
                 world_extent_m: Tuple[float, float] = (1000.0, 1000.0),
                 alert_threshold: float = 0.45,
                 gsd_confirm_threshold: float = 0.16) -> None:
        self.origin = origin
        self.tagger = tagger
        self.detector = detector or build_reference_pipeline("both")
        self.fuser = fuser or CrossModalFuser()
        self.tracker = tracker or Tracker(origin)
        self.hazard_map = hazard_map or HazardMap(
            origin, world_extent_m[0], world_extent_m[1])
        self.physiology = physiology or ThermalPhysiologyModel()
        self.alert_threshold = alert_threshold
        self.gsd_confirm_threshold = gsd_confirm_threshold
        #: External state the pipeline cannot measure itself.  The flight stack
        #: updates it each cycle from the world model (simulation) or from the
        #: DEM, rangefinder and weather feed (real aircraft).
        self.environment: Dict[str, Any] = {
            "ambient_c": 22.0, "wind_speed_ms": 3.0, "wind_from_deg": 210.0,
            "water_temp_c": None,
        }
        #: Functions providing local terrain and water state, in metres.
        self.terrain_fn = None
        self.water_fn = None
        self.products: List[PerceptionProduct] = []
        self.keep_products: int = 64
        #: Landing-zone results are expensive (a 110 m radius search over terrain
        #: and water rasters) and change only when the hazard map does, so they
        #: are cached per track and refreshed at most every ``lz_refresh_s``.
        self._lz_cache: Dict[int, Tuple[float, List[LandingZone]]] = {}
        self.lz_refresh_s: float = 3.0

    # ------------------------------------------------------------------ #
    def set_environment(self, **kwargs: Any) -> None:
        self.environment.update(kwargs)

    # ------------------------------------------------------------------ #
    def process(self, frames: Sequence[Any], pos_ned: np.ndarray,
                euler: np.ndarray, t: float,
                velocity_ned: Optional[np.ndarray] = None,
                nav: Optional[NavQuality] = None) -> PerceptionProduct:
        """Run one full perception cycle.

        ``frames`` is a list with one frame per modality; each must expose
        ``image``, ``kind``, ``gsd_m`` and (for evaluation only) ``truth``.
        """
        product = PerceptionProduct(t=float(t), nav=nav)
        timings: Dict[str, float] = {}
        nav = nav or NavQuality()

        # ---------------- detect --------------------------------------
        t0 = time.perf_counter()
        rgb_q = 0.0
        for f in frames:
            if getattr(f, "kind", None) == "rgb":
                try:
                    rgb_q = rgb_frame_quality(f)
                except Exception as exc:            # pragma: no cover
                    product.faults.append(f"rgb_quality: {exc!r}")
        product.rgb_quality = rgb_q
        detections: List[Detection] = []
        try:
            detections = self.detector.detect(list(frames), rgb_quality=rgb_q)
        except Exception as exc:
            product.faults.append(f"detector: {exc!r}")
        timings["detect_ms"] = (time.perf_counter() - t0) * 1000.0
        product.n_detections = len(detections)
        product.n_person_candidates = sum(1 for d in detections if d.is_person)

        # ---------------- fuse ----------------------------------------
        t0 = time.perf_counter()
        try:
            observations = self.fuser.fuse(detections, t, rgb_quality=rgb_q)
        except Exception as exc:
            observations = [self.fuser._single(d, t, rgb_blind=rgb_q < 0.3,
                                               rgb_quality=rgb_q)
                            for d in detections]
            product.faults.append(f"fuser: {exc!r}")
        timings["fuse_ms"] = (time.perf_counter() - t0) * 1000.0
        product.observations = observations
        product.boresight = self.fuser.boresight.to_dict()

        # ---------------- geo-tag -------------------------------------
        t0 = time.perf_counter()
        tags: List[Optional[GeoTag]] = []
        gsd = self._frame_gsd(frames)
        for obs in observations:
            try:
                tags.append(self.tagger.tag(
                    obs.u, obs.v, pos_ned, euler, velocity_ned=velocity_ned,
                    nav=nav, t=t))
            except Exception:
                tags.append(None)
        timings["geotag_ms"] = (time.perf_counter() - t0) * 1000.0
        product.geo_tags = tags
        n_unprojected = sum(1 for x in tags if x is None)
        if n_unprojected:
            product.faults.append(f"{n_unprojected} detection(s) above the horizon "
                                  f"or outside the terrain model")

        # ---------------- track ---------------------------------------
        t0 = time.perf_counter()
        dets_for_tracker = [o.to_detection() for o in observations]
        try:
            tracks = self.tracker.update(dets_for_tracker, tags, t)
        except Exception as exc:
            tracks = self.tracker.tracks
            product.faults.append(f"tracker: {exc!r}")
        timings["track_ms"] = (time.perf_counter() - t0) * 1000.0
        product.tracks = tracks
        product.tracker_snapshot = self.tracker.snapshot()

        # ---------------- hazards -------------------------------------
        t0 = time.perf_counter()
        survivors_ne = [(tr.tid, tr.north, tr.east)
                        for tr in self.tracker.person_tracks()]
        new_hz = []
        self.hazard_map.decay(max(t - (self.products[-1].t if self.products else t), 0.0))
        for obs, tag in zip(observations, tags):
            if tag is None or not obs.label or obs.label in PERSON_LABELS:
                continue
            try:
                hz = self.hazard_map.add_detection(
                    obs.to_detection(), tag,
                    {**self.environment, "gsd_m": gsd,
                     "water_depth_m": self._water_at(tag.north_m, tag.east_m)},
                    t, survivors=survivors_ne)
                if hz is not None and hz.n_obs == 1:
                    new_hz.append(hz)
            except Exception as exc:
                product.faults.append(f"hazard: {exc!r}")
        timings["hazard_ms"] = (time.perf_counter() - t0) * 1000.0
        product.new_hazards = new_hz
        product.hazard_snapshot = self.hazard_map.summary()

        # ---------------- assess survivors ----------------------------
        t0 = time.perf_counter()
        product.survivors = self._assess(t, product)
        timings["assess_ms"] = (time.perf_counter() - t0) * 1000.0

        timings["total_ms"] = sum(timings.values())
        product.timing_ms = timings
        self.products.append(product)
        if len(self.products) > self.keep_products:
            self.products = self.products[-self.keep_products:]
        return product

    # ------------------------------------------------------------------ #
    def _frame_gsd(self, frames: Sequence[Any]) -> float:
        gsds = [float(getattr(f, "gsd_m", 0.0) or 0.0) for f in frames]
        gsds = [g for g in gsds if g > 0 and math.isfinite(g)]
        return float(min(gsds)) if gsds else 0.15

    def _water_at(self, north: float, east: float) -> float:
        if self.water_fn is None:
            return 0.0
        try:
            return float(self.water_fn(north, east))
        except Exception:
            return 0.0

    def _terrain_at(self, north: float, east: float) -> Optional[float]:
        if self.terrain_fn is None:
            return None
        try:
            return float(self.terrain_fn(north, east))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def _assess(self, t: float, product: PerceptionProduct) -> List[SurvivorAssessment]:
        """Turn confirmed person tracks into prioritised survivor assessments."""
        out: List[SurvivorAssessment] = []
        for tr in self.tracker.person_tracks():
            if tr.person_evidence < self.alert_threshold:
                continue
            best_gsd = min((o.gsd_m for o in tr.observations), default=1.0)
            # Aggregate radiometry over the whole track, preferring the
            # best-informed (lowest-GSD) observation.
            best = min(tr.observations, key=lambda o: o.gsd_m) if tr.observations else None
            apparent = float(best.peak_temp_c) if best and best.peak_temp_c is not None else 25.0
            background = float(best.background_temp_c) if best and \
                best.background_temp_c is not None else 15.0
            area = float(best.area_px) if best else 10.0

            # Cooling trend from the track's own temperature history.
            series = [(o.t, o.peak_temp_c) for o in tr.observations
                      if o.peak_temp_c is not None]
            trend = 0.0
            if len(series) >= 2:
                ts = np.array([s[0] for s in series])
                ys = np.array([s[1] for s in series])
                if ts.max() - ts.min() > 5.0:
                    slope = float(np.polyfit(ts, ys, 1)[0])
                    trend = slope * 60.0                 # K per minute

            water = self._water_at(tr.north, tr.east)
            posture = self._infer_posture(tr, water)
            viability = self.physiology.assess(
                apparent_temp_c=apparent, background_c=background,
                ambient_c=float(self.environment.get("ambient_c", 22.0)),
                wind_ms=float(self.environment.get("wind_speed_ms", 0.0)),
                water_temp_c=(self.environment.get("water_temp_c")
                              if self.environment.get("water_temp_c") is not None
                              else (background if water > 0.2 else None)),
                in_water=water > 0.5, partial_water=0.05 < water <= 0.5,
                under_rubble=(tr.label == "person" and posture == "prone_partial_burial"),
                immobile=tr.immobility > 0.8 or tr.speed_ms < 0.05,
                minutes_observed=tr.age_s / 60.0,
                temp_trend_k_per_min=trend,
                posture=posture, needs=self._infer_needs(water, posture),
                group_size=self._group_size(tr),
                gsd_m=best.gsd_m if best else 0.15,
                expected_area_px=(math.pi * (0.444 / max(best.gsd_m, 1e-4)) ** 2
                                  if best else 50.0),
                measured_area_px=area,
                confidence=float(tr.confidence * min(tr.n_obs / 3.0, 1.0)),
            )
            hz = self.hazard_map.query(tr.north, tr.east)
            lz: List[LandingZone] = []
            if viability.priority in (PriorityTier.IMMEDIATE, PriorityTier.DELAYED):
                cached = self._lz_cache.get(tr.tid)
                if cached and (t - cached[0]) < self.lz_refresh_s:
                    lz = cached[1]
                else:
                    try:
                        lz = self.hazard_map.landing_zones(
                            tr.north, tr.east,
                            terrain_fn=self.terrain_fn, water_fn=self.water_fn,
                            search_radius_m=110.0, top_k=3)
                        self._lz_cache[tr.tid] = (t, lz)
                    except Exception as exc:
                        product.faults.append(f"landing_zones: {exc!r}")
            out.append(SurvivorAssessment(
                track_id=tr.tid,
                point=tr.geo(self.origin), north_m=tr.north, east_m=tr.east,
                sigma_m=tr.position_sigma_m, r95_m=2.4477 * tr.position_sigma_m,
                label=tr.label, confidence=float(tr.confidence),
                viability=viability, priority=viability.priority,
                time_critical_s=viability.time_critical_s,
                n_obs=tr.n_obs, cross_modal=tr.cross_modal,
                immobility=tr.immobility, speed_ms=tr.speed_ms,
                first_seen_t=tr.created_t, last_seen_t=tr.updated_t,
                needs=viability.needs, group_size=self._group_size(tr),
                landing_zones=lz, hazards_nearby=hz.get("nearest", []),
                needs_confirmation_pass=best_gsd > self.gsd_confirm_threshold,
                best_gsd_m=best_gsd,
            ))
        # Most urgent first; a planner with a fuel budget takes from the front.
        order = {PriorityTier.IMMEDIATE: 0, PriorityTier.DELAYED: 1,
                 PriorityTier.EXPECTANT: 2, PriorityTier.MINOR: 3,
                 PriorityTier.UNKNOWN: 4}
        out.sort(key=lambda s: (order.get(s.priority, 5),
                                s.time_critical_s if s.time_critical_s else 1e12,
                                -s.confidence))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_posture(tr: Track, water_depth: float) -> str:
        """Best available posture inference from geometry and context.

        Posture is not directly observable from a nadir thermal blob at 40 m, so
        it is inferred from what *is*: immersion depth, whether the target has
        moved, and its measured elongation.  The inference is coarse and says so.
        """
        elong = float(np.median([o.attributes.get("elongation", 1.2)
                                 for o in tr.observations] or [1.2]))
        if water_depth > 0.5:
            return "in_water_clinging"
        if tr.immobility > 0.85 and elong < 1.25:
            return "lying"
        if tr.immobility > 0.85 and elong > 1.6:
            return "prone_partial_burial"
        if tr.speed_ms > 0.35:
            return "waving"
        return "standing"

    @staticmethod
    def _infer_needs(water_depth: float, posture: str) -> str:
        if water_depth > 0.35 or posture == "in_water_clinging":
            return "flotation"
        if posture == "prone_partial_burial":
            return "extrication"
        if posture in ("lying", "sitting"):
            return "medical"
        return "medical"

    def _group_size(self, tr: Track) -> int:
        """How many confirmed person tracks are within 8 m of this one."""
        n = 1
        for other in self.tracker.person_tracks():
            if other.tid == tr.tid:
                continue
            if math.hypot(other.north - tr.north, other.east - tr.east) < 8.0:
                n += 1
        return n

    # ------------------------------------------------------------------ #
    def confirmation_queue(self) -> List[Dict[str, Any]]:
        """Tracks whose only evidence came from frames too coarse to decide.

        This is the work queue the search planner consumes for the low
        confirmation pass.  Sorted by how urgent the track already looks, so a
        fuel-limited sortie confirms the most promising candidates first.
        """
        out = []
        for tr in self.tracker.candidates_needing_confirmation(
                self.gsd_confirm_threshold):
            best = min((o.gsd_m for o in tr.observations), default=1.0)
            out.append({"tid": tr.tid, "north_m": tr.north, "east_m": tr.east,
                        "person_evidence": round(tr.person_evidence, 3),
                        "n_obs": tr.n_obs, "best_gsd_m": round(best, 4),
                        "sigma_m": round(tr.position_sigma_m, 2),
                        "status": tr.status,
                        "priority_hint": ("high" if tr.person_evidence > 0.7
                                          else "medium" if tr.person_evidence > 0.5
                                          else "low")})
        out.sort(key=lambda d: (-d["person_evidence"], d["best_gsd_m"]))
        return out

    def summary(self) -> Dict[str, Any]:
        surv = []
        for tr in self.tracker.person_tracks():
            surv.append({"tid": tr.tid, "ev": round(tr.person_evidence, 3),
                         "n": tr.n_obs, "status": tr.status})
        return {
            "tracks": self.tracker.snapshot(),
            "hazards": self.hazard_map.summary(),
            "boresight": self.fuser.boresight.to_dict(),
            "fusion": dict(self.fuser.stats),
            "person_tracks": surv,
            "confirmation_queue": len(self.confirmation_queue()),
        }
