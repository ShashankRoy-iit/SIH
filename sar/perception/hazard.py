"""Hazard classification, risk mapping, and safe-zone inference.

Detection produces labelled boxes.  A disaster response needs a **risk field**:
where the fire is heading, which streets are impassable, where a conductor is
lying in flood water, and - critically for this system - where a relief payload
or a ground team can actually be put down.  This module turns detections plus
the world's own state into that field.

Severity is contextual, not intrinsic
-------------------------------------
The same label is not always the same danger:

* a downed conductor on dry ground blocks a road;
* the same conductor **in flood water electrifies the water**, which is the
  single highest-severity hazard a flood UAV can find, because the danger
  extends well beyond the visible conductor and kills without warning.  That
  conjunction is only detectable by fusing two independent detections, which is
  why the hazard map is built from the fusion of thermal and visible channels
  rather than from either alone.
* fire downwind of an occupied building is an evacuation trigger; the same fire
  with the wind blowing away is a monitoring task;
* a collapsed structure with a survivor inside is an immediate tier with a
  do-not-enter warning (secondary collapse), whereas the same structure empty is
  a mapping product.

So severity is computed from the label **and** the local context: wind vector,
water presence, distance to confirmed survivors, and time since last observation.

Plume and spread prediction
---------------------------
Fire and chemical hazards are advected.  A single frame gives a position; the
wind field gives a direction and rate.  This module projects an elliptical
exclusion zone downwind whose length grows with wind speed and time since
observation, and whose confidence decays as the projection ages.  That is enough
to keep the aircraft out of smoke column and enough to tell the incident
commander which way a fire is moving - and it is deliberately *not* a fire
behaviour model, which would need fuel moisture, slope and canopy data we do not
have.  The uncertainty is reported so nobody mistakes a projection for a
measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, local_to_wgs84
from sar.perception.detector import Detection, HAZARD_LABELS

__all__ = ["HazardSeverity", "Hazard", "HazardMap", "HazardClassifier", "LandingZone"]


class HazardSeverity(IntEnum):
    """Ordered so comparisons and colour maps work directly on the value."""

    NONE = 0
    INFORMATIONAL = 1     # mapping product, no immediate danger
    CAUTION = 2           # avoid, or proceed with care
    DANGEROUS = 3         # do not enter without protection
    CRITICAL = 4          # life-threatening now, extends beyond what is visible


#: Base severity per label, before contextual modification.
BASE_SEVERITY: Dict[str, HazardSeverity] = {
    "fire": HazardSeverity.CRITICAL,
    "chemical_plume": HazardSeverity.CRITICAL,
    "exposed_powerline": HazardSeverity.DANGEROUS,
    "collapsed_structure": HazardSeverity.DANGEROUS,
    "flood_water": HazardSeverity.DANGEROUS,
    "flash_flood_channel": HazardSeverity.CRITICAL,
    "landslide_zone": HazardSeverity.DANGEROUS,
    "damaged_structure": HazardSeverity.CAUTION,
    "debris_field": HazardSeverity.CAUTION,
    "blocked_road": HazardSeverity.CAUTION,
    "smoke": HazardSeverity.CAUTION,
    "vehicle": HazardSeverity.INFORMATIONAL,
    "hot_rock": HazardSeverity.INFORMATIONAL,
}


@dataclass
class Hazard:
    """One classified hazard with context, extent and a projected exclusion zone."""

    hid: int
    label: str
    north: float
    east: float
    severity: HazardSeverity = HazardSeverity.CAUTION
    confidence: float = 0.5
    area_m2: float = 0.0
    extent_m: float = 0.0
    first_seen_t: float = 0.0
    last_seen_t: float = 0.0
    n_obs: int = 0
    #: Context flags that drove the severity decision.  Reported verbatim to the
    #: operator, because a severity number without its reason is not actionable.
    context: List[str] = field(default_factory=list)
    #: Downwind projection: bearing and how far the exclusion zone extends.
    plume_bearing_deg: Optional[float] = None
    plume_length_m: float = 0.0
    plume_width_m: float = 0.0
    #: Survivors inside or near this hazard, by track id.
    affected_tracks: List[int] = field(default_factory=list)
    sigma_m: float = 5.0
    origin: Optional[GeoPoint] = None

    # ------------------------------------------------------------------ #
    @property
    def age_s(self) -> float:
        return max(self.last_seen_t - self.first_seen_t, 0.0)

    @property
    def exclusion_radius_m(self) -> float:
        """Keep-out radius for the aircraft and for ground teams."""
        base = max(math.sqrt(self.area_m2 / math.pi), 6.0)
        mult = {HazardSeverity.CRITICAL: 3.0, HazardSeverity.DANGEROUS: 2.0,
                HazardSeverity.CAUTION: 1.3, HazardSeverity.INFORMATIONAL: 1.0,
                HazardSeverity.NONE: 1.0}[self.severity]
        return float(base * mult + self.sigma_m)

    def geo(self, origin: GeoPoint) -> GeoPoint:
        return local_to_wgs84(self.north, self.east, 0.0, origin)

    def exclusion_polygon(self, origin: GeoPoint, n: int = 24) -> List[GeoPoint]:
        """Downwind-elongated keep-out polygon (a circle if there is no plume)."""
        out = []
        for i in range(n):
            th = 2.0 * math.pi * i / n
            r = self.exclusion_radius_m
            if self.plume_length_m > 0 and self.plume_bearing_deg is not None:
                b = math.radians(self.plume_bearing_deg)
                # Elongate along the plume axis, keep the crosswind width modest.
                along = math.cos(th - b)
                r = r * (1.0 + max(along, 0.0) * (self.plume_length_m / max(r, 1.0)))
            out.append(local_to_wgs84(self.north + r * math.cos(th),
                                      self.east + r * math.sin(th), 0.0, origin))
        return out

    def to_dict(self, origin: Optional[GeoPoint] = None) -> Dict[str, Any]:
        g = self.geo(origin) if origin else None
        return {
            "hid": self.hid, "label": self.label,
            "severity": self.severity.name, "severity_value": int(self.severity),
            "confidence": round(self.confidence, 3),
            "north_m": round(self.north, 1), "east_m": round(self.east, 1),
            "lat": g.lat if g else None, "lon": g.lon if g else None,
            "area_m2": round(self.area_m2, 1),
            "exclusion_radius_m": round(self.exclusion_radius_m, 1),
            "plume_bearing_deg": self.plume_bearing_deg,
            "plume_length_m": round(self.plume_length_m, 1),
            "context": list(self.context),
            "affected_tracks": list(self.affected_tracks),
            "n_obs": self.n_obs, "age_s": round(self.age_s, 1),
            "first_seen_t": self.first_seen_t, "last_seen_t": self.last_seen_t,
        }


def _batch_query(fn, norths: np.ndarray, easts: np.ndarray,
                 default: float = 0.0) -> np.ndarray:
    """Call ``fn`` once with arrays; fall back to per-element if it is scalar-only.

    The world model's samplers are vectorised, and in flight the DEM and flood
    rasters are too, so the array path is the normal one.  The scalar fallback
    exists so a simple test double or a single-point rangefinder lookup still
    works without the caller having to know which it is.
    """
    if fn is None:
        return np.full(norths.shape, default, dtype=float)
    try:
        out = np.asarray(fn(norths, easts), dtype=float)
        if out.shape == norths.shape and np.all(np.isfinite(out)):
            return np.nan_to_num(out, nan=default)
    except Exception:
        pass
    vals = []
    for n, e in zip(norths, easts):
        try:
            vals.append(float(fn(float(n), float(e))))
        except Exception:
            vals.append(default)
    return np.nan_to_num(np.asarray(vals, dtype=float), nan=default)


@dataclass
class LandingZone:
    """A ranked place to put a payload down, or to direct a ground team to."""

    north: float
    east: float
    score: float
    distance_to_target_m: float
    slope_deg: float = 0.0
    water_depth_m: float = 0.0
    hazard_penalty: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def geo(self, origin: GeoPoint) -> GeoPoint:
        return local_to_wgs84(self.north, self.east, 0.0, origin)

    def to_dict(self, origin: Optional[GeoPoint] = None) -> Dict[str, Any]:
        g = self.geo(origin) if origin else None
        return {"north_m": round(self.north, 1), "east_m": round(self.east, 1),
                "lat": g.lat if g else None, "lon": g.lon if g else None,
                "score": round(self.score, 3),
                "distance_to_target_m": round(self.distance_to_target_m, 1),
                "slope_deg": round(self.slope_deg, 2),
                "water_depth_m": round(self.water_depth_m, 2),
                "hazard_penalty": round(self.hazard_penalty, 2),
                "reasons": list(self.reasons)}


# --------------------------------------------------------------------------- #
class HazardClassifier:
    """Assigns context-dependent severity to detected hazards."""

    #: A conductor within this distance of standing water is assumed to be in it.
    POWERLINE_WATER_RADIUS_M = 12.0
    #: Fire within this distance of a survivor triggers an evacuation flag.
    FIRE_SURVIVOR_RADIUS_M = 45.0

    #: Fire spread rate as a fraction of wind speed.  A flaming front in
    #: grass/urban-edge fuel advances at roughly 0.05-0.3x the wind speed, NOT at
    #: the wind speed - only the smoke does.  Projecting flame at wind speed would
    #: produce a 400 m exclusion zone around a burning car and waste the sortie.
    FIRE_SPREAD_FRACTION = 0.12
    #: How far ahead the projection looks on a fresh observation [s].
    PROJECTION_HORIZON_S = 180.0

    def __init__(self, wind_advection_s: float = 600.0,
                 max_plume_length_m: float = 400.0) -> None:
        self.wind_advection_s = wind_advection_s
        self.max_plume_length_m = max_plume_length_m

    # ------------------------------------------------------------------ #
    def classify(self, label: str, north: float, east: float,
                 confidence: float, area_m2: float, gsd_m: float,
                 wind_speed_ms: float = 0.0, wind_from_deg: float = 0.0,
                 water_depth_m: float = 0.0, water_within_m: Optional[float] = None,
                 survivors_within_m: Optional[float] = None,
                 seconds_since_observed: float = 0.0) -> Tuple[HazardSeverity, List[str]]:
        """Return ``(severity, context_reasons)``.

        ``water_within_m`` and ``survivors_within_m`` are distances to the
        nearest water body and nearest confirmed survivor.  Passing them in
        rather than looking them up keeps this class free of world dependencies
        and therefore unit-testable.
        """
        sev = BASE_SEVERITY.get(label, HazardSeverity.INFORMATIONAL)
        ctx: List[str] = []

        # ---- electrocution: the conjunction that only fusion can see --------
        if label == "exposed_powerline":
            in_water = (water_depth_m > 0.05
                        or (water_within_m is not None
                            and water_within_m < self.POWERLINE_WATER_RADIUS_M))
            if in_water:
                sev = HazardSeverity.CRITICAL
                ctx.append("conductor in or adjacent to standing water: assume the "
                           "water is energised. Exclusion extends beyond the visible "
                           "span; do not wade, do not send a boat.")
                area_m2 = max(area_m2, math.pi * 25.0 ** 2)
            else:
                ctx.append("conductor down on dry ground: blocks passage, low "
                           "electrocution risk unless it becomes wet")

        # ---- advection: fire, smoke, chemical -------------------------------
        if label in ("fire", "smoke", "chemical_plume") and wind_speed_ms > 0.4:
            # wind_from_deg is the direction the wind comes FROM (meteorological).
            # Plume travels toward wind_from + 180.
            to_deg = (float(wind_from_deg) + 180.0) % 360.0
            # Age the projection: how long since we last looked at it.
            length, width = self._plume_extent(label, wind_speed_ms, area_m2,
                                               seconds_since_observed)
            rate_txt = ("smoke/plume advection at wind speed"
                        if label != "fire" else
                        f"flame front at {self.FIRE_SPREAD_FRACTION * wind_speed_ms:.2f} m/s")
            ctx.append(f"advecting toward {to_deg:.0f} deg ({rate_txt}, wind "
                       f"{wind_speed_ms:.1f} m/s): projected {length:.0f} m "
                       f"exclusion downwind over the next "
                       f"{self.PROJECTION_HORIZON_S:.0f} s + observed age "
                       f"(projection, not measurement)")
            if label == "fire" and survivors_within_m is not None \
                    and survivors_within_m < max(length, self.FIRE_SURVIVOR_RADIUS_M):
                sev = HazardSeverity.CRITICAL
                ctx.append(f"survivor {survivors_within_m:.0f} m away and downwind of "
                           f"the fire front: evacuation, not just monitoring")

        # ---- water ----------------------------------------------------------
        if label in ("flood_water", "flash_flood_channel"):
            if water_depth_m > 1.5:
                sev = HazardSeverity.CRITICAL
                ctx.append(f"depth {water_depth_m:.1f} m: unsurvivable without "
                           f"flotation, and too deep for wading rescue")
            elif water_depth_m > 0.5:
                ctx.append(f"depth {water_depth_m:.1f} m: sweeps people off their "
                           f"feet (0.5 m is the accepted wading limit)")
            else:
                ctx.append(f"depth {water_depth_m:.1f} m: shallow but hides debris "
                           f"and open drains")
            if label == "flash_flood_channel":
                sev = HazardSeverity.CRITICAL
                ctx.append("in a flash-flood channel: rising faster than a person "
                           "can walk out. Highest time criticality of any hazard.")

        # ---- structural -----------------------------------------------------
        if label == "collapsed_structure":
            if survivors_within_m is not None and survivors_within_m < 15.0:
                ctx.append("survivor within 15 m of a collapse: void-space search "
                           "likely; DO NOT walk or land on the debris pile "
                           "(secondary collapse)")
            else:
                ctx.append("collapsed structure: assume void spaces and survivors "
                           "underneath until searched")
        elif label == "damaged_structure":
            ctx.append("damaged but standing: unsafe to enter, usable as a "
                       "landmark and as shelter from wind")

        # ---- sub-pixel honesty ---------------------------------------------
        # A hazard seen once from high altitude has an area that is a lower
        # bound; say so rather than reporting a precise number.
        if gsd_m > 0.18:
            ctx.append(f"observed at {gsd_m:.2f} m/px: extent is a lower bound, "
                       f"confirm with a low pass before acting")

        if confidence < 0.4:
            sev = HazardSeverity(min(int(sev), int(HazardSeverity.CAUTION)))
            ctx.append(f"low detection confidence {confidence:.2f}: severity capped "
                       f"at CAUTION until re-observed")
        return sev, ctx

    # ------------------------------------------------------------------ #
    def plume(self, label: str, wind_speed_ms: float, wind_from_deg: float,
              area_m2: float, seconds_since_observed: float = 0.0
              ) -> Tuple[Optional[float], float, float]:
        """(bearing_deg, length_m, width_m) of the downwind exclusion."""
        if label not in ("fire", "smoke", "chemical_plume") or wind_speed_ms < 0.4:
            return None, 0.0, 0.0
        to_deg = (float(wind_from_deg) + 180.0) % 360.0
        length, width = self._plume_extent(label, wind_speed_ms, area_m2,
                                           seconds_since_observed)
        return to_deg, length, width

    def _plume_extent(self, label: str, wind_speed_ms: float, area_m2: float,
                      seconds_since_observed: float) -> Tuple[float, float]:
        """Downwind extent of the exclusion zone, and its crosswind width.

        Smoke and chemical plumes travel at the wind speed.  A flame front
        travels at a fraction of it.  Both are projected forward over a fixed
        horizon plus however long it has been since we looked, so an old
        observation produces a larger (and less certain) zone rather than a
        stale small one.
        """
        rate = (wind_speed_ms * self.FIRE_SPREAD_FRACTION if label == "fire"
                else wind_speed_ms)
        age = min(max(seconds_since_observed, 0.0), self.wind_advection_s)
        length = min(rate * (age + self.PROJECTION_HORIZON_S), self.max_plume_length_m)
        width = max(8.0, 0.35 * length + math.sqrt(max(area_m2, 1.0)))
        return length, width


# --------------------------------------------------------------------------- #
class HazardMap:
    """Accumulated hazard picture on a metric grid, with safe-zone inference.

    The map is a persistent product of the sortie, not a per-frame list.  It is
    what the command centre displays, what the route planner penalises, and what
    gets exported as the mission's geospatial deliverable (GeoJSON / PNG).
    """

    def __init__(self, origin: GeoPoint, north_m: float, east_m: float,
                 cell_m: float = 4.0,
                 classifier: Optional[HazardClassifier] = None) -> None:
        self.origin = origin
        self.north_m = float(north_m)
        self.east_m = float(east_m)
        self.cell_m = float(cell_m)
        self.shape = (int(math.ceil(north_m / cell_m)) + 1,
                      int(math.ceil(east_m / cell_m)) + 1)
        self.classifier = classifier or HazardClassifier()
        #: Severity raster (0..4), float so it can be blended.
        self.severity = np.zeros(self.shape, dtype=np.float32)
        #: Per-label presence probability, decays with time.
        self.presence: Dict[str, np.ndarray] = {}
        self.hazards: List[Hazard] = []
        self._next_hid = 1

    # ------------------------------------------------------------------ #
    def _idx(self, north: float, east: float) -> Tuple[int, int]:
        r = int(round(north / self.cell_m))
        c = int(round(east / self.cell_m))
        return (int(np.clip(r, 0, self.shape[0] - 1)),
                int(np.clip(c, 0, self.shape[1] - 1)))

    def _ensure(self, label: str) -> np.ndarray:
        if label not in self.presence:
            self.presence[label] = np.zeros(self.shape, dtype=np.float32)
        return self.presence[label]

    # ------------------------------------------------------------------ #
    def add_detection(self, det: Detection, tag, world_state: Dict[str, Any],
                      t: float, survivors: Sequence[Tuple[int, float, float]] = ()
                      ) -> Optional[Hazard]:
        """Fuse one hazard detection into the map.

        ``world_state`` carries the local context the classifier needs:
        ``water_depth_m``, ``water_within_m``, ``wind_speed_ms``,
        ``wind_from_deg``, ``gsd_m``.  ``survivors`` is a sequence of
        ``(track_id, north, east)``.
        """
        if det.label not in HAZARD_LABELS:
            return None
        if tag is None or not math.isfinite(getattr(tag.point, "lat", float("nan"))):
            return None
        # GeoTag carries local metres directly; fall back to converting the
        # geodetic point for tags produced by an older/other tagger.
        if math.isfinite(getattr(tag, "north_m", float("nan"))):
            north, east = float(tag.north_m), float(tag.east_m)
        else:
            from sar.core.geo import wgs84_to_local
            n_, e_, _ = wgs84_to_local(tag.point, self.origin)
            north, east = float(n_), float(e_)

        area_px = max(det.area_px, 1.0)
        gsd = float(getattr(tag, "gsd_m", 0.0) or world_state.get("gsd_m", 0.1))
        area_m2 = area_px * gsd * gsd
        extent_m = math.sqrt(area_m2 / math.pi) * 2.0

        # Nearest survivor distance, for context.
        surv_within = None
        affected: List[int] = []
        if survivors:
            dists = [(math.hypot(north - sn, east - se), tid) for tid, sn, se in survivors]
            dists.sort()
            surv_within = dists[0][0]
            affected = [tid for d, tid in dists if d < 60.0]

        sev, ctx = self.classifier.classify(
            det.label, north, east, det.score, area_m2, gsd,
            wind_speed_ms=float(world_state.get("wind_speed_ms", 0.0)),
            wind_from_deg=float(world_state.get("wind_from_deg", 0.0)),
            water_depth_m=float(world_state.get("water_depth_m", 0.0)),
            water_within_m=world_state.get("water_within_m"),
            survivors_within_m=surv_within,
            seconds_since_observed=0.0,
        )
        bearing, plume_len, plume_w = self.classifier.plume(
            det.label, float(world_state.get("wind_speed_ms", 0.0)),
            float(world_state.get("wind_from_deg", 0.0)), area_m2)

        existing = self._find_existing(det.label, north, east)
        if existing is not None:
            existing.n_obs += 1
            existing.last_seen_t = t
            # Grow the extent, never shrink it: a hazard that appears smaller on
            # a later frame is more likely to be partially occluded than to have
            # retreated, and under-reporting a fire's size is the worse error.
            existing.area_m2 = max(existing.area_m2, area_m2)
            existing.extent_m = max(existing.extent_m, extent_m)
            existing.confidence = min(0.99, existing.confidence + 0.5 * det.score
                                      * (1.0 - existing.confidence))
            existing.severity = HazardSeverity(max(int(existing.severity), int(sev)))
            for c in ctx:
                if c not in existing.context:
                    existing.context.append(c)
            if bearing is not None:
                existing.plume_bearing_deg = bearing
                existing.plume_length_m = max(existing.plume_length_m, plume_len)
                existing.plume_width_m = plume_w
            for tid in affected:
                if tid not in existing.affected_tracks:
                    existing.affected_tracks.append(tid)
            existing.sigma_m = float(getattr(tag, "sigma_m", 5.0))
            self._stamp(existing, t)
            return existing

        hz = Hazard(hid=self._next_hid, label=det.label, north=north, east=east,
                    severity=sev, confidence=float(det.score), area_m2=area_m2,
                    extent_m=extent_m, first_seen_t=t, last_seen_t=t, n_obs=1,
                    context=ctx, plume_bearing_deg=bearing, plume_length_m=plume_len,
                    plume_width_m=plume_w, affected_tracks=affected,
                    sigma_m=float(getattr(tag, "sigma_m", 5.0)), origin=self.origin)
        self._next_hid += 1
        self.hazards.append(hz)
        self._stamp(hz, t)
        return hz

    def _find_existing(self, label: str, north: float, east: float) -> Optional[Hazard]:
        best = None
        best_d = 1e18
        for h in self.hazards:
            if h.label != label:
                continue
            d = math.hypot(h.north - north, h.east - east)
            tol = max(1.6 * math.sqrt(max(h.area_m2, 1.0) / math.pi), 14.0)
            if d <= tol and d < best_d:
                best, best_d = h, d
        return best

    def _stamp(self, hz: Hazard, t: float) -> None:
        """Paint a hazard onto the severity and presence rasters."""
        layer = self._ensure(hz.label)
        r, c = self._idx(hz.north, hz.east)
        rad_cells = max(1, int(math.ceil(hz.exclusion_radius_m / self.cell_m)))
        r0, r1 = max(0, r - rad_cells), min(self.shape[0], r + rad_cells + 1)
        c0, c1 = max(0, c - rad_cells), min(self.shape[1], c + rad_cells + 1)
        if r0 >= r1 or c0 >= c1:
            return
        rr, cc = np.mgrid[r0:r1, c0:c1]
        d = np.hypot((rr - r) * self.cell_m, (cc - c) * self.cell_m)
        core = float(hz.severity) * min(hz.confidence + 0.25, 1.0)
        falloff = np.clip(1.0 - d / max(hz.exclusion_radius_m, 1e-6), 0.0, 1.0)
        # Plume elongation along the wind.
        if hz.plume_length_m > 0 and hz.plume_bearing_deg is not None:
            b = math.radians(hz.plume_bearing_deg)
            dn = (rr - r) * self.cell_m
            de = (cc - c) * self.cell_m
            along = dn * math.cos(b) + de * math.sin(b)
            cross = -dn * math.sin(b) + de * math.cos(b)
            in_plume = (along > 0) & (along < hz.plume_length_m) & \
                       (np.abs(cross) < max(hz.plume_width_m, 1.0) / 2.0)
            plume_sev = 0.6 * float(hz.severity)
            falloff = np.maximum(falloff, np.where(in_plume, 0.85, 0.0))
            self.severity[r0:r1, c0:c1] = np.maximum(
                self.severity[r0:r1, c0:c1], np.where(in_plume, plume_sev, 0.0))
        self.severity[r0:r1, c0:c1] = np.maximum(
            self.severity[r0:r1, c0:c1], (core * falloff).astype(np.float32))
        layer[r0:r1, c0:c1] = np.maximum(
            layer[r0:r1, c0:c1], (hz.confidence * falloff).astype(np.float32))

    # ------------------------------------------------------------------ #
    def decay(self, dt_s: float, half_life_s: float = 1800.0) -> None:
        """Fade un-refreshed hazards.

        A hazard we have not seen for a while is not gone - fires do not go out
        and floods do not drain - but our *knowledge* of it is stale, and the map
        must show that rather than presenting a half-hour-old fire as current.
        Severity decays toward a floor of 60% of its peak; it never reaches zero.
        """
        if dt_s <= 0:
            return
        f = 0.5 ** (dt_s / max(half_life_s, 1.0))
        for layer in self.presence.values():
            layer *= f
        floor = 0.6
        self.severity = np.maximum(self.severity * f,
                                   np.where(self.severity > 0.05,
                                            floor * self.severity, 0.0)).astype(np.float32)

    # ------------------------------------------------------------------ #
    def query(self, north: float, east: float) -> Dict[str, Any]:
        r, c = self._idx(north, east)
        sev = float(self.severity[r, c])
        labels = sorted(((lab, float(layer[r, c])) for lab, layer in self.presence.items()
                         if layer[r, c] > 0.05), key=lambda kv: -kv[1])
        return {"severity": sev,
                "severity_name": HazardSeverity(int(round(min(sev, 4)))).name,
                "labels": labels,
                "nearest": self._nearest_summary(north, east)}

    def _nearest_summary(self, north: float, east: float, k: int = 3) -> List[Dict[str, Any]]:
        out = []
        for h in sorted(self.hazards, key=lambda h: math.hypot(h.north - north, h.east - east))[:k]:
            out.append({"hid": h.hid, "label": h.label,
                        "severity": h.severity.name,
                        "distance_m": round(math.hypot(h.north - north, h.east - east), 1),
                        "exclusion_m": round(h.exclusion_radius_m, 1)})
        return out

    def risk_field(self, floor: float = 0.0) -> np.ndarray:
        """0..1 traversability penalty field for the route planner."""
        return np.clip((self.severity / 4.0) * (1.0 + floor), 0.0, 1.0)

    def no_fly_zones(self, min_severity: HazardSeverity = HazardSeverity.DANGEROUS
                     ) -> List[Tuple[float, float, float, str]]:
        """(north, east, radius, label) keep-out circles for the avoidance layer."""
        return [(h.north, h.east, h.exclusion_radius_m, h.label)
                for h in self.hazards if h.severity >= min_severity]

    # ------------------------------------------------------------------ #
    def landing_zones(self, target_north: float, target_east: float,
                      terrain_fn=None, water_fn=None,
                      search_radius_m: float = 120.0, cell_step_m: float = 6.0,
                      max_slope_deg: float = 12.0, max_water_depth_m: float = 0.10,
                      top_k: int = 4) -> List[LandingZone]:
        """Rank candidate drop / landing / rendezvous sites near a survivor.

        This is what makes the payload-drop concept usable: a life jacket or a
        water bottle dropped into a fire plume, onto a collapsing roof, or into
        2 m of water is worse than not dropped at all.  Candidates are scored on
        hazard clearance first and distance second, because a site 40 m away and
        safe beats a site 8 m away and electrified.

        ``terrain_fn(north, east) -> height`` and ``water_fn(north, east) -> depth``
        come from the world in simulation and from a DEM plus the flood raster in
        flight.  Both are optional: without them the function scores on the hazard
        map alone.
        """
        # Pass 1: enumerate candidate cells.  Pass 2: query terrain and water in
        # ONE vectorised call each.  The scalar-per-cell version took 425 ms per
        # assessment because each world sample rebuilt its own bilinear index; a
        # survivor with an IMMEDIATE tier must not cost half a second to plan a
        # drop zone for.
        half = int(math.ceil(search_radius_m / cell_step_m))
        cr, cc = self._idx(target_north, target_east)
        cand: List[Tuple[int, int, float]] = []
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r, c = cr + dr, cc + dc
                if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
                    continue
                n_, e_ = r * self.cell_m, c * self.cell_m
                d = math.hypot(n_ - target_north, e_ - target_east)
                if d > search_radius_m:
                    continue
                sev = float(self.severity[r, c])
                if sev >= HazardSeverity.DANGEROUS:
                    continue
                cand.append((r, c, d))
        if not cand:
            return []
        ns = np.array([c_[0] * self.cell_m for c_ in cand])
        es = np.array([c_[1] * self.cell_m for c_ in cand])
        depths = _batch_query(water_fn, ns, es, default=0.0)
        if terrain_fn is not None:
            h0 = _batch_query(terrain_fn, ns, es, default=0.0)
            hx = _batch_query(terrain_fn, ns + self.cell_m, es, default=0.0)
            hy = _batch_query(terrain_fn, ns, es + self.cell_m, default=0.0)
            slopes = np.degrees(np.arctan(
                np.hypot(hx - h0, hy - h0) / max(self.cell_m, 1e-6)))
        else:
            slopes = np.zeros(len(cand))

        out: List[LandingZone] = []
        for i, ((r, c, d), depth, slope) in enumerate(zip(cand, depths, slopes)):
            sev = float(self.severity[r, c])
            if depth > max_water_depth_m or slope > max_slope_deg:
                continue
            # Score: hazard clearance dominates, distance breaks ties.  A site
            # 40 m away and safe beats a site 8 m away and electrified.
            score = (1.0 - 0.25 * sev) * math.exp(-d / max(search_radius_m, 1.0) * 2.2)
            score *= (1.0 - 0.35 * min(slope / max(max_slope_deg, 1e-6), 1.0))
            reasons = []
            if sev > 0:
                reasons.append(f"severity {sev:.1f} in cell")
            if depth > 0:
                reasons.append(f"shallow water {depth:.2f} m")
            if slope > 6.0:
                reasons.append(f"slope {slope:.1f} deg")
            if not reasons:
                reasons.append("clear of mapped hazards")
            out.append(LandingZone(north=float(ns[i]), east=float(es[i]),
                                   score=float(score), distance_to_target_m=float(d),
                                   slope_deg=float(slope), water_depth_m=float(depth),
                                   hazard_penalty=float(sev), reasons=reasons))
        out.sort(key=lambda z: -z.score)
        # Non-maximum suppression so the top-k are not four adjacent cells.
        kept: List[LandingZone] = []
        for z in out:
            if all(math.hypot(z.north - k.north, z.east - k.east) > 15.0 for k in kept):
                kept.append(z)
            if len(kept) >= top_k:
                break
        return kept

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        by_label: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        for h in self.hazards:
            by_label[h.label] = by_label.get(h.label, 0) + 1
            by_sev[h.severity.name] = by_sev.get(h.severity.name, 0) + 1
        return {
            "n_hazards": len(self.hazards),
            "by_label": by_label,
            "by_severity": by_sev,
            "cells_above_dangerous": int((self.severity >= HazardSeverity.DANGEROUS).sum()),
            "grid_shape": list(self.shape),
            "cell_m": self.cell_m,
            "affected_tracks": sorted({tid for h in self.hazards
                                       for tid in h.affected_tracks}),
        }

    def export_geojson(self) -> Dict[str, Any]:
        """GeoJSON FeatureCollection for the dashboard and for hand-off.

        A disaster-management agency does not want a Python object; they want
        something that opens in QGIS.  This is the mission's geospatial
        deliverable.
        """
        feats = []
        for h in self.hazards:
            g = h.geo(self.origin)
            poly = h.exclusion_polygon(self.origin)
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [g.lon, g.lat]},
                "properties": h.to_dict(self.origin),
            })
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": [[[p.lon, p.lat] for p in poly]
                                             + [[poly[0].lon, poly[0].lat]]]},
                "properties": {"hid": h.hid, "kind": "exclusion_zone",
                               "label": h.label, "severity": h.severity.name},
            })
        return {"type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": feats}
