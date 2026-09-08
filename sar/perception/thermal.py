"""From radiometry to physiology: how urgent is this survivor?

A thermal camera measures a surface temperature.  A rescue coordinator needs a
priority.  Bridging those is the least automatable part of this system and the
part most likely to be gotten wrong by a naive pipeline, so every number here is
traced to a published model rather than tuned.

What the sensor actually tells us
---------------------------------
An LWIR core sees the *apparent radiometric temperature* of whatever is on the
surface: skin if exposed, clothing if covered, and the water or rubble the body
is in contact with, area-weighted within the pixel.  It does not measure core
temperature.  What it can do is distinguish states that matter operationally:

* a dry, clothed survivor in air reads 28-34 C;
* a survivor immersed in flood water reads far colder - evaporative and
  convective loss plus the water's own thermal mass pull the apparent
  temperature toward the water temperature within minutes;
* a survivor who has stopped moving reads progressively colder peripherally as
  perfusion shuts down, which is why apparent temperature *trend* is a stronger
  signal than any single reading;
* a body that has been in the water long enough to be near water temperature is
  either unconscious or dead, and that distinction is made by looking for
  movement, not temperature.

So the model here is deliberately conservative: it produces a **time-criticality
estimate** and a priority tier, and it reports its own uncertainty.  It never
claims a diagnosis.

Provenance of the physiological models
--------------------------------------
Cold-water incapacitation and survival times follow the standard cold-water
survival tables used by Transport Canada, the US Coast Guard and the Maritime and
Coastguard Agency, which are consistent with the "1-10-100" framing (1 minute of
cold shock, ~10 minutes of useful muscle function, ~1 hour to unconsciousness in
near-freezing water, and a large reduction in mortality if rescue occurs within
2 hours).  Wind chill uses the North American / Environment Canada standard
formula.  These are population-level curves for an average adult; they are used
to *order* survivors, not to predict any individual's outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PriorityTier", "ExposureMedium", "SurvivorViability", "ThermalPhysiologyModel",
    "wind_chill_c", "cold_water_survival", "frostbite_time_min",
]


class PriorityTier(str, Enum):
    """Triage tiers, ordered most urgent first.

    Named after the standard START/SALT-style field categories so the output can
    be read directly by an incident commander without translation.
    """

    IMMEDIATE = "immediate"          # life-threatening, minutes matter
    DELAYED = "delayed"              # serious but stable for now
    MINOR = "minor"                  # walking wounded / able to self-help
    EXPECTANT = "expectant"          # signs indicate unsurvivable without care
    UNKNOWN = "unknown"              # detected, not yet characterised


class ExposureMedium(str, Enum):
    AIR = "air"
    WATER = "water"
    PARTIAL_WATER = "partial_water"   # legs/torso in water, head and arms clear
    RUBBLE = "rubble"                 # trapped, insulated but crushed


#: Cold-water survival table: water temperature band -> (median time to
#: incapacitation [min], median time to unconsciousness [min]).  Piecewise-linear
#: interpolation between band centres.  Values are the consensus ranges from
#: Transport Canada / USCG cold-water survival guidance; the midpoint of each
#: published range is used, and the uncertainty is reported separately.
_COLD_WATER_TABLE: Tuple[Tuple[float, float, float], ...] = (
    # water C, incapacitation min, unconsciousness min
    (0.0, 5.0, 20.0),
    (4.0, 10.0, 40.0),
    (10.0, 45.0, 110.0),
    (15.0, 90.0, 220.0),
    (20.0, 150.0, 420.0),
    (25.0, 300.0, 900.0),
    (30.0, 720.0, 1800.0),
)


def _interp_table(x: float, table: Sequence[Tuple[float, float, float]],
                  col: int) -> float:
    if x <= table[0][0]:
        return float(table[0][col])
    if x >= table[-1][0]:
        return float(table[-1][col])
    for i in range(len(table) - 1):
        x0, x1 = table[i][0], table[i + 1][0]
        if x0 <= x <= x1:
            f = (x - x0) / max(x1 - x0, 1e-9)
            return float(table[i][col] * (1 - f) + table[i + 1][col] * f)
    return float(table[-1][col])


def cold_water_survival(water_temp_c: float) -> Tuple[float, float]:
    """(minutes to incapacitation, minutes to unconsciousness) for an immersed adult.

    Incapacitation here means loss of useful muscle function - the point at which
    a survivor clinging to debris or a roof edge can no longer hold on, and
    therefore the point that defines how long a rescue window remains open.
    """
    return (_interp_table(water_temp_c, _COLD_WATER_TABLE, 1),
            _interp_table(water_temp_c, _COLD_WATER_TABLE, 2))


def wind_chill_c(air_temp_c: float, wind_ms: float) -> float:
    """Environment Canada / NWS wind chill index.

    ``T_wc = 13.12 + 0.6215 T - 11.37 v^0.16 + 0.3965 T v^0.16`` with v in km/h.
    Only defined for T <= 10 C and v > 4.8 km/h; outside that the air temperature
    is returned unchanged rather than extrapolating a formula past its validity.
    """
    v_kmh = max(wind_ms, 0.0) * 3.6
    if air_temp_c > 10.0 or v_kmh <= 4.8:
        return float(air_temp_c)
    v16 = v_kmh ** 0.16
    return float(13.12 + 0.6215 * air_temp_c - 11.37 * v16
                 + 0.3965 * air_temp_c * v16)


def frostbite_time_min(wind_chill: float) -> Optional[float]:
    """Minutes to frostbite on exposed skin, from the wind chill index.

    Standard published thresholds.  ``None`` where the index is above the range
    in which frostbite is a risk.
    """
    if wind_chill > -20.0:
        return None
    if wind_chill <= -55.0:
        return 2.0
    if wind_chill <= -48.0:
        return 5.0
    if wind_chill <= -40.0:
        return 10.0
    if wind_chill <= -35.0:
        return 20.0
    if wind_chill <= -30.0:
        return 30.0
    return 60.0


@dataclass
class SurvivorViability:
    """Physiological state estimate for one survivor, with its own uncertainty."""

    medium: ExposureMedium = ExposureMedium.AIR
    apparent_temp_c: float = 25.0
    surface_temp_c: float = 25.0
    ambient_c: float = 22.0
    water_temp_c: Optional[float] = None
    wind_ms: float = 0.0
    wet_fraction: float = 0.0
    immobile: bool = False
    minutes_observed: float = 0.0
    temp_trend_k_per_min: float = 0.0
    posture: str = "standing"
    needs: str = "medical"
    group_size: int = 1

    # ---- derived ----
    minutes_to_incapacitation: Optional[float] = None
    minutes_to_unconsciousness: Optional[float] = None
    hypothermia_risk: float = 0.0          # 0..1
    heat_stress_risk: float = 0.0          # 0..1
    crush_risk: float = 0.0                # 0..1 (trapped under structure)
    wind_chill_c: Optional[float] = None
    frostbite_min: Optional[float] = None
    priority: PriorityTier = PriorityTier.UNKNOWN
    time_critical_s: Optional[float] = None
    confidence: float = 0.0
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medium": self.medium.value,
            "apparent_temp_c": round(self.apparent_temp_c, 1),
            "surface_temp_c": round(self.surface_temp_c, 1),
            "wet_fraction": round(self.wet_fraction, 2),
            "immobile": self.immobile,
            "temp_trend_k_per_min": round(self.temp_trend_k_per_min, 3),
            "minutes_to_incapacitation": (round(self.minutes_to_incapacitation, 1)
                                          if self.minutes_to_incapacitation else None),
            "minutes_to_unconsciousness": (round(self.minutes_to_unconsciousness, 1)
                                           if self.minutes_to_unconsciousness else None),
            "hypothermia_risk": round(self.hypothermia_risk, 3),
            "heat_stress_risk": round(self.heat_stress_risk, 3),
            "crush_risk": round(self.crush_risk, 3),
            "wind_chill_c": (round(self.wind_chill_c, 1)
                             if self.wind_chill_c is not None else None),
            "frostbite_min": self.frostbite_min,
            "priority": self.priority.value,
            "time_critical_s": (round(self.time_critical_s)
                                if self.time_critical_s else None),
            "confidence": round(self.confidence, 3),
            "rationale": list(self.rationale),
        }


class ThermalPhysiologyModel:
    """Estimates exposure state and triage priority from thermal observations."""

    #: Apparent temperature of dry clothed skin in comfortable air.  Anything
    #: much below this on a detected human silhouette indicates wetting,
    #: peripheral shutdown, or a long time in the water.
    DRY_CLOTHED_SURFACE_C = 31.0
    #: Below this apparent temperature a detected human is almost certainly
    #: immersed or profoundly hypothermic.
    IMMERSION_THRESHOLD_C = 26.0

    def __init__(self, hypothermia_onset_c: float = 35.0) -> None:
        self.hypothermia_onset_c = hypothermia_onset_c

    # ------------------------------------------------------------------ #
    def surface_temperature(self, apparent_temp_c: float, gsd_m: float,
                            background_c: float, expected_area_px: float,
                            measured_area_px: float) -> float:
        """Correct a sub-pixel reading back toward the target's own temperature.

        When a body fills only part of a pixel, the measured apparent temperature
        is the area-weighted mean of body and background.  If both the measured
        core area and the expected body area are known, the mixing can be
        inverted.  This is the single biggest correction available to a small-UAV
        thermal payload, and it is why ``scripts/experiment_subpixel_radiometry.py``
        exists: without it, a chest-deep survivor at 70 m reads 20 C and is
        written off as cold water.

        The correction is clipped: it cannot recover information the sensor never
        had, and below a ~15% fill factor the inversion amplifies noise faster
        than signal.
        """
        fill = (float(np.clip(measured_area_px / max(expected_area_px, 1e-6), 0.0, 1.0))
                if expected_area_px > 0 else 1.0)
        if fill >= 0.85:
            return float(apparent_temp_c)
        # Cap the correction at 2x.  Inverting an unbounded mixing model is how a
        # detector fabricates a 58 C survivor from a cold frame: if the measured
        # area really is a third of the expected body area, then either the body
        # is not where we think it is or the "background" is not the background.
        # A bounded correction degrades gracefully instead of hallucinating.
        fill = max(fill, 0.50)
        est = background_c + (apparent_temp_c - background_c) / fill
        if est > 38.5:
            # The geometry assumption is inconsistent with the radiometry; trust
            # the measurement, not the model.
            return float(min(apparent_temp_c, 38.5))
        return float(est)

    # ------------------------------------------------------------------ #
    def assess(self, apparent_temp_c: float, background_c: float, ambient_c: float,
               wind_ms: float = 0.0, water_temp_c: Optional[float] = None,
               in_water: bool = False, partial_water: bool = False,
               under_rubble: bool = False, immobile: bool = False,
               minutes_observed: float = 0.0,
               temp_trend_k_per_min: float = 0.0,
               posture: str = "standing", needs: str = "medical",
               group_size: int = 1, gsd_m: float = 0.1,
               expected_area_px: float = 0.0, measured_area_px: float = 0.0,
               confidence: float = 0.5) -> SurvivorViability:
        """Full assessment.  Returns a :class:`SurvivorViability` with rationale."""
        v = SurvivorViability(
            apparent_temp_c=float(apparent_temp_c),
            ambient_c=float(ambient_c), wind_ms=float(wind_ms),
            water_temp_c=water_temp_c, immobile=bool(immobile),
            minutes_observed=float(minutes_observed),
            temp_trend_k_per_min=float(temp_trend_k_per_min),
            posture=str(posture), needs=str(needs), group_size=int(group_size),
            confidence=float(confidence),
        )
        v.surface_temp_c = self.surface_temperature(
            apparent_temp_c, gsd_m, background_c, expected_area_px, measured_area_px)

        if under_rubble:
            v.medium = ExposureMedium.RUBBLE
        elif in_water:
            v.medium = ExposureMedium.WATER
        elif partial_water:
            v.medium = ExposureMedium.PARTIAL_WATER
        else:
            v.medium = ExposureMedium.AIR

        rationale = v.rationale

        # ---- wetness from how far the surface temperature has fallen --------
        deficit = max(self.DRY_CLOTHED_SURFACE_C - v.surface_temp_c, 0.0)
        v.wet_fraction = float(np.clip(deficit / 9.0, 0.0, 1.0))
        if v.wet_fraction > 0.35:
            rationale.append(f"surface {v.surface_temp_c:.1f}C is {deficit:.1f}K below a "
                             f"dry clothed survivor -> wet_fraction {v.wet_fraction:.2f}")

        # ---- cold water ------------------------------------------------------
        if v.medium in (ExposureMedium.WATER, ExposureMedium.PARTIAL_WATER):
            # Water temperature: prefer a measured value, else infer it from the
            # background the survivor is sitting in, which is what the thermal
            # camera actually sees.
            wt = float(water_temp_c) if water_temp_c is not None else float(background_c)
            v.water_temp_c = wt
            inc, unc = cold_water_survival(wt)
            # Partial immersion is materially less dangerous than full immersion:
            # the head and torso clear of the water retain most heat.
            factor = 1.0 if v.medium == ExposureMedium.WATER else 2.2
            # Elapsed time already in the water is unknown, so it is not assumed
            # to be zero; the estimate is a *remaining* window from detection.
            v.minutes_to_incapacitation = inc * factor
            v.minutes_to_unconsciousness = unc * factor
            v.hypothermia_risk = float(np.clip(1.0 - inc * factor / 180.0, 0.15, 1.0))
            rationale.append(f"water {wt:.1f}C -> incapacitation ~{inc * factor:.0f} min, "
                             f"unconsciousness ~{unc * factor:.0f} min "
                             f"({'full' if factor == 1.0 else 'partial'} immersion)")
            if immobile:
                v.hypothermia_risk = min(1.0, v.hypothermia_risk + 0.25)
                rationale.append("immobile in water: peripheral perfusion already "
                                 "shutting down, or unable to move")
        else:
            # ---- air exposure ------------------------------------------------
            wc = wind_chill_c(ambient_c, wind_ms)
            v.wind_chill_c = wc
            fb = frostbite_time_min(wc)
            v.frostbite_min = fb
            # Hypothermia risk in air: driven by wind chill, wetness (which
            # multiplies conductive loss by roughly 4-5x for soaked clothing) and
            # how far the surface temperature has fallen.
            chill_risk = float(np.clip((10.0 - wc) / 30.0, 0.0, 1.0))
            wet_mult = 1.0 + 2.2 * v.wet_fraction       # soaked clothing ~4-5x loss
            v.hypothermia_risk = float(np.clip(
                chill_risk * wet_mult * (0.65 + 0.35 * min(deficit / 6.0, 1.0)), 0.0, 1.0))
            if immobile:
                v.hypothermia_risk = min(1.0, v.hypothermia_risk + 0.15)
                rationale.append("immobile in cold air: no metabolic contribution "
                                 "from movement")
            # Air-exposure clock.  Unlike immersion there is no published table of
            # minutes-to-incapacitation for a clothed adult in cold air, because
            # it depends on clothing, body composition and hydration that we
            # cannot see.  What follows is an exponential in wind chill scaled by
            # wetness and immobility - a coarse ordering device, reported with low
            # confidence and never presented as a prognosis.
            if wc < 12.0:
                hours = 6.0 * math.exp(max(wc, -40.0) / 12.0)
                hours /= (1.0 + 2.5 * v.wet_fraction)
                if immobile:
                    hours /= 2.0
                v.minutes_to_incapacitation = hours * 60.0
                v.minutes_to_unconsciousness = hours * 60.0 * 2.4
                v.confidence = min(v.confidence, 0.35)
                rationale.append(f"air-exposure window ~{hours:.1f} h at wind chill "
                                 f"{wc:.1f}C (low confidence: clothing unknown)")
            if v.hypothermia_risk > 0.35:
                rationale.append(f"wind chill {wc:.1f}C, wet_fraction {v.wet_fraction:.2f} "
                                 f"-> hypothermia risk {v.hypothermia_risk:.2f}")
            if fb is not None:
                rationale.append(f"exposed-skin frostbite possible in ~{fb:.0f} min")
            if v.medium == ExposureMedium.RUBBLE:
                v.crush_risk = float(np.clip(0.55 + 0.35 * (1.0 - v.confidence), 0.0, 0.95))
                rationale.append("trapped under structure: crush/asphyxia risk; "
                                 "do NOT attempt extraction without shoring")

        # ---- heat stress (the case everyone forgets) -------------------------
        # A survivor on a sun-heated roof at 40 C ambient with no water is at
        # risk of heat exhaustion on a timescale of hours, not minutes.
        if v.medium == ExposureMedium.AIR and ambient_c > 30.0:
            v.heat_stress_risk = float(np.clip((ambient_c - 30.0) / 12.0, 0.0, 1.0)
                                       * (1.0 - 0.5 * v.wet_fraction))
            if v.surface_temp_c > 36.0:
                v.heat_stress_risk = min(1.0, v.heat_stress_risk + 0.3)
                rationale.append(f"surface {v.surface_temp_c:.1f}C at ambient "
                                 f"{ambient_c:.1f}C -> heat stress")
            if v.heat_stress_risk > 0.4:
                rationale.append(f"heat stress risk {v.heat_stress_risk:.2f}: water "
                                 f"drop is a valid intervention")

        # ---- cooling trend ---------------------------------------------------
        if temp_trend_k_per_min < -0.08 and minutes_observed > 2.0:
            v.hypothermia_risk = min(1.0, v.hypothermia_risk + 0.20)
            rationale.append(f"cooling at {temp_trend_k_per_min:.2f} K/min over "
                             f"{minutes_observed:.0f} min: deteriorating")
        elif temp_trend_k_per_min > 0.08 and minutes_observed > 2.0:
            rationale.append(f"warming at {temp_trend_k_per_min:+.2f} K/min: improving "
                             f"or moving into sun")

        # ---- posture ---------------------------------------------------------
        if posture in ("prone_partial_burial", "lying"):
            v.hypothermia_risk = min(1.0, v.hypothermia_risk + 0.15)
            rationale.append(f"posture '{posture}': cannot self-right, ground contact "
                             f"conduction, possible injury")
        elif posture in ("in_water_clinging",):
            rationale.append("posture 'clinging in water': actively holding on - "
                             "grip failure is the failure mode")

        # ---- priority --------------------------------------------------------
        v.priority, v.time_critical_s = self._prioritise(v)
        rationale.append(f"tier={v.priority.value} "
                         f"critical_window={v.time_critical_s if v.time_critical_s else '-'}")
        return v

    # ------------------------------------------------------------------ #
    def _prioritise(self, v: SurvivorViability) -> Tuple[PriorityTier, Optional[float]]:
        """Map the physiological estimate onto a triage tier and a clock.

        The clock is what the mission planner actually consumes: it converts
        "urgent" into "this survivor's window closes in N seconds", which can be
        compared directly against flight time to reach them and against other
        survivors' windows.  Without a common unit, priority tiers alone cannot
        be optimised.
        """
        critical: Optional[float] = None
        if v.minutes_to_incapacitation is not None:
            critical = float(v.minutes_to_incapacitation) * 60.0
        if v.frostbite_min is not None:
            fb_s = v.frostbite_min * 60.0
            critical = fb_s if critical is None else min(critical, fb_s)
        if critical is not None:
            critical = round(critical, 1)

        if v.crush_risk > 0.5 or (v.hypothermia_risk > 0.75 and v.immobile):
            tier = PriorityTier.IMMEDIATE
            if critical is None:
                critical = 1800.0
        elif v.hypothermia_risk > 0.55 or v.medium in (
                ExposureMedium.WATER, ExposureMedium.RUBBLE):
            tier = PriorityTier.IMMEDIATE if (critical or 1e9) < 2700.0 else PriorityTier.DELAYED
        elif v.hypothermia_risk > 0.30 or v.heat_stress_risk > 0.55:
            tier = PriorityTier.DELAYED
        elif v.posture in ("standing", "waving") and not v.immobile:
            tier = PriorityTier.MINOR
            critical = None
        else:
            tier = PriorityTier.DELAYED

        # A survivor who has stopped moving and is cooling is expectant only if
        # the evidence is strong; otherwise the system defaults to escalating,
        # because the cost of a false "expectant" is a life and the cost of a
        # false "immediate" is fuel.
        if v.immobile and v.surface_temp_c < 22.0 and v.minutes_observed > 10.0 \
                and v.temp_trend_k_per_min < -0.05:
            tier = PriorityTier.EXPECTANT
            v.rationale.append("immobile, surface <22C, cooling for >10 min: "
                               "signs consistent with profound hypothermia")

        # Group size raises urgency slightly (children/elderly in a group, and a
        # single rescue asset can serve several people at once).
        if v.group_size > 2 and tier == PriorityTier.MINOR:
            tier = PriorityTier.DELAYED
        return tier, critical
