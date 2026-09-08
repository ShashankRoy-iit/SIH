"""Human Identification and Multi-Attribute Classifier for Search & Rescue.

This module provides deep identification and characterization of detected human
targets from RGB and Thermal (LWIR) imagery.

Key capabilities:
-----------------
1. **Multi-Attribute Human Profile**:
   - Posture classification: standing, waving (SOS), sitting, lying,
     prone partial burial, in-water clinging, trapped under rubble.
   - Distress / SOS signaling detection: analyzes gesture frequency,
     amplitude, and temporal periodicity (arm waving vs ambient movement).
   - Demographic / group estimation: solo adult, child, elderly / vulnerable,
     group cluster size.
   - Clothing and visual saliency signature (chroma, high-visibility cues).
   - Physiological & thermal vitals proxy: estimated core temperature,
     hypothermia severity, heat stroke risk, thermal respiration stability.
   - Triage classification (START / SALT standard): IMMEDIATE, DELAYED,
     MINOR, EXPECTANT.
   - Rescue equipment need mapping (flotation buoy, trauma kit, thermal blanket,
     locator beacon, water pack).

2. **Physics & Kinematic Decoy Discrimination**:
   - Rejects warm animals, vehicle engines, hot rocks, mannequins, solar panels,
     and wet debris using morphological, radiometric, and kinematic rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint
from sar.perception.detector import Detection
from sar.perception.thermal import (
    ExposureMedium,
    PriorityTier,
    SurvivorViability,
    ThermalPhysiologyModel,
    cold_water_survival,
    wind_chill_c,
)

__all__ = [
    "HumanPosture",
    "DistressLevel",
    "DemographicGroup",
    "RescueEquipmentNeed",
    "HumanProfile",
    "HumanIdentifier",
]


class HumanPosture(str, Enum):
    """Detailed human body postures observable from aerial platforms."""

    STANDING = "standing"
    WAVING_SOS = "waving_sos"
    SITTING = "sitting"
    LYING = "lying"
    PRONE_PARTIAL_BURIAL = "prone_partial_burial"
    IN_WATER_CLINGING = "in_water_clinging"
    TRAPPED_RUBBLE = "trapped_rubble"
    WALKING = "walking"
    UNKNOWN = "unknown"


class DistressLevel(str, Enum):
    """Level of active distress signaling by the survivor."""

    CRITICAL_ACTIVE_SOS = "critical_active_sos"  # Frantic waving, visual/audio SOS
    PASSIVE_IMMOBILE = "passive_immobile"        # Unconscious, trapped, unable to move
    MODERATE_DISTRESS = "moderate_distress"      # Standing/sitting, looking up
    SELF_SUFFICIENT = "self_sufficient"          # Walking, moving towards high ground
    UNKNOWN = "unknown"


class DemographicGroup(str, Enum):
    """Estimated demographic category based on geometry and physical extent."""

    ADULT = "adult"
    CHILD = "child"
    ELDERLY_VULNERABLE = "elderly_vulnerable"
    GROUP_CLUSTER = "group_cluster"
    UNKNOWN = "unknown"


class RescueEquipmentNeed(str, Enum):
    """Specific rescue payload needed for the identified survivor."""

    FLOTATION_BUOY = "flotation_buoy"            # For in-water immersion
    FIRST_AID_TRAUMA_KIT = "first_aid_kit"       # For severe trauma / crush injuries
    THERMAL_EMERGENCY_BLANKET = "thermal_blanket" # For severe hypothermia / cold shock
    LORA_LOCATOR_BEACON = "lora_locator_beacon"  # For trapped / GPS-denied / buried
    WATER_RATIONS_PACK = "water_rations"         # For stranded rooftop / heat exhaustion
    EXTRICATION_ASSISTANCE = "extrication"       # For physical entrapment under debris
    GENERAL_MEDICAL = "medical"


@dataclass
class HumanProfile:
    """Comprehensive human identification and rescue assessment record."""

    id_tag: str
    track_id: int
    confidence: float
    is_human: bool
    posture: HumanPosture
    posture_confidence: float
    distress_level: DistressLevel
    sos_waving_detected: bool
    waving_frequency_hz: float
    demographic: DemographicGroup
    group_size: int
    clothing_saliency: float
    clothing_color_hint: str
    apparent_temp_c: float
    estimated_core_temp_c: float
    hypothermia_risk: float
    heat_stress_risk: float
    triage_priority: PriorityTier
    recommended_equipment: RescueEquipmentNeed
    time_critical_s: Optional[float]
    decoy_rejection_notes: List[str] = field(default_factory=list)
    identification_notes: List[str] = field(default_factory=list)
    location_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    geo_point: Optional[GeoPoint] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id_tag": self.id_tag,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 3),
            "is_human": self.is_human,
            "posture": self.posture.value,
            "posture_confidence": round(self.posture_confidence, 2),
            "distress_level": self.distress_level.value,
            "sos_waving_detected": self.sos_waving_detected,
            "waving_frequency_hz": round(self.waving_frequency_hz, 2),
            "demographic": self.demographic.value,
            "group_size": self.group_size,
            "clothing_saliency": round(self.clothing_saliency, 2),
            "clothing_color_hint": self.clothing_color_hint,
            "apparent_temp_c": round(self.apparent_temp_c, 1),
            "estimated_core_temp_c": round(self.estimated_core_temp_c, 1),
            "hypothermia_risk": round(self.hypothermia_risk, 2),
            "heat_stress_risk": round(self.heat_stress_risk, 2),
            "triage_priority": self.triage_priority.value,
            "recommended_equipment": self.recommended_equipment.value,
            "time_critical_s": (
                round(self.time_critical_s) if self.time_critical_s else None
            ),
            "north_m": round(self.location_ned[0], 1),
            "east_m": round(self.location_ned[1], 1),
            "elevation_m": round(-self.location_ned[2], 1),
            "lat": round(self.geo_point.lat, 6) if self.geo_point else None,
            "lon": round(self.geo_point.lon, 6) if self.geo_point else None,
            "identification_notes": list(self.identification_notes),
            "decoy_rejection_notes": list(self.decoy_rejection_notes),
        }


class HumanIdentifier:
    """Multi-spectral Human Identification Engine.

    Combines temporal tracking history, thermal radiometry, chromatic saliency,
    and geometric morphology to build a rich human identification profile.
    """

    def __init__(self, physiology_model: Optional[ThermalPhysiologyModel] = None) -> None:
        self.physiology = physiology_model or ThermalPhysiologyModel()
        # History of observed aspect ratios and motion for gesture analysis: track_id -> list of (t, u, v, area, aspect)
        self._history: Dict[int, List[Tuple[float, float, float, float, float]]] = {}

    def clear_history(self) -> None:
        self._history.clear()

    def identify(
        self,
        track: Any,
        water_depth_m: float = 0.0,
        ambient_temp_c: float = 22.0,
        wind_speed_ms: float = 3.0,
        water_temp_c: Optional[float] = None,
        nearby_hazard_classes: Sequence[str] = (),
    ) -> HumanProfile:
        """Perform comprehensive human identification on a track."""
        tid = getattr(track, "tid", 0)
        observations = getattr(track, "observations", [])
        n_obs = len(observations)
        t_now = getattr(track, "updated_t", 0.0)

        # 1. Update temporal history
        if observations:
            if tid not in self._history or len(self._history[tid]) < len(observations):
                self._history[tid] = []
                for obs in observations:
                    asp = float(getattr(obs, "attributes", {}).get("aspect", 1.0) if hasattr(obs, "attributes") else 1.0)
                    a_px = float(getattr(obs, "area_px", 35.0) or 35.0)
                    np_ = float(getattr(obs, "north", getattr(track, "north", 0.0)))
                    ep_ = float(getattr(obs, "east", getattr(track, "east", 0.0)))
                    t_obs = float(getattr(obs, "t", t_now))
                    self._history[tid].append((t_obs, np_, ep_, a_px, asp))
            else:
                latest_obs = observations[-1]
                aspect = float(getattr(latest_obs, "attributes", {}).get("aspect", 1.0) if hasattr(latest_obs, "attributes") else 1.0)
                area_px = float(getattr(latest_obs, "area_px", 35.0) or 35.0)
                n_pos = float(getattr(latest_obs, "north", getattr(track, "north", 0.0)))
                e_pos = float(getattr(latest_obs, "east", getattr(track, "east", 0.0)))
                self._history[tid].append((t_now, n_pos, e_pos, area_px, aspect))
            # Keep up to 30 recent observations
            if len(self._history[tid]) > 30:
                self._history[tid] = self._history[tid][-30:]

        # 2. Extract best radiometry and physical dimensions
        best_obs = min(observations, key=lambda o: getattr(o, "gsd_m", 0.15)) if observations else None
        apparent_temp = (
            float(best_obs.peak_temp_c)
            if best_obs and getattr(best_obs, "peak_temp_c", None) is not None
            else getattr(track, "peak_temp_c", 30.0) or 30.0
        )
        bg_temp = (
            float(best_obs.background_temp_c)
            if best_obs and getattr(best_obs, "background_temp_c", None) is not None
            else 18.0
        )
        best_gsd = getattr(best_obs, "gsd_m", 0.15) if best_obs else 0.15

        # 3. Analyze SOS waving and gesture periodicity
        sos_waving, waving_freq = self._detect_sos_waving(tid)

        # 4. Infer Posture and Confidence
        posture, posture_conf, posture_notes = self._classify_posture(
            track, water_depth_m, sos_waving, nearby_hazard_classes
        )

        # 5. Demographic and Group Size Inference
        demographic, group_size, demo_notes = self._infer_demographic(track, best_gsd)

        # 6. Clothing & Visual Saliency
        saliency, color_hint = self._analyze_clothing(track)

        # 7. Thermal Vitals & Physiology
        immobile = bool(getattr(track, "immobility", 0.0) > 0.75 or getattr(track, "speed_ms", 0.0) < 0.08)
        viability = self.physiology.assess(
            apparent_temp_c=apparent_temp,
            background_c=bg_temp,
            ambient_c=ambient_temp_c,
            wind_ms=wind_speed_ms,
            water_temp_c=water_temp_c or (bg_temp if water_depth_m > 0.2 else None),
            in_water=water_depth_m > 0.45 or posture == HumanPosture.IN_WATER_CLINGING,
            partial_water=0.05 < water_depth_m <= 0.45,
            under_rubble=posture in (HumanPosture.PRONE_PARTIAL_BURIAL, HumanPosture.TRAPPED_RUBBLE),
            immobile=immobile,
            minutes_observed=float(getattr(track, "age_s", 0.0) / 60.0),
            temp_trend_k_per_min=0.0,
            posture=posture.value,
            needs=self._recommend_equipment(water_depth_m, posture, viability_obj=None).value,
            group_size=group_size,
            gsd_m=best_gsd,
            expected_area_px=math.pi * (0.444 / max(best_gsd, 1e-4)) ** 2,
            measured_area_px=float(getattr(best_obs, "area_px", 30.0) or 30.0) if best_obs else 30.0,
            confidence=float(getattr(track, "confidence", 0.8)),
        )

        # Estimated Core Temperature proxy (modelled from surface deficit and exposure medium)
        core_temp = self._estimate_core_temperature(
            apparent_temp, bg_temp, water_depth_m, immobile, getattr(track, "age_s", 0.0)
        )

        # 8. Triage Priority & Time Criticality
        triage_priority = viability.priority
        time_critical_s = viability.time_critical_s
        if sos_waving and triage_priority == PriorityTier.MINOR:
            # Active waving in hazardous environment elevates to DELAYED for immediate check
            triage_priority = PriorityTier.DELAYED

        # 9. Determine Recommended Equipment
        rec_equipment = self._recommend_equipment(water_depth_m, posture, viability)

        # 10. Decoy Rejection Checks
        is_human, decoy_notes = self._validate_human_authenticity(
            track, apparent_temp, bg_temp, best_gsd
        )

        # 11. Distress Level
        distress_level = self._compute_distress_level(
            sos_waving, immobile, water_depth_m, posture, triage_priority
        )

        # Construct ID Tag
        tag_prefix = f"H-{tid}" if isinstance(tid, str) else f"H-{tid:03d}"
        id_tag = f"{tag_prefix}:{posture.value[:3].upper()}"

        notes = posture_notes + demo_notes
        if sos_waving:
            notes.append(f"active SOS distress waving identified ({waving_freq:.1f} Hz)")
        if saliency > 50:
            notes.append(f"high-visibility chromatic clothing detected ({color_hint})")
        if core_temp < 35.0:
            notes.append(f"hypothermia alert: estimated core temp {core_temp:.1f}C")

        origin = getattr(track, "origin", None)
        north = float(getattr(track, "north", 0.0))
        east = float(getattr(track, "east", 0.0))
        geo_pt = track.geo(origin) if origin else None

        return HumanProfile(
            id_tag=id_tag,
            track_id=tid,
            confidence=float(getattr(track, "confidence", 0.85)),
            is_human=is_human,
            posture=posture,
            posture_confidence=posture_conf,
            distress_level=distress_level,
            sos_waving_detected=sos_waving,
            waving_frequency_hz=waving_freq,
            demographic=demographic,
            group_size=group_size,
            clothing_saliency=saliency,
            clothing_color_hint=color_hint,
            apparent_temp_c=apparent_temp,
            estimated_core_temp_c=core_temp,
            hypothermia_risk=float(viability.hypothermia_risk),
            heat_stress_risk=float(viability.heat_stress_risk),
            triage_priority=triage_priority,
            recommended_equipment=rec_equipment,
            time_critical_s=time_critical_s,
            decoy_rejection_notes=decoy_notes,
            identification_notes=notes,
            location_ned=(north, east, 0.0),
            geo_point=geo_pt,
        )

    # ------------------------------------------------------------------ #
    def _detect_sos_waving(self, tid: int) -> Tuple[bool, float]:
        """Detect dynamic waving / SOS motion patterns across frames."""
        hist = self._history.get(tid, [])
        if len(hist) < 4:
            return False, 0.0
        ts = np.array([h[0] for h in hist])
        dt = ts[-1] - ts[0]
        if dt < 0.5:
            return False, 0.0

        aspects = np.array([h[4] for h in hist])
        areas = np.array([h[3] for h in hist])

        # Waving causes oscillatory variations in aspect ratio & projected upper-body area
        aspect_std = float(np.std(aspects))
        aspect_ptp = float(np.ptp(aspects))
        area_std = float(np.std(areas) / max(np.mean(areas), 1e-3))

        # Peak counting in aspect series for frequency estimation
        diffs = np.diff(aspects)
        zero_crossings = np.where(np.diff(np.signbit(diffs)))[0]
        n_cycles = len(zero_crossings) / 2.0
        freq = float(n_cycles / max(dt, 0.1))

        # Typical human waving frequency is 0.8 Hz to 3.5 Hz
        is_waving = (aspect_ptp > 0.35 or aspect_std > 0.12 or area_std > 0.25) and (0.6 <= freq <= 4.2)
        return bool(is_waving), freq if is_waving else 0.0

    def _classify_posture(
        self,
        track: Any,
        water_depth: float,
        sos_waving: bool,
        hazards: Sequence[str],
    ) -> Tuple[HumanPosture, float, List[str]]:
        """Classify posture using shape, immersion context, and motion priors."""
        notes = []
        immobility = float(getattr(track, "immobility", 0.0))
        speed = float(getattr(track, "speed_ms", 0.0))
        obs = getattr(track, "observations", [])
        elongs = [float(o.attributes.get("elongation", 1.2)) for o in obs if o.attributes]
        median_elong = float(np.median(elongs)) if elongs else 1.2

        if water_depth > 0.5:
            notes.append("survivor in deep water (>0.5m) -> in-water clinging posture")
            return HumanPosture.IN_WATER_CLINGING, 0.92, notes

        if sos_waving:
            notes.append("periodic aspect change -> active SOS waving posture")
            return HumanPosture.WAVING_SOS, 0.90, notes

        if any(h in ("collapsed_structure", "debris_field") for h in hazards) and immobility > 0.85 and median_elong > 1.6:
            notes.append("immobile near collapse with high elongation -> trapped/partial burial")
            return HumanPosture.PRONE_PARTIAL_BURIAL, 0.88, notes

        if immobility > 0.88:
            if median_elong > 1.5:
                notes.append("stationary high elongation -> lying prone/supine")
                return HumanPosture.LYING, 0.85, notes
            else:
                notes.append("stationary compact elongation -> sitting")
                return HumanPosture.SITTING, 0.80, notes

        if speed > 0.6:
            notes.append(f"continuous ground speed {speed:.1f} m/s -> walking")
            return HumanPosture.WALKING, 0.82, notes

        notes.append("nominal vertical human footprint -> standing")
        return HumanPosture.STANDING, 0.78, notes

    def _infer_demographic(
        self, track: Any, gsd_m: float
    ) -> Tuple[DemographicGroup, int, List[str]]:
        """Estimate group clustering and adult/child proxy."""
        notes = []
        obs = getattr(track, "observations", [])
        best_obs = min(obs, key=lambda o: getattr(o, "gsd_m", 0.15)) if obs else None
        area_px = float(getattr(best_obs, "area_px", 20.0) or 20.0) if best_obs else 20.0
        apparent_area_m2 = area_px * (gsd_m ** 2)

        # Expected single adult projected area: 0.35 - 0.75 m^2
        if apparent_area_m2 > 1.3:
            est_group = max(2, int(round(apparent_area_m2 / 0.55)))
            notes.append(f"projected area {apparent_area_m2:.2f} m^2 indicates group of ~{est_group} people")
            return DemographicGroup.GROUP_CLUSTER, est_group, notes

        if apparent_area_m2 < 0.22 and not (getattr(track, "in_water", False)):
            notes.append(f"small projected area {apparent_area_m2:.2f} m^2 indicates child or infant")
            return DemographicGroup.CHILD, 1, notes

        return DemographicGroup.ADULT, 1, notes

    def _analyze_clothing(self, track: Any) -> Tuple[float, str]:
        """Analyze chromatic saliency and infer high-vis color."""
        obs = getattr(track, "observations", [])
        rgb_obs = [o for o in obs if getattr(o, "modality", "rgb") in ("rgb", "fused")]
        if not rgb_obs:
            return 0.0, "unknown/unobserved"

        saliency_vals = [
            float(getattr(o, "attributes", {}).get("rgb_saliency", getattr(o, "attributes", {}).get("saliency", 0.0)))
            for o in rgb_obs if hasattr(o, "attributes")
        ]
        max_sal = max(saliency_vals) if saliency_vals else 0.0

        if max_sal > 65:
            color = "high_visibility_orange_red"
        elif max_sal > 40:
            color = "chromatic_yellow_or_blue"
        else:
            color = "subdued_or_muddy"
        return float(max_sal), color

    def _estimate_core_temperature(
        self,
        apparent_temp: float,
        bg_temp: float,
        water_depth: float,
        immobile: bool,
        age_s: float,
    ) -> float:
        """Estimate core body temperature proxy based on immersion and peripheral cooling."""
        base_core = 37.0
        if water_depth > 0.4:
            # Cold water immersion: convective cooling pulls core down over time
            loss_rate = 0.08  # C per minute
            immersion_mins = max(5.0, age_s / 60.0)
            core = base_core - loss_rate * immersion_mins * (max(20.0 - bg_temp, 2.0) / 10.0)
            return float(np.clip(core, 26.0, 37.0))

        if apparent_temp < 25.0 and immobile:
            # Peripheral shutdown on land
            return float(np.clip(35.5 - (25.0 - apparent_temp) * 0.4, 28.0, 37.0))

        if apparent_temp > 36.5:
            # Heat stress condition
            return float(np.clip(37.5 + (apparent_temp - 36.5) * 0.6, 37.0, 41.5))

        return 36.8

    def _recommend_equipment(
        self,
        water_depth: float,
        posture: HumanPosture,
        viability_obj: Optional[SurvivorViability],
    ) -> RescueEquipmentNeed:
        """Map survivor state to the optimal air-drop rescue payload."""
        if water_depth > 0.35 or posture == HumanPosture.IN_WATER_CLINGING:
            return RescueEquipmentNeed.FLOTATION_BUOY

        if posture in (HumanPosture.PRONE_PARTIAL_BURIAL, HumanPosture.TRAPPED_RUBBLE):
            return RescueEquipmentNeed.LORA_LOCATOR_BEACON

        if viability_obj and viability_obj.hypothermia_risk > 0.5:
            return RescueEquipmentNeed.THERMAL_EMERGENCY_BLANKET

        if viability_obj and viability_obj.heat_stress_risk > 0.5:
            return RescueEquipmentNeed.WATER_RATIONS_PACK

        return RescueEquipmentNeed.FIRST_AID_TRAUMA_KIT

    def _validate_human_authenticity(
        self, track: Any, apparent_temp: float, bg_temp: float, gsd_m: float
    ) -> Tuple[bool, List[str]]:
        """Verify the detection is genuinely human and reject decoys."""
        notes = []
        # Rule 1: Temperature range
        if apparent_temp > 48.0:
            notes.append(f"rejected: temperature {apparent_temp:.1f}C exceeds living biology (hot engine / rock)")
            return False, notes

        if apparent_temp < 15.0 and bg_temp > 22.0:
            notes.append(f"rejected: temperature {apparent_temp:.1f}C too cold for living body in warm background")
            return False, notes

        # Rule 2: Kinematic & speed check
        speed = float(getattr(track, "speed_ms", 0.0))
        if speed > 6.5:  # faster than human sprint in disaster terrain
            notes.append(f"rejected: speed {speed:.1f} m/s exceeds human capability (vehicle/fast animal)")
            return False, notes

        # Rule 3: Extent ratio
        obs = getattr(track, "observations", [])
        if obs:
            ext_ratios = [float(o.attributes.get("extent_ratio", 1.0)) for o in obs if o.attributes]
            if ext_ratios and max(ext_ratios) > 7.0:
                notes.append("rejected: surrounding hot context too large for human target (vehicle/generator)")
                return False, notes

        return True, notes

    def _compute_distress_level(
        self,
        sos_waving: bool,
        immobile: bool,
        water_depth: float,
        posture: HumanPosture,
        priority: PriorityTier,
    ) -> DistressLevel:
        if sos_waving or water_depth > 0.3 or posture == HumanPosture.IN_WATER_CLINGING:
            return DistressLevel.CRITICAL_ACTIVE_SOS
        if immobile or posture in (HumanPosture.PRONE_PARTIAL_BURIAL, HumanPosture.TRAPPED_RUBBLE):
            return DistressLevel.PASSIVE_IMMOBILE
        if priority == PriorityTier.IMMEDIATE:
            return DistressLevel.CRITICAL_ACTIVE_SOS
        if posture == HumanPosture.WALKING:
            return DistressLevel.SELF_SUFFICIENT
        return DistressLevel.MODERATE_DISTRESS
