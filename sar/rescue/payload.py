"""Ballistic & Aerodynamic Rescue Payload Drop Simulation.

Physics-grounded simulation of precision payload drops from UAVs for search
and rescue operations.

Key physics modeled:
--------------------
- Gravitational acceleration ($g = 9.80665 m/s^2$).
- Atmospheric density lapse $\rho(z)$ with altitude and temperature.
- Quadratic aerodynamic drag: $\vec{F}_d = -\frac{1}{2} \rho C_d A |\vec{v}_{rel}| \vec{v}_{rel}$.
- Wind vector field $\vec{w}(z)$ with altitude shear and gusts.
- Two-stage descent: Freefall phase followed by parachute deployment canopy deceleration.
- Inverse Ballistic Targeting: Computes the precise release lead distance $\Delta \vec{r}_{lead}$
  and release trigger timing so the payload lands on the target with minimal error.
- Trajectory recording for visualization and animation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, local_to_wgs84
from sar.perception.human_id import RescueEquipmentNeed


class RescuePayloadType(str, Enum):
    """Types of deployable SAR emergency payloads."""

    FLOTATION_BUOY = "flotation_buoy"            # Self-inflating lifebuoy / raft
    FIRST_AID_TRAUMA_KIT = "first_aid_kit"       # Sealed medical & trauma kit
    THERMAL_EMERGENCY_BLANKET = "thermal_blanket" # Radiant foil pack & warming pack
    LORA_LOCATOR_BEACON = "lora_locator_beacon"  # 900MHz self-locating acoustic/radio beacon
    WATER_RATIONS_PACK = "water_rations"         # Emergency hydration & electrolytes
    EXTRICATION_MARKER = "extrication_marker"    # Smoke / strobe marker for search teams


class PayloadStatus(str, Enum):
    """Lifecycle status of a rescue payload."""

    STOWED = "stowed"
    RELEASED = "released"
    DESCENDING = "descending"
    PARACHUTE_DEPLOYED = "parachute_deployed"
    LANDED = "landed"
    BEACON_ACTIVE = "beacon_active"
    RETRIEVED = "retrieved"


@dataclass
class PayloadSpec:
    """Aerodynamic and physical properties of a payload package."""

    payload_type: RescuePayloadType
    mass_kg: float
    freefall_cd_area: float       # Cd * A before parachute deployment (m^2)
    canopy_cd_area: float         # Cd * A with parachute open (m^2)
    parachute_delay_s: float      # Seconds after release before canopy opens
    beacon_equipped: bool = True
    color_rgb: Tuple[int, int, int] = (255, 100, 20)  # High-visibility orange
    label: str = "Rescue Payload"


#: Payload specifications calibrated against real SAR equipment specs
PAYLOAD_SPECS: Dict[RescuePayloadType, PayloadSpec] = {
    RescuePayloadType.FLOTATION_BUOY: PayloadSpec(
        payload_type=RescuePayloadType.FLOTATION_BUOY,
        mass_kg=0.85,
        freefall_cd_area=0.035,
        canopy_cd_area=0.85,
        parachute_delay_s=0.4,
        beacon_equipped=True,
        color_rgb=(255, 120, 0),
        label="Auto-Inflating Lifebuoy",
    ),
    RescuePayloadType.FIRST_AID_TRAUMA_KIT: PayloadSpec(
        payload_type=RescuePayloadType.FIRST_AID_TRAUMA_KIT,
        mass_kg=0.65,
        freefall_cd_area=0.025,
        canopy_cd_area=0.75,
        parachute_delay_s=0.35,
        beacon_equipped=True,
        color_rgb=(230, 40, 40),
        label="Emergency Trauma Kit",
    ),
    RescuePayloadType.THERMAL_EMERGENCY_BLANKET: PayloadSpec(
        payload_type=RescuePayloadType.THERMAL_EMERGENCY_BLANKET,
        mass_kg=0.35,
        freefall_cd_area=0.015,
        canopy_cd_area=0.60,
        parachute_delay_s=0.30,
        beacon_equipped=True,
        color_rgb=(240, 220, 50),
        label="Thermal Exposure Blanket",
    ),
    RescuePayloadType.LORA_LOCATOR_BEACON: PayloadSpec(
        payload_type=RescuePayloadType.LORA_LOCATOR_BEACON,
        mass_kg=0.25,
        freefall_cd_area=0.012,
        canopy_cd_area=0.50,
        parachute_delay_s=0.25,
        beacon_equipped=True,
        color_rgb=(50, 150, 255),
        label="LoRa SAR Locator Tag",
    ),
    RescuePayloadType.WATER_RATIONS_PACK: PayloadSpec(
        payload_type=RescuePayloadType.WATER_RATIONS_PACK,
        mass_kg=0.90,
        freefall_cd_area=0.030,
        canopy_cd_area=0.80,
        parachute_delay_s=0.40,
        beacon_equipped=False,
        color_rgb=(80, 200, 120),
        label="Hydration & Electrolytes Pack",
    ),
    RescuePayloadType.EXTRICATION_MARKER: PayloadSpec(
        payload_type=RescuePayloadType.EXTRICATION_MARKER,
        mass_kg=0.40,
        freefall_cd_area=0.020,
        canopy_cd_area=0.65,
        parachute_delay_s=0.30,
        beacon_equipped=True,
        color_rgb=(220, 60, 220),
        label="High-Intensity Extrication Marker",
    ),
}


@dataclass
class DropTrajectoryPoint:
    """One discrete state point along the payload descent trajectory."""

    t: float
    pos_ned: np.ndarray          # North, East, Down (metres)
    vel_ned: np.ndarray          # m/s
    altitude_agl: float          # Height above local ground (metres)
    parachute_open: bool = False
    speed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "north_m": round(float(self.pos_ned[0]), 2),
            "east_m": round(float(self.pos_ned[1]), 2),
            "altitude_agl_m": round(float(self.altitude_agl), 2),
            "speed_ms": round(float(self.speed_ms), 2),
            "parachute_open": self.parachute_open,
        }


@dataclass
class DropResult:
    """Result of an executed or simulated payload drop."""

    drop_id: str
    payload_type: RescuePayloadType
    target_id: int
    target_pos_ned: np.ndarray
    release_pos_ned: np.ndarray
    release_vel_ned: np.ndarray
    release_time: float
    impact_pos_ned: np.ndarray
    impact_time: float
    impact_velocity_ms: float
    miss_distance_m: float
    success: bool
    trajectory: List[DropTrajectoryPoint] = field(default_factory=list)
    wind_vector_ms: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drop_id": self.drop_id,
            "payload_type": self.payload_type.value,
            "target_id": self.target_id,
            "target_north_m": round(float(self.target_pos_ned[0]), 2),
            "target_east_m": round(float(self.target_pos_ned[1]), 2),
            "release_time": round(float(self.release_time), 2),
            "impact_time": round(float(self.impact_time), 2),
            "release_pos_ned": [round(float(x), 2) for x in self.release_pos_ned],
            "impact_pos_ned": [round(float(x), 2) for x in self.impact_pos_ned],
            "release_north_m": round(float(self.release_pos_ned[0]), 2),
            "release_east_m": round(float(self.release_pos_ned[1]), 2),
            "release_altitude_m": round(float(-self.release_pos_ned[2]), 2),
            "impact_north_m": round(float(self.impact_pos_ned[0]), 2),
            "impact_east_m": round(float(self.impact_pos_ned[1]), 2),
            "impact_velocity_ms": round(float(self.impact_velocity_ms), 2),
            "miss_distance_m": round(float(self.miss_distance_m), 2),
            "flight_time_s": round(float(self.impact_time - self.release_time), 2),
            "fall_time_s": round(float(self.impact_time - self.release_time), 2),
            "success": self.success,
            "trajectory_points": len(self.trajectory),
            "trajectory": [p.to_dict() for p in self.trajectory[::max(1, len(self.trajectory) // 40)]],
        }


class BallisticDropCalculator:
    """Aerodynamic ballistic trajectory engine for precision payload drops."""

    GRAVITY = 9.80665

    def __init__(self, air_density_kg_m3: float = 1.225) -> None:
        self.rho0 = float(air_density_kg_m3)

    # ------------------------------------------------------------------ #
    def simulate_drop(
        self,
        spec: PayloadSpec,
        release_pos_ned: np.ndarray,
        release_vel_ned: np.ndarray,
        release_time: float = 0.0,
        wind_fn: Optional[Callable[[float, float, float, float], np.ndarray]] = None,
        terrain_z_fn: Optional[Callable[[float, float], float]] = None,
        dt: float = 0.02,
        max_time_s: float = 30.0,
    ) -> DropResult:
        """Numerically integrate payload descent from release to ground impact."""
        pos = np.asarray(release_pos_ned, dtype=np.float64).copy()
        vel = np.asarray(release_vel_ned, dtype=np.float64).copy()
        m = max(spec.mass_kg, 0.05)

        t = release_time
        trajectory: List[DropTrajectoryPoint] = []
        chute_open = False

        ground_z = 0.0
        if terrain_z_fn:
            try:
                ground_z = float(terrain_z_fn(pos[0], pos[1]))
            except Exception:
                ground_z = 0.0

        # Initial point
        agl = max(0.0, -pos[2] - ground_z)
        trajectory.append(DropTrajectoryPoint(
            t=t, pos_ned=pos.copy(), vel_ned=vel.copy(), altitude_agl=agl,
            parachute_open=False, speed_ms=float(np.linalg.norm(vel))
        ))

        t_end = t + max_time_s
        while t < t_end:
            # 1. Update ground elevation at current horizontal coordinate
            if terrain_z_fn:
                try:
                    ground_z = float(terrain_z_fn(pos[0], pos[1]))
                except Exception:
                    ground_z = 0.0

            # 2. Check touchdown condition (-pos[2] is altitude MSL above home)
            current_alt_msl = -pos[2]
            if current_alt_msl <= ground_z:
                pos[2] = -ground_z
                break

            # 3. Check parachute deployment
            time_since_release = t - release_time
            chute_open = time_since_release >= spec.parachute_delay_s
            cd_a = spec.canopy_cd_area if chute_open else spec.freefall_cd_area

            # 4. Air density at altitude
            rho = self.rho0 * math.exp(-max(0.0, current_alt_msl) / 8500.0)

            # 5. Wind at current position
            if wind_fn:
                try:
                    wind = np.asarray(wind_fn(pos[0], pos[1], current_alt_msl, t), dtype=np.float64)
                except Exception:
                    wind = np.zeros(3)
            else:
                wind = np.zeros(3)

            # 6. Relative airspeed vector
            v_rel = vel - wind
            speed_rel = float(np.linalg.norm(v_rel))

            # 7. Forces: Gravity + Drag
            f_drag = -0.5 * rho * cd_a * speed_rel * v_rel
            f_grav = np.array([0.0, 0.0, m * self.GRAVITY])
            acc = (f_drag + f_grav) / m

            # 8. Numerical Integration (Velocity Verlet / Symplectic Euler)
            vel += acc * dt
            pos += vel * dt
            t += dt

            agl = max(0.0, -pos[2] - ground_z)
            trajectory.append(DropTrajectoryPoint(
                t=t, pos_ned=pos.copy(), vel_ned=vel.copy(), altitude_agl=agl,
                parachute_open=chute_open, speed_ms=float(np.linalg.norm(vel))
            ))

        impact_pos = pos.copy()
        impact_vel = float(np.linalg.norm(vel))

        return DropResult(
            drop_id=f"drop_{int(t*1000)}",
            payload_type=spec.payload_type,
            target_id=0,
            target_pos_ned=np.zeros(3),
            release_pos_ned=release_pos_ned.copy(),
            release_vel_ned=release_vel_ned.copy(),
            release_time=release_time,
            impact_pos_ned=impact_pos,
            impact_time=t,
            impact_velocity_ms=impact_vel,
            miss_distance_m=0.0,
            success=True,
            trajectory=trajectory,
            wind_vector_ms=tuple(wind[:3]) if 'wind' in locals() else (0.0, 0.0, 0.0),
        )

    # ------------------------------------------------------------------ #
    def compute_release_solution(
        self,
        spec: PayloadSpec,
        target_pos_ned: np.ndarray,
        aircraft_alt_agl_m: float,
        aircraft_ground_speed_ms: float,
        approach_heading_deg: float,
        wind_fn: Optional[Callable[[float, float, float, float], np.ndarray]] = None,
        terrain_z_fn: Optional[Callable[[float, float], float]] = None,
    ) -> Dict[str, Any]:
        """Compute the optimal release lead vector and release coordinates.

        Returns:
            release_lead_m: (lead_north, lead_east) distance before target to trigger drop.
            release_pos_ned: exact 3D coordinates where drone must drop.
            estimated_fall_time_s: expected descent duration.
            impact_speed_ms: terminal landing speed.
        """
        hdg_rad = math.radians(approach_heading_deg)
        vx = aircraft_ground_speed_ms * math.cos(hdg_rad)
        vy = aircraft_ground_speed_ms * math.sin(hdg_rad)
        vz = 0.0

        target_n, target_e = target_pos_ned[0], target_pos_ned[1]
        target_ground_z = float(terrain_z_fn(target_n, target_e)) if terrain_z_fn else 0.0

        # Aircraft release altitude NED (Z is negative up)
        release_z = -(target_ground_z + aircraft_alt_agl_m)
        v_release = np.array([vx, vy, vz], dtype=np.float64)

        # First simulate a nominal drop at (0, 0) to find the ballistic offset
        sim_res = self.simulate_drop(
            spec=spec,
            release_pos_ned=np.array([0.0, 0.0, release_z]),
            release_vel_ned=v_release,
            wind_fn=wind_fn,
            terrain_z_fn=terrain_z_fn,
        )

        drift_n = sim_res.impact_pos_ned[0]
        drift_e = sim_res.impact_pos_ned[1]
        fall_time = sim_res.impact_time - sim_res.release_time

        # To land on target, release point must be offset by -drift
        release_n = target_n - drift_n
        release_e = target_e - drift_e

        # Lead distance along the approach vector
        lead_vector = np.array([drift_n, drift_e])
        lead_dist = float(np.linalg.norm(lead_vector))

        return {
            "lead_distance_m": round(lead_dist, 2),
            "lead_vector_ned": (round(float(drift_n), 2), round(float(drift_e), 2)),
            "release_pos_ned": np.array([release_n, release_e, release_z]),
            "target_pos_ned": np.asarray(target_pos_ned),
            "estimated_fall_time_s": round(float(fall_time), 2),
            "impact_speed_ms": round(float(sim_res.impact_velocity_ms), 2),
            "chute_deploy_alt_agl_m": round(aircraft_alt_agl_m - 0.5 * self.GRAVITY * (spec.parachute_delay_s ** 2), 1),
        }
