"""Autonomous Rescue Mission Coordinator and Task Manager.

Coordinates the end-to-end autonomous rescue pipeline:
1. Ingests HumanIdentification profiles from perception.
2. Prioritizes survivors using START/SALT triage criteria.
3. Automatically computes ballistic drop solutions for required emergency payloads.
4. Triggers MAVLink actuation / simulated payload drops (`DO_SET_SERVO`).
5. Generates safe ground & water rescue response routes for emergency teams.
6. Tracks active payload descents, landing beacons, and rescue completion status.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint
from sar.perception.human_id import (
    HumanIdentifier,
    HumanPosture,
    HumanProfile,
    RescueEquipmentNeed,
)
from sar.perception.thermal import PriorityTier
from sar.rescue.payload import (
    BallisticDropCalculator,
    DropResult,
    PAYLOAD_SPECS,
    PayloadSpec,
    PayloadStatus,
    RescuePayloadType,
)
from sar.rescue.routing import GroundRescueRouter, RescueRoute, RescueTeamType

log = logging.getLogger("sar.rescue")

__all__ = [
    "RescueTaskState",
    "RescueOperation",
    "RescueCoordinator",
]


class RescueTaskState(str, Enum):
    """Lifecycle stages of a single survivor rescue operation."""

    DETECTED = "detected"
    HUMAN_IDENTIFIED = "human_identified"
    TRIAGED = "triaged"
    DROP_PLANNED = "drop_planned"
    DROP_DISPATCHED = "drop_dispatched"
    PAYLOAD_IN_FLIGHT = "payload_in_flight"
    PAYLOAD_DELIVERED = "payload_delivered"
    BEACON_TRANSMITTING = "beacon_transmitting"
    GROUND_TEAM_DISPATCHED = "ground_team_dispatched"
    RESCUE_COMPLETED = "rescue_completed"


@dataclass
class RescueOperation:
    """Complete record of an active or completed rescue operation."""

    op_id: str
    target_id: int
    profile: HumanProfile
    state: RescueTaskState = RescueTaskState.DETECTED
    assigned_payload: RescuePayloadType = RescuePayloadType.FIRST_AID_TRAUMA_KIT
    drop_result: Optional[DropResult] = None
    ground_routes: Dict[str, RescueRoute] = field(default_factory=dict)
    active_team: Optional[RescueTeamType] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    payload_dropped_at: Optional[float] = None
    rescue_completed_at: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "target_id": self.target_id,
            "state": self.state.value,
            "assigned_payload": self.assigned_payload.value,
            "profile": self.profile.to_dict(),
            "drop_result": self.drop_result.to_dict() if self.drop_result else None,
            "ground_routes": {k: v.to_dict() for k, v in self.ground_routes.items()},
            "active_team": self.active_team.value if self.active_team else None,
            "created_at": round(self.created_at, 1),
            "updated_at": round(self.updated_at, 1),
            "payload_dropped_at": (
                round(self.payload_dropped_at, 1) if self.payload_dropped_at else None
            ),
            "rescue_completed_at": (
                round(self.rescue_completed_at, 1) if self.rescue_completed_at else None
            ),
            "notes": list(self.notes[-6:]),
        }


class RescueCoordinator:
    """Autonomous Rescue Operations Manager."""

    def __init__(
        self,
        origin: GeoPoint,
        world: Optional[Any] = None,
        hazard_map: Optional[Any] = None,
        mav_conn: Optional[Any] = None,
        auto_drop_enabled: bool = True,
    ) -> None:
        self.origin = origin
        self.world = world
        self.hazard_map = hazard_map
        self.conn = mav_conn
        self.auto_drop_enabled = auto_drop_enabled

        self.human_id_engine = HumanIdentifier()
        self.drop_calc = BallisticDropCalculator()
        self.router = GroundRescueRouter()

        self.operations: Dict[int, RescueOperation] = {}
        self.completed_drops: List[DropResult] = []
        self._op_counter = 0

    # ------------------------------------------------------------------ #
    def process_track(
        self,
        track: Any,
        t_mission: float,
        water_depth_m: float = 0.0,
        ambient_temp_c: float = 22.0,
        wind_speed_ms: float = 3.0,
        water_temp_c: Optional[float] = None,
    ) -> RescueOperation:
        """Process a confirmed track through human identification and task planning."""
        tid = getattr(track, "tid", 0)

        # 1. Run Human Identification
        profile = self.human_id_engine.identify(
            track=track,
            water_depth_m=water_depth_m,
            ambient_temp_c=ambient_temp_c,
            wind_speed_ms=wind_speed_ms,
            water_temp_c=water_temp_c,
        )

        # 2. Get or create operation
        if tid not in self.operations:
            self._op_counter += 1
            op = RescueOperation(
                op_id=f"SAR-OP-{self._op_counter:03d}",
                target_id=tid,
                profile=profile,
                state=RescueTaskState.HUMAN_IDENTIFIED,
                created_at=t_mission,
                updated_at=t_mission,
            )
            # Map equipment recommendation
            op.assigned_payload = self._map_equipment_to_payload(profile.recommended_equipment)
            self.operations[tid] = op
            op.notes.append(f"Operation initialized for {profile.id_tag} (Triage: {profile.triage_priority.value})")
            log.info("RESCUE OP INITIALIZED: %s for %s (%s)", op.op_id, profile.id_tag, op.assigned_payload.value)
        else:
            op = self.operations[tid]
            op.profile = profile
            op.updated_at = t_mission
            if op.state == RescueTaskState.DETECTED:
                op.state = RescueTaskState.HUMAN_IDENTIFIED

        # 3. Compute Ground Routes if not yet planned
        if not op.ground_routes:
            self._plan_ground_routes(op)

        return op

    # ------------------------------------------------------------------ #
    def _map_equipment_to_payload(self, need: RescueEquipmentNeed) -> RescuePayloadType:
        mapping = {
            RescueEquipmentNeed.FLOTATION_BUOY: RescuePayloadType.FLOTATION_BUOY,
            RescueEquipmentNeed.FIRST_AID_TRAUMA_KIT: RescuePayloadType.FIRST_AID_TRAUMA_KIT,
            RescueEquipmentNeed.THERMAL_EMERGENCY_BLANKET: RescuePayloadType.THERMAL_EMERGENCY_BLANKET,
            RescueEquipmentNeed.LORA_LOCATOR_BEACON: RescuePayloadType.LORA_LOCATOR_BEACON,
            RescueEquipmentNeed.WATER_RATIONS_PACK: RescuePayloadType.WATER_RATIONS_PACK,
            RescueEquipmentNeed.EXTRICATION_ASSISTANCE: RescuePayloadType.EXTRICATION_MARKER,
        }
        return mapping.get(need, RescuePayloadType.FIRST_AID_TRAUMA_KIT)

    def _plan_ground_routes(self, op: RescueOperation) -> None:
        """Plan routes for all applicable response team modalities."""
        target_ned = op.profile.location_ned
        start_ned = (0.0, 0.0, 0.0)  # GCS / Base Camp

        # 1. Foot rescue team route
        foot_route = self.router.plan_route(
            start_ned=start_ned,
            target_ned=target_ned,
            team_type=RescueTeamType.FOOT_RESCUE_TEAM,
            world=self.world,
            hazard_map=self.hazard_map,
            target_id=op.target_id,
        )
        op.ground_routes["foot_team"] = foot_route

        # 2. Rescue boat route (if water present)
        if self.world and (getattr(self.world, "water_level", 0.0) > 0.1 or op.profile.posture == HumanPosture.IN_WATER_CLINGING):
            boat_route = self.router.plan_route(
                start_ned=start_ned,
                target_ned=target_ned,
                team_type=RescueTeamType.AMPHIBIOUS_RESCUE_BOAT,
                world=self.world,
                hazard_map=self.hazard_map,
                target_id=op.target_id,
            )
            op.ground_routes["rescue_boat"] = boat_route
            op.active_team = RescueTeamType.AMPHIBIOUS_RESCUE_BOAT
        else:
            op.active_team = RescueTeamType.FOOT_RESCUE_TEAM

    # ------------------------------------------------------------------ #
    def execute_payload_drop(
        self,
        target_id: int,
        aircraft_pos_ned: np.ndarray,
        aircraft_vel_ned: np.ndarray,
        t_mission: float,
        payload_type: Optional[RescuePayloadType] = None,
        servo_channel: int = 9,
    ) -> Optional[DropResult]:
        """Execute a precision payload drop over the designated survivor."""
        if target_id not in self.operations:
            log.warning("Cannot drop payload: Target %d not registered in operations", target_id)
            return None

        op = self.operations[target_id]
        p_type = payload_type or op.assigned_payload
        spec = PAYLOAD_SPECS.get(p_type, PAYLOAD_SPECS[RescuePayloadType.FIRST_AID_TRAUMA_KIT])

        wind_fn = getattr(self.world, "wind", None).sample if hasattr(self.world, "wind") else None
        terrain_fn = getattr(self.world, "terrain_z", None)

        # 1. Simulate aerodynamic ballistic descent
        drop_res = self.drop_calc.simulate_drop(
            spec=spec,
            release_pos_ned=aircraft_pos_ned,
            release_vel_ned=aircraft_vel_ned,
            release_time=t_mission,
            wind_fn=wind_fn,
            terrain_z_fn=terrain_fn,
        )

        # 2. Compute error to target
        target_pos = np.array(op.profile.location_ned)
        drop_res.target_id = target_id
        drop_res.target_pos_ned = target_pos
        miss_dist = float(math.hypot(
            drop_res.impact_pos_ned[0] - target_pos[0],
            drop_res.impact_pos_ned[1] - target_pos[1]
        ))
        drop_res.miss_distance_m = miss_dist
        drop_res.success = bool(miss_dist <= 25.0)  # within usable reach

        # 3. Trigger physical / MAVLink actuator if connected
        if self.conn is not None:
            try:
                self.conn.drop_payload(channel=servo_channel, hold_open_s=0.5)
            except Exception as exc:
                log.warning("MAVLink drop command failed: %r", exc)

        # 4. Record payload in simulation world if available
        if self.world and hasattr(self.world, "reliefs"):
            self.world.reliefs.append({
                "drop_id": drop_res.drop_id,
                "type": p_type.value,
                "target_id": target_id,
                "impact_north_m": float(drop_res.impact_pos_ned[0]),
                "impact_east_m": float(drop_res.impact_pos_ned[1]),
                "miss_distance_m": miss_dist,
                "beacon_active": spec.beacon_equipped,
            })

        # 5. Update operation state
        op.drop_result = drop_res
        op.payload_dropped_at = t_mission
        op.state = RescueTaskState.PAYLOAD_DELIVERED if drop_res.success else RescueTaskState.BEACON_TRANSMITTING
        op.notes.append(
            f"Payload {p_type.value} delivered (impact error: {miss_dist:.1f} m, landing vel: {drop_res.impact_velocity_ms:.1f} m/s)"
        )

        self.completed_drops.append(drop_res)
        log.info("PAYLOAD DROPPED: %s for %s -> err=%.1fm (success=%s)",
                 p_type.value, op.op_id, miss_dist, drop_res.success)
        return drop_res

    # ------------------------------------------------------------------ #
    def dispatch_ground_team(self, target_id: int, team_type: RescueTeamType, t_mission: float) -> bool:
        """Command the emergency ground team / boat to deploy along the safe route."""
        if target_id not in self.operations:
            return False
        op = self.operations[target_id]
        op.active_team = team_type
        op.state = RescueTaskState.GROUND_TEAM_DISPATCHED
        op.notes.append(f"Ground rescue squad dispatched ({team_type.value}) at t={t_mission:.1f}s")
        return True

    def complete_rescue(self, target_id: int, t_mission: float) -> bool:
        """Mark the survivor as successfully evacuated/rescued."""
        if target_id not in self.operations:
            return False
        op = self.operations[target_id]
        op.state = RescueTaskState.RESCUE_COMPLETED
        op.rescue_completed_at = t_mission
        op.notes.append(f"Survivor successfully evacuated at t={t_mission:.1f}s")
        log.info("RESCUE COMPLETED for %s (%s)", op.op_id, op.profile.id_tag)
        return True

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """Aggregate summary of all rescue operations."""
        by_state: Dict[str, int] = {}
        for op in self.operations.values():
            s = op.state.value
            by_state[s] = by_state.get(s, 0) + 1

        misses = [d.miss_distance_m for d in self.completed_drops]
        mean_miss = float(np.mean(misses)) if misses else 0.0

        return {
            "total_operations": len(self.operations),
            "by_state": by_state,
            "completed_drops": len(self.completed_drops),
            "mean_drop_error_m": round(mean_miss, 2),
            "successful_drops": sum(1 for d in self.completed_drops if d.success),
            "active_operations": [op.to_dict() for op in self.operations.values()],
        }
