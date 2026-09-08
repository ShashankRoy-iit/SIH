"""Unit and integration tests for Rescue subsystem (Payload Ballistics, Routing, Coordinator)."""

import math
import numpy as np
import pytest

from sar.core.geo import GeoPoint
from sar.perception.human_id import HumanPosture, HumanProfile, RescueEquipmentNeed
from sar.perception.thermal import PriorityTier
from sar.rescue.coordinator import RescueCoordinator, RescueOperation, RescueTaskState
from sar.rescue.payload import (
    BallisticDropCalculator,
    PAYLOAD_SPECS,
    PayloadSpec,
    RescuePayloadType,
)
from sar.rescue.routing import GroundRescueRouter, RescueRoute, RescueTeamType
from sar.sim.scenario import build_reference_scenario


def test_ballistic_drop_physics_simulation():
    calc = BallisticDropCalculator()
    spec = PAYLOAD_SPECS[RescuePayloadType.FIRST_AID_TRAUMA_KIT]

    # Drop from 45m altitude at 8 m/s forward speed
    rel_pos = np.array([100.0, 100.0, -45.0])
    rel_vel = np.array([8.0, 0.0, 0.0])

    res = calc.simulate_drop(
        spec=spec,
        release_pos_ned=rel_pos,
        release_vel_ned=rel_vel,
        release_time=10.0,
    )

    assert res.impact_pos_ned[2] >= -0.1  # Reached ground
    assert res.impact_time > 10.0
    assert len(res.trajectory) > 10
    # Parachute should deploy and slow down vertical descent
    assert res.impact_velocity_ms < 15.0
    # Drift forward due to forward release momentum
    assert res.impact_pos_ned[0] > 100.0


def test_inverse_ballistic_targeting_lead_solution():
    calc = BallisticDropCalculator()
    spec = PAYLOAD_SPECS[RescuePayloadType.FLOTATION_BUOY]
    target_pos = np.array([250.0, 300.0, 0.0])

    sol = calc.compute_release_solution(
        spec=spec,
        target_pos_ned=target_pos,
        aircraft_alt_agl_m=40.0,
        aircraft_ground_speed_ms=10.0,
        approach_heading_deg=45.0,
    )

    assert sol["lead_distance_m"] > 0.0
    assert sol["release_pos_ned"][2] == -40.0
    # Verify release position is before target along approach vector
    assert sol["release_pos_ned"][0] < target_pos[0]
    assert sol["release_pos_ned"][1] < target_pos[1]


def test_ground_and_boat_safe_routing():
    world, _, _ = build_reference_scenario("flood", smoke=True)
    router = GroundRescueRouter()

    # Plan foot rescue route
    start = (0.0, 0.0, 0.0)
    target = (150.0, 150.0, 0.0)
    route_foot = router.plan_route(
        start_ned=start,
        target_ned=target,
        team_type=RescueTeamType.FOOT_RESCUE_TEAM,
        world=world,
    )

    assert route_foot.total_distance_m > 0.0
    assert route_foot.estimated_eta_s > 0.0
    assert len(route_foot.waypoints) >= 2

    # Plan amphibious boat route
    route_boat = router.plan_route(
        start_ned=start,
        target_ned=target,
        team_type=RescueTeamType.AMPHIBIOUS_RESCUE_BOAT,
        world=world,
    )

    assert route_boat.total_distance_m > 0.0
    assert route_boat.team_type == RescueTeamType.AMPHIBIOUS_RESCUE_BOAT


def test_rescue_coordinator_lifecycle():
    origin = GeoPoint(25.185, 75.835, 250.0)
    world, _, _ = build_reference_scenario("flood", smoke=True)
    coord = RescueCoordinator(origin=origin, world=world, auto_drop_enabled=True)

    class MockTrack:
        def __init__(self, tid=1):
            self.tid = tid
            self.north = 120.0
            self.east = 120.0
            self.confidence = 0.92
            self.immobility = 0.8
            self.speed_ms = 0.0
            self.age_s = 25.0
            self.label = "person"
            self.observations = []

        def geo(self, o):
            return GeoPoint(o.lat, o.lon, o.alt)

    # 1. Ingest Track
    tr = MockTrack(tid=1)
    op = coord.process_track(tr, t_mission=10.0, water_depth_m=0.7)

    assert op.target_id == 1
    assert op.state == RescueTaskState.HUMAN_IDENTIFIED
    assert op.assigned_payload == RescuePayloadType.FLOTATION_BUOY
    assert "foot_team" in op.ground_routes

    # 2. Execute Payload Drop
    ac_pos = np.array([115.0, 115.0, -40.0])
    ac_vel = np.array([5.0, 5.0, 0.0])
    drop_res = coord.execute_payload_drop(
        target_id=1,
        aircraft_pos_ned=ac_pos,
        aircraft_vel_ned=ac_vel,
        t_mission=15.0,
    )

    assert drop_res is not None
    assert op.drop_result is not None
    assert op.state in (RescueTaskState.PAYLOAD_DELIVERED, RescueTaskState.BEACON_TRANSMITTING)

    # 3. Dispatch Ground Squad
    assert coord.dispatch_ground_team(1, RescueTeamType.AMPHIBIOUS_RESCUE_BOAT, t_mission=20.0)
    assert op.state == RescueTaskState.GROUND_TEAM_DISPATCHED

    # 4. Complete Rescue
    assert coord.complete_rescue(1, t_mission=45.0)
    assert op.state == RescueTaskState.RESCUE_COMPLETED

    summary = coord.summary()
    assert summary["total_operations"] == 1
    assert summary["completed_drops"] == 1
