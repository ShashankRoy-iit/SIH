#!/usr/bin/env python3
"""Multi-Scenario Benchmark Suite for Human Identification & Precision Rescue.

Evaluates:
1. Human identification and posture classification across disaster presets.
2. Physiological assessment and triage prioritization (Immediate, Delayed, Minor).
3. Ballistic air-drop precision and mean miss distance (m).
4. Safe ground and amphibious access route planning (ETA, distance, safety score).

Scenarios tested:
- `flood`: Inundation, water-clinging victims, flotation buoy deployments.
- `earthquake`: Urban collapse, trapped under rubble, extrication beacons.
- `wildfire`: Active fire fronts, smoke plumes, thermal blanket / rations delivery.
- `landslide`: Remote mudslides, steep terrain, 4x4 and foot squad routing.
- `stress`: Compound disaster with rain, smoke, and complex hazards.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sar.core.geo import GeoPoint
from sar.perception.human_id import (
    HumanIdentifier,
    HumanPosture,
    HumanProfile,
    RescueEquipmentNeed,
)
from sar.perception.thermal import PriorityTier
from sar.rescue.coordinator import RescueCoordinator
from sar.rescue.payload import BallisticDropCalculator, PAYLOAD_SPECS, RescuePayloadType
from sar.rescue.routing import GroundRescueRouter, RescueTeamType
from sar.sim.scenario import build_reference_scenario, scenario_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sar.benchmark.rescue")


def run_scenario_benchmark(scenario_name: str) -> Dict[str, Any]:
    """Run full rescue pipeline benchmark on a single scenario preset."""
    world, lwir_spec, rgb_spec = build_reference_scenario(scenario_name, smoke=False)
    human_id_engine = HumanIdentifier()
    drop_calc = BallisticDropCalculator()
    router = GroundRescueRouter()

    victims = getattr(world, "victims", [])
    n_victims = len(victims)

    profiles: List[HumanProfile] = []
    drop_results = []
    routes = []

    for idx, v in enumerate(victims):
        # Create synthetic track representation
        class MockObs:
            def __init__(self, t_val):
                self.t = t_val
                self.gsd_m = 0.12
                self.confidence = 0.90
                self.peak_temp_c = 36.8
                self.background_temp_c = 22.0
                self.modality = "fused"
                self.area_px = 42.0
                self.north = float(v.north)
                self.east = float(v.east)
                is_waving = (v.category == "hypothermia_risk" or idx % 3 == 0)
                self.attributes = {
                    "aspect": 1.1 + (0.5 if is_waving else 0.0),
                    "elongation": 1.8 if v.on_rooftop else 1.2,
                    "rgb_saliency": 72.0 if idx % 2 == 0 else 30.0,
                }

        class MockTrack:
            def __init__(self, victim, i):
                self.tid = i + 1
                self.label = "person"
                self.confidence = 0.91
                self.north = float(victim.north)
                self.east = float(victim.east)
                self.position_sigma_m = 1.2
                self.observations = [MockObs(0.0), MockObs(0.5), MockObs(1.0), MockObs(1.5)]
                self.n_obs = 4
                self.immobility = 0.92 if victim.on_rooftop else 0.15
                self.speed_ms = 0.04
                self.age_s = 35.0

            def geo(self, o):
                return GeoPoint(o.lat + 0.0001, o.lon + 0.0001, o.alt)

        tr = MockTrack(v, idx)
        water_depth = 0.85 if v.in_water else 0.0
        ambient = getattr(world.spec, "ambient_c", 22.0)
        wind = getattr(world.wind, "speed_ms", 3.0)

        # 1. Human Identification
        prof = human_id_engine.identify(
            track=tr,
            water_depth_m=water_depth,
            ambient_temp_c=ambient,
            wind_speed_ms=wind,
        )
        profiles.append(prof)

        # 2. Precision Ballistic Air-Drop with Inverse Lead Targeting
        p_need = prof.recommended_equipment
        p_type = {
            RescueEquipmentNeed.FLOTATION_BUOY: RescuePayloadType.FLOTATION_BUOY,
            RescueEquipmentNeed.FIRST_AID_TRAUMA_KIT: RescuePayloadType.FIRST_AID_TRAUMA_KIT,
            RescueEquipmentNeed.THERMAL_EMERGENCY_BLANKET: RescuePayloadType.THERMAL_EMERGENCY_BLANKET,
            RescueEquipmentNeed.LORA_LOCATOR_BEACON: RescuePayloadType.LORA_LOCATOR_BEACON,
            RescueEquipmentNeed.WATER_RATIONS_PACK: RescuePayloadType.WATER_RATIONS_PACK,
            RescueEquipmentNeed.EXTRICATION_ASSISTANCE: RescuePayloadType.EXTRICATION_MARKER,
        }.get(p_need, RescuePayloadType.FIRST_AID_TRAUMA_KIT)

        spec = PAYLOAD_SPECS[p_type]
        target_ned = np.array([v.north, v.east, 0.0])

        # Compute lead solution from drone cruising at 40m AGL, 8 m/s, heading 45 deg
        sol = drop_calc.compute_release_solution(
            spec=spec,
            target_pos_ned=target_ned,
            aircraft_alt_agl_m=40.0,
            aircraft_ground_speed_ms=8.0,
            approach_heading_deg=45.0,
        )

        drop_res = drop_calc.simulate_drop(
            spec=spec,
            release_pos_ned=sol["release_pos_ned"],
            release_vel_ned=np.array([5.65, 5.65, 0.0]),
            release_time=10.0 + idx * 5.0,
        )
        drop_res.target_id = idx + 1
        miss = float(math.hypot(drop_res.impact_pos_ned[0] - v.north, drop_res.impact_pos_ned[1] - v.east))
        drop_res.miss_distance_m = miss
        drop_res.success = miss <= 20.0
        drop_results.append(drop_res)

        # 3. Safe Access Evacuation Route
        team = RescueTeamType.AMPHIBIOUS_RESCUE_BOAT if v.in_water else RescueTeamType.FOOT_RESCUE_TEAM
        rt = router.plan_route(
            start_ned=(0.0, 0.0, 0.0),
            target_ned=target_ned,
            team_type=team,
            world=world,
            target_id=idx + 1,
        )
        routes.append(rt)

    # Aggregate Statistics
    miss_dists = [d.miss_distance_m for d in drop_results]
    mean_miss = float(np.mean(miss_dists)) if miss_dists else 0.0
    hit_rate = float(sum(1 for d in drop_results if d.success) / max(len(drop_results), 1))

    triage_counts = {
        PriorityTier.IMMEDIATE.value: sum(1 for p in profiles if p.triage_priority == PriorityTier.IMMEDIATE),
        PriorityTier.DELAYED.value: sum(1 for p in profiles if p.triage_priority == PriorityTier.DELAYED),
        PriorityTier.MINOR.value: sum(1 for p in profiles if p.triage_priority == PriorityTier.MINOR),
    }

    postures = {}
    for p in profiles:
        pos = p.posture.value
        postures[pos] = postures.get(pos, 0) + 1

    route_dists = [r.total_distance_m for r in routes]
    route_etas = [r.estimated_transit_time_s for r in routes]
    safety_scores = [r.safety_score for r in routes]

    return {
        "scenario": scenario_name,
        "victims_count": n_victims,
        "identified_count": sum(1 for p in profiles if p.is_human),
        "posture_distribution": postures,
        "triage_distribution": triage_counts,
        "mean_drop_error_m": round(mean_miss, 2),
        "drop_hit_rate_pct": round(hit_rate * 100.0, 1),
        "mean_route_distance_m": round(float(np.mean(route_dists)), 1) if route_dists else 0.0,
        "mean_route_eta_min": round(float(np.mean(route_etas)) / 60.0, 1) if route_etas else 0.0,
        "mean_safety_score": round(float(np.mean(safety_scores)), 2) if safety_scores else 1.0,
        "payloads_deployed": [d.to_dict() for d in drop_results],
        "profiles": [p.to_dict() for p in profiles],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAR Rescue Pipeline Benchmark")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    art_dir = Path(args.artifacts)
    art_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["flood", "earthquake", "wildfire", "landslide", "stress"]
    all_results = {}

    print("\n" + "=" * 80)
    print("  SAHYOG AUTONOMOUS HUMAN IDENTIFICATION & PRECISION RESCUE BENCHMARK")
    print("=" * 80 + "\n")

    for sc in scenarios:
        log.info("Benchmarking scenario '%s'...", sc)
        res = run_scenario_benchmark(sc)
        all_results[sc] = res

    # Output JSON artifact
    out_file = art_dir / "rescue_benchmark_results.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    log.info("Saved benchmark results to %s", out_file)

    # Print Summary Table
    print("\n" + "-" * 88)
    print(f"{'Scenario':<12} | {'Victims':<8} | {'Identified':<10} | {'Immediate':<9} | {'Drop Err (m)':<12} | {'Hit Rate':<8} | {'Route ETA':<9} | {'Safety'}")
    print("-" * 88)
    for sc, r in all_results.items():
        print(
            f"{sc:<12} | {r['victims_count']:<8} | {r['identified_count']:<10} | "
            f"{r['triage_distribution']['immediate']:<9} | {r['mean_drop_error_m']:<12.1f} | "
            f"{r['drop_hit_rate_pct']:<7.1f}% | {r['mean_route_eta_min']:<6.1f} min | {r['mean_safety_score']:<0.2f}"
        )
    print("-" * 88 + "\n")


if __name__ == "__main__":
    main()
