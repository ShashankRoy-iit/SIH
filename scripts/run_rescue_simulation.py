#!/usr/bin/env python3
"""Run an autonomous Search, Human Identification, and Precision Rescue simulation.

Simulates the complete autonomy loop:
1. Flies search pattern over disaster area (flood basin, debris, thermal hotspots).
2. Runs dual-modal RGB + LWIR thermal perception pipeline.
3. Classifies survivor posture, SOS waving gestures, demographic profile, and vitals.
4. Computes ballistic drop solutions for emergency payloads and deploys precision drops.
5. Plans obstacle-aware A* safe transit routes for ground/boat rescue squads.
6. Emits a full mission report artifact in JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# --- repo-root bootstrap ---------------------------------------------------
# Running `python scripts/<name>.py` puts scripts/ on sys.path, not the repo
# root, so `import sar` fails.  This makes the script runnable from a clone with
# no install step, and refuses to run against a foreign PyPI `sar` package.
# See scripts/_bootstrap.py and docs/HOWTO_RUN.md.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap as _bootstrap  # noqa: E402

_bootstrap()
# ---------------------------------------------------------------------------

from sar.core.geo import GeoPoint
from sar.decision.coverage import BoustrophedonPlanner
from sar.mission.runner import MissionRunner
from sar.nav import (
    EkfSourceManager,
    ExternalNavFeeder,
    NavQualityMonitor,
    SimulatedVioSource,
    SourceSetPolicy,
)
from sar.rescue.coordinator import RescueCoordinator
from sar.sim.scenario import build_reference_scenario
from sar.sim.sitl import MiniSITL
from sar.vehicle.sensors import VioSensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sar.sim.rescue")


def run_simulation(
    scenario_name: str = "flood",
    duration_s: float = 60.0,
    speedup: float = 2.0,
    live_gcs: bool = False,
    return_home: bool = False,
    port: int = 5760,
    gcs_port: int = 8088,
    artifacts_dir: str = "artifacts",
) -> Path:
    art_path = Path(artifacts_dir)
    art_path.mkdir(parents=True, exist_ok=True)

    log.info("Building scenario '%s'...", scenario_name)
    world, lwir_spec, rgb_spec = build_reference_scenario(scenario_name, smoke=True)

    log.info("Spawning MiniSITL simulated vehicle plant...")
    sitl = MiniSITL(home=world.origin, speedup=speedup, port=port)
    sitl.start()

    conn = sitl.connect()
    conn.request_streams()
    time.sleep(0.5)
    conn.pump(0.5)

    # Launch VIO External Nav Feeder so EKF pre-arm checks pass cleanly
    vio = VioSensor(seed=42)
    source = SimulatedVioSource(vio, truth_fn=lambda t: sitl.truth())
    feeder = ExternalNavFeeder(conn, source, rate_hz=30.0, use_odometry=True)
    feeder.start()
    t0 = time.time()
    while feeder.measured_hz < 15.0 and time.time() - t0 < 5.0:
        time.sleep(0.1)

    monitor = NavQualityMonitor(denial_zones=[], hysteresis_s=1.0)
    manager = EkfSourceManager(conn, monitor, extnav=feeder, policy=SourceSetPolicy())

    runner = MissionRunner(
        conn=conn,
        world=world,
        rgb_spec=rgb_spec,
        lwir_spec=lwir_spec,
        state_fn=lambda t: sitl.truth(),
        scenario_name=scenario_name,
        artifacts_dir=art_path,
        auto_rescue=True,
    )

    # Optional Live Dashboard Server
    dashboard = None
    if live_gcs:
        try:
            from sar.gcs.dashboard import Dashboard
            dashboard = Dashboard(uplink=runner.uplink, runner=runner, world=world, port=gcs_port)
            dashboard.start()
            log.info("Live GCS Dashboard active at http://0.0.0.0:%d", gcs_port)
        except Exception as exc:
            log.warning("Could not start dashboard: %r", exc)

    log.info("Creating survey plan for %dx%d m world (budget: %.0fs)...",
             world.north_m, world.east_m, duration_s)
    runner.make_plan(max_duration_s=None, area_north_m=min(320.0, world.north_m), area_east_m=min(320.0, world.east_m))

    log.info("Executing Autonomous Mission Loop...")
    t_start = time.time()
    try:
        report = runner.run(max_duration_s=duration_s, takeoff_altitude_agl_m=45.0, return_home=return_home)
    finally:
        feeder.stop()
        sitl.stop()
        if dashboard:
            dashboard.stop()

    out_file = art_path / f"rescue_mission_{scenario_name}.json"
    report.write(out_file)
    log.info("Mission finished in %.1fs wall-clock (%.1fs simulated flight)",
             time.time() - t_start, report.flight_time_s)
    log.info("Report written to %s", out_file)
    log.info("Survivors Identified: %d | Payloads Dropped: %d | Mean Drop Error: %.1fm",
             len(report.human_identifications),
             report.rescue.get("completed_drops", 0),
             report.rescue.get("mean_drop_error_m", 0.0))

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAR Autonomous Rescue Simulation")
    parser.add_argument("--scenario", default="flood", help="Scenario name (flood, earthquake, wildfire, landslide)")
    parser.add_argument("--duration", type=float, default=60.0, help="Mission max duration in seconds")
    parser.add_argument("--speedup", type=float, default=1.0, help="Simulation speedup multiplier")
    parser.add_argument("--live", action="store_true", help="Launch live web dashboard")
    parser.add_argument("--return-home", action="store_true", help="RTL at end of sortie")
    parser.add_argument("--port", type=int, default=5760, help="MAVLink TCP port")
    parser.add_argument("--gcs-port", type=int, default=8088, help="Dashboard web port")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory")

    args = parser.parse_args()
    run_simulation(
        scenario_name=args.scenario,
        duration_s=args.duration,
        speedup=args.speedup,
        live_gcs=args.live,
        return_home=args.return_home,
        port=args.port,
        gcs_port=args.gcs_port,
        artifacts_dir=args.artifacts,
    )


if __name__ == "__main__":
    main()
