# Plan — Complete Operational Stack (No Simulation)

This folder contains the **operational plan** according to the full project plan.
Every file needed for real drone flight is here. **No Gazebo, no AirSim, no simulation launcher files.**
Simulation is handled separately in `emulator/`.

## What's Inside

- `sar/core/` — Geo conversions, frames, clock, events.
- `sar/mavlink/` — PX4 MAVLink connection, protocol constants.
- `sar/perception/` — AI pipeline (YOLOv8n), thermal camera interface (real, not simulated), cross-modal fusion, tracker.
- `sar/mission/` — Mission runner (real flight commands, no sim backend).
- `sar/decision/` — Search planner, coverage grid, Bayesian belief.
- `sar/rescue/` — Payload coordinator, drop routing, A* ground team routing.
- `sar/ai/` — Model registry, runtime, calibration, export scripts.
- `sar/gcs/` — Rescue dashboard (same as base, zero internet).
- `scripts/` — Mission runner, onboard loop, deploy helpers.
- `deploy/` — Systemd service, companion install, device rules.
- `configs/ardupilot_plan_params.parm` — Full PX4 parameter set for real flight.
- `docs/OPERATIONAL_PLAN.md` — Detailed plan documentation.

## Quick Start (Real Drone / Hardware Only)

```bash
cd plan
python3 scripts/mission_runner.py --scenario flood --duration 300 --live
```

## Key Design Choice: No Simulation Here

This folder intentionally excludes:
- Any `sim/` folder.
- Any Gazebo world files.
- Any AirSim settings or launcher scripts.
- Any `run_simulation.py` or simulation-only scripts.

Actual simulation should be run from `../emulator/` using the separate simulation folder.
