# Emulator — Actual Simulation Folder

Contains all simulation components: **Gazebo worlds**, **AirSim settings**, **thermal camera simulation**, **AI simulation pipeline**, and **dashboard simulation**.

This folder is separated from `plan/` (operational, no sim) and `base/` (ready flight code).

## Quick Start

```bash
cd emulator
python3 run_simulation.py --world gazebo_worlds/rescue_flood.world --thermal thermal_sim/thermal_renderer.py --ai ai_sim/yolo_sim_inference.py --dashboard dashboard_sim/rescue_dashboard_sim.py --duration 300
```

## Components

- `gazebo_worlds/` — Physical world files (flood, earthquake, urban rescue).
- `airsim_settings.json` — AirSim configuration for visual simulation.
- `simulation_runner/` — Main simulation launcher.
- `thermal_sim/` — Energy-conserving thermal renderer (same physics as base, but explicitly for simulation).
- `ai_sim/` — YOLO inference running against simulated thermal frames.
- `dashboard_sim/` — Dashboard that shows simulated detections.
