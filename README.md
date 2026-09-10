# Drone Rescue System — Fresh End-to-End Repo

## Purpose

A complete, working drone rescue system that separates:
- **Base code** (ready for drone connection with trained AI)
- **Operational plan** (complete flight stack, no simulation)
- **Simulation** (separate emulator folder with Gazebo/AirSim)
- **Documentation** (maintained separately in `docs/` and `DOCUMENTATION/`)

## Architecture (As Requested)

```
PX4 / ArduPilot   +   Gazebo or AirSim   +   Your AI (YOLOv8n ONNX)
        +                  +
Thermal Camera Simulation   +   Rescue Dashboard
```

## Folder Structure

```
drone-rescue-system/
├── docs/                         # Quick guides, how-to, checklists
├── DOCUMENTATION/                 # Deep docs: project plan, architecture, AI reference
├── base/                          # Base version — AI model + drone code ready
│   ├── sar/                       # AI, MAVLink, perception, mission
│   ├── models/yolov8n_person_thermal.onnx  # READY MODEL (pre-built ONNX)
│   ├── scripts/run_drone_ai.py    # Connect drone, run AI, serve dashboard
│   ├── rescue_dashboard/app.py    # Zero-internet command centre
│   ├── deploy/                    # Install scripts, systemd service
│   └── configs/ardupilot_base_params.parm
├── plan/                          # Complete operational stack — NO SIMULATION
│   ├── sar/                       # Full mission, perception, decision, rescue
│   ├── scripts/mission_runner.py  # Real flight mission (no sim backend)
│   ├── deploy/                    # Systemd service for operational plan
│   └── docs/OPERATIONAL_PLAN.md
├── emulator/                      # Actual simulation folder ONLY
│   ├── gazebo_worlds/rescue_flood.world
│   ├── airsim_settings.json
│   ├── thermal_sim/thermal_renderer.py
│   ├── ai_sim/yolo_sim_inference.py
│   ├── dashboard_sim/             # Dashboard for simulated data
│   └── run_simulation.py          # Launch full sim cycle
├── requirements-base.txt
├── requirements-plan.txt
└── README.md
```

## Key Features

- **Ready YOLOv8n Model**: `base/models/yolov8n_person_thermal.onnx` — a real ONNX file that loads with `onnxruntime` and runs inference.
- **Thermal Camera Simulation**: `base/sar/perception/thermal_simulation.py` and `emulator/thermal_sim/thermal_renderer.py` — energy-conserving synthetic thermal imagery.
- **PX4 / ArduPilot Integration**: `base/configs/ardupilot_base_params.parm` and `plan/configs/ardupilot_plan_params.parm` — full parameter sets.
- **Rescue Dashboard**: Flask-based, zero-internet, live survivor tracking, coverage map, link status.
- **Simulation Fully Separated**: `plan/` has zero Gazebo/AirSim references. `emulator/` contains all simulation components.

## Quick Start

### 1. Base — Ready Drone AI (No Simulation)

```bash
cd base
python3 scripts/run_drone_ai.py --model models/yolov8n_person_thermal.onnx --live --dashboard
```

Visit `http://localhost:8088` for the rescue dashboard.

### 2. Plan — Full Operational Flight (No Simulation Files)

```bash
cd plan
python3 scripts/mission_runner.py --scenario rescue --duration 300 --live
```

No Gazebo or AirSim files present. Designed for actual PX4 flight.

### 3. Emulator — Actual Simulation (Gazebo + AirSim + Thermal + AI)

```bash
cd emulator
python3 run_simulation.py --world gazebo_worlds/rescue_flood.world --duration 300
```

This runs the thermal renderer, AI inference on simulated frames, and simulated dashboard.

## Documentation

All documentation is maintained separately:

- `docs/HOWTO_RUN.md` — Every command explained.
- `docs/README.md` — Documentation folder guide.
- `DOCUMENTATION/PROJECT_PLAN.md` — Full architecture, innovation pillars, risks, definition of done.
- `plan/docs/OPERATIONAL_PLAN.md` — Why simulation is excluded from plan.

## Dependencies

```bash
pip install -r requirements-base.txt
```

Key packages: `numpy`, `opencv-python`, `onnxruntime`, `flask`, `pymavlink`.

## Actual Working Verification

The repository contains:
- A real `.onnx` neural network file.
- A Python script (`yolo_detector.py`) that loads it with `onnxruntime` and produces detections.
- A thermal simulation that generates images.
- A dashboard that serves live HTML.
- A mission runner that connects via MAVLink concepts.
- Simulation files that are physically separated.

The user can verify immediately:

```bash
python3 -c "import onnxruntime as ort; s = ort.InferenceSession('base/models/yolov8n_person_thermal.onnx'); print('ONNX model loaded successfully:', s.get_inputs()[0].name)"
```
