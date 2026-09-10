# Operational Plan — Complete (No Simulation)

## Scope

This folder implements the full project plan **excluding any simulation components**. It is designed for actual PX4 / ArduPilot flight, real thermal cameras, and live MAVLink communication.

## Key Decisions

- **Simulation excluded**: No `sim/` folder, no Gazebo worlds, no AirSim settings.
- **Real MAVLink only**: `sar/mavlink/connection.py` connects to physical or HITL flight controllers.
- **Real thermal camera**: `thermal_camera/` provides hardware interfaces, not synthetic renderers.
- **AI model**: Uses same `models/yolov8n_person_thermal.onnx` from base/ (or reference path).
- **Dashboard**: Zero-internet, live only.

## Simulation Separation

Actual simulation must be run from `../emulator/`. This separation ensures:
- The operational stack has zero simulation dependencies.
- Simulation can be developed independently (different world files, different AI settings, different thermal modeling).
- A bug in simulation cannot corrupt the operational flight code.

## File Checklist

- [x] `README.md`
- [x] `sar/` (core, mavlink, mission, perception, decision, rescue, ai, gcs)
- [x] `scripts/mission_runner.py`
- [x] `deploy/`
- [x] `configs/`
- [x] `docs/OPERATIONAL_PLAN.md`
- [ ] `tests/` (add as needed)
