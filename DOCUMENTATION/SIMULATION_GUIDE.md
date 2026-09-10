# Simulation Guide — emulator/ Only

## Why Simulation is Separated

The `plan/` folder contains zero simulation files. This separation:
- Prevents simulation dependencies from leaking into flight code.
- Allows simulation development independently (different worlds, AI settings).
- Ensures bugs in simulation do not corrupt operational mission scripts.

## Components

| Component | File / Folder | Purpose |
|---|---|---|
| Gazebo World | `gazebo_worlds/rescue_flood.world` | Physical environment (flood water, victims) |
| AirSim Settings | `airsim_settings.json` | Multirotor settings, camera, sensors |
| Thermal Renderer | `thermal_sim/thermal_renderer.py` | Energy-conserving synthetic thermal imagery |
| AI Simulation | `ai_sim/yolo_sim_inference.py` | Runs same YOLO model against simulated frames |
| Dashboard Sim | `dashboard_sim/rescue_dashboard_sim.py` | Shows simulated detections |
| Runner | `run_simulation.py` | Orchestrates simulation cycle |

## Running Simulation

```bash
cd emulator
python3 run_simulation.py --world gazebo_worlds/rescue_flood.world --duration 300
```

Note: Actual Gazebo/AirSim should be launched externally (e.g., via ROS launch files or AirSim CLI). The `run_simulation.py` provides the Python-level integration but does not launch the 3D engines directly.
