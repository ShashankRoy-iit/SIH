# Drone Rescue System — Full Project Plan

## 1. Architecture Overview

```
PX4 / ArduPilot (Flight Controller)
        +
Your AI (YOLOv8n Trained Person Detection — ONNX / QNN / TFLite)
        +
Thermal Camera Simulation (Energy-Conserving LWIR + RGB)
        +
Rescue Dashboard (Zero-Internet Command Centre)
        +
Gazebo / AirSim (Simulation — ONLY in emulator/ folder)
```

## 2. Folder Structure

| Folder | Purpose | Simulation? |
|---|---|---|
| `base/` | Base version — ready for drone connection. Trained AI model, PX4 MAVLink, thermal camera code, dashboard. | **No** (only real/hardware simulation) |
| `plan/` | Complete operational stack according to plan. All flight, mission, payload, deploy scripts. Zero Gazebo/AirSim files. | **None** |
| `emulator/` | Actual simulation folder. Contains Gazebo worlds, AirSim settings, thermal simulation renderer, AI sim pipeline, dashboard sim. | **Full simulation** |
| `docs/` + `DOCUMENTATION/` | All documentation maintained separately. | N/A |

## 3. Design Decisions

- **Library-based simulation, no 3D graphics** for algorithm testing.
- **Two interchangeable backends**: PX4 SITL / Real Drone. Same MAVLink wire.
- **Energy-conserving thermal renderer** — sub-pixel radiometry means something.
- **Store-and-forward link** — priority tiers, so a survivor alert never loses to imagery.
- **Coverage is probability**, not boolean.

## 4. AI Model

- Model: YOLOv8n (nano), trained for thermal + RGB person detection.
- Format: ONNX (included as `.onnx`) + TFLite conversion script.
- Inference time: <30 ms on modern laptop; target <50 ms on Qualcomm RB3 Gen 2.
- Detection calibrated by Ground Sample Distance (GSD).

## 5. Simulation Strategy (emulator/ only)

- **Gazebo**: Physical world model (buildings, flood water, smoke).
- **AirSim**: Optional high-fidelity visual simulator with thermal camera plugin.
- **Thermal Camera Simulation**: Energy-conserving numpy renderer producing LWIR images.
- **AI Pipeline**: Same ONNX model runs against simulated thermal frames; detections geo-tagged using simulated PX4 MAVLink estimates.
- **Rescue Dashboard**: Flask-based live dashboard showing survivor list, coverage map, link status.

## 6. Hardware Mapping

- Flight Controller: PX4 / ArduPilot (`TBS_LUCID_H7_WING`)
- Companion Computer: Qualcomm RB3 Gen 2 / VOXL 2
- Thermal Camera: LWIR module (simulated in `emulator/`)
- RGB Camera: Standard FPV / companion camera
- Control Link: ELRS + MAVLink telemetry
- Data Link: LoRa 900 (simulated link budget)
- Payload: Servo latch for drop (channel 9)

## 7. Definition of Done

A person who did not write it can:
1. Clone repo.
2. Run `base/scripts/run_drone_ai.py` with drone connected (or simulated thermal feed).
3. See people identified on dashboard.
4. Switch to `plan/` for full mission execution (no simulation files present).
5. Switch to `emulator/` for full Gazebo/AirSim simulation with thermal camera and AI.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Endurance too short | Belief-weighted lane ordering, plan truncation |
| Small-object thermal recall weak | Confirm at lower altitude; physics-gated cross-modal veto |
| GPS-denied drift exceeds geo-tag honesty | Sigma grows with time since fix |
| No companion computer at demo | MiniSITL + laptop-hosted perception reproduces full stack |
