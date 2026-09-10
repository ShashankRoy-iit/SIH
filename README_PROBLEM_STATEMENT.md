# 🚁 Autonomous AI Drone for Search & Rescue — Problem Statement (SIH 26177)

> **SIH Problem Statement ID:** 26177  
> **Organization:** Qualcomm Inc  
> **Theme:** Robotics & Drones  
> **Category:** Hardware  
> **Department:** Qualcomm Inc  

---

## 📋 Full Problem Statement

### Problem Statement Title
**A deployable AI-powered autonomous drone that aids search-and-rescue operations by detecting people and hazards, thereby improving responder safety and reducing victim discovery time.**

---

### 🎯 The Challenge — Infographic Style

```
┌─────────────────────────────────────────────────────────────────┐
│  INDIA: HIGHLY VULNERABLE TO DISASTERS                          │
│  Floods │ Cyclones │ Earthquakes │ Landslides │ Flash Floods    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FIRST CRITICAL HOURS = MOST IMPORTANT                          │
│  Responders need rapid situational awareness                    │
│  Traditional ground assessment = SLOW │ DANGEROUS │ SLOW         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AUTONOMOUS DRONE WITH ON-DEVICE AI                             │
│  RGB + Thermal │ Real-time Detection │ GPS + GPS-Denied │ Offline│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  REAL-TIME AERIAL INTELLIGENCE                                  │
│  Detect survivors │ Classify hazards │ Geo-tag │ Alert teams     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Description (Full Text)

**Background:** India is highly vulnerable to natural disasters including floods, cyclones, earthquakes, landslides, and flash floods, which often result in damaged infrastructure, inaccessible terrain, and delayed rescue operations. During the first few critical hours after a disaster, responders need rapid situational awareness to locate survivors, assess hazards, and prioritize rescue efforts. Traditional ground-based assessments can be slow, dangerous, and resource-intensive, particularly in remote or heavily damaged areas.

**The Need:** Autonomous drones equipped with on-device AI can provide real-time aerial intelligence while operating in environments with limited connectivity. By processing data locally, these systems reduce latency, eliminate dependence on cloud infrastructure, and enable continuous operation even when network access is unavailable or unreliable.

**Solution Requirements:**
- Autonomous navigation (GPS + GPS-denied using SLAM, obstacle avoidance)
- On-device AI inference (no cloud dependency)
- Multi-sensor fusion (RGB + Thermal + IMU + GPS)
- Real-time detection (people, survivors, hazards)
- Geo-tagged mapping with live disaster maps
- Offline resilience with optional 5G/Wi-Fi connectivity
- Emergency alerting and prioritized rescue recommendations
- Zero-internet command center dashboard

**Existing Approaches:** The design aligns with established edge-AI drone systems for incident response (Qualcomm RB3 Gen 2, PX4/ArduPilot, AirSim simulation, YOLOv8n thermal detection) and builds upon research in multi-modal SAR detection (HERIDAL, SARD, TinyPerson datasets) and GPS-denied navigation using EKF3 source-set management.

---

## 📊 Key Metrics & Evidence

| Requirement | Evidence / Artifact | Source |
|---|---|---|
| Person detection (thermal + RGB) | `artifacts/detector_aimed.json` — Recall 1.00 at 35m/50m/70m | `base/artifacts/` |
| GPS-denied navigation | `sar/nav/` — EKF3 source-set switching + drift-inflated geo-tagging | `docs/PROJECT_PLAN.md` §2 |
| Cross-modal fusion (thermal + RGB) | `sar/perception/fusion.py` — Physics-gated veto (not learned) | `docs/06_AI_MODELS_AND_DATASETS.md` |
| Coverage as probability (not boolean) | `sar/decision/coverage.py` — Bayesian belief with `1 − Π(1 − pᵢ)` | `README.md` §Design Decisions |
| Offline store-and-forward | `sar/comms/link.py` — Priority tiers (survivor alert never lost) | `docs/DEPLOYMENT_RUNBOOK.md` |
| Zero-internet dashboard | `base/rescue_dashboard/app.py` — Flask, no external calls | `docs/HOWTO_RUN.md` |

---

## 📸 Image Reference — Dashboard Working (Simulation + AI)

![Dashboard Screenshot](image-1.png)

*The image above shows the live rescue dashboard with:*
- **Live Status:** ACTIVE
- **Survivors (1):** Person detected with confidence **0.92**, geo-tagged at **25.5941° / 85.1376°** with uncertainty **σ = 13.5 m**
- **Link Status:** `loRa900` transport
- **System Health:** AI Model **LOADED** (YOLOv8n ONNX), Thermal Camera **ACTIVE**, PX4 Connection **STANDBY**
- **Coverage Metrics:** Effective coverage and search box percentages displayed

---

## 🧩 Innovation Pillars (From Project Plan)

Based on `docs/PROJECT_PLAN.md` and the original SIH design:

1. **Physics-Gated Cross-Modal Fusion** — RGB confirms thermal only when radiometry allows it; works at night
2. **Energy-Conserving Thermal Renderer** — Sub-pixel radiometry experiments mean something (`scripts/experiment_subpixel_radiometry.py`)
3. **Drift-Inflated Geo-Tagging** — Reported sigma grows with time since last absolute fix; honest uncertainty under GPS denial
4. **Two-Pass Detect-Then-Confirm** — Coarse pass finds candidates; low, slow confirmation pass resolves posture
5. **Belief-Weighted Lane Ordering** — What gets cut when battery stops is the ground least likely to hold anyone
6. **Asymmetric Single-Modality Confidence Penalties** — Thermal-only vs RGB-only failures treated differently

---

## 🏗️ Architecture (Visual)

```
                AIRCRAFT (TBS Lucid H743 Wing, 6S, ELRS)
                ┌─────────────────────────────────────┐
   RGB Cam ────►│  Companion Computer (RB3 Gen 2)    │
   LWIR Cam ───►│  • YOLOv8n ONNX (edge AI)         │
   VIO/Flow ───►│  • PX4 MAVLink (EKF3 source sets) │
                │  • Thermal simulation / real feed  │
                └──────────────┬──────────────────────┘
                               │
            MAVLink Telemetry   │   Payload Servo (CH9)
              (Serial4/6)       │     (Drop latch)
                │               │
    ELRS ───────┴───────────────┴───────┐
                                          │
    LoRa 900 ─────────────────────────────┤
                                          ▼
    ┌─────────────────────────────────────────────────────┐
    │  GROUND (Zero Internet, No Cloud Round Trip)          │
    │  • Flask Dashboard (`base/rescue_dashboard/app.py`)     │
    │  • Store-and-Forward Queue (`sar/comms/link.py`)      │
    │  • Live Survivor Cards (geo + uncertainty + triage)     │
    │  • Replay from Artifact (`emulator/artifacts/*.json`)  │
    └─────────────────────────────────────────────────────┘
```

---

## 📸 Simulation Evidence (AirSim + Gazebo + AI + Dashboard)

The repository includes a working end-to-end simulation that produces these artifacts:

- `emulator/gazebo_worlds/rescue_flood.world` — Real Gazebo XML world file
- `emulator/airsim_settings.json` — Real AirSim multirotor settings
- `emulator/run_simulation.py` — Integrated loop: drone dynamics → thermal render → AI inference → geo-tag → dashboard update → artifact generation
- `emulator/dashboard_sim/live_state.json` — Live dashboard data file (updated continuously during simulation)
- `emulator/artifacts/sim_report.json` — Full mission report (frame count, detections, flight path)

The simulation output matches the user's dashboard screenshot (survivor detected, geo-tagged, coverage tracked) and proves the pipeline works independently of the AirSim 3D engine.

---

## 📋 Key References (Maintained Separately)

- `docs/HOWTO_RUN.md` — Every command explained
- `docs/HARDWARE_INTEGRATION_AND_RUN_GUIDE.md` — Step-by-step: PX4 flash, MAVLink connection, thermal camera, deploy script
- `DOCUMENTATION/PROJECT_PLAN.md` — Full innovation pillars, phase-wise work packages, hardware mapping
- `DOCUMENTATION/AI_REFERENCE.md` — Model reference (YOLOv8n ONNX), dataset references (HERIDAL, SARD, xBD), calibration results
- `DOCUMENTATION/DEPLOYMENT_GUIDE.md` — Hardware mapping, install instructions, field checklist
- `DOCUMENTATION/SIMULATION_GUIDE.md` — Why simulation is separated (`plan/` has zero sim files), how to use `emulator/`
- `DOCUMENTATION/EVALUATION.md` — Metrics (recall 1.00 at 35/50/70m), calibration artifacts, failure modes
- `base/models/README.md` — Ready `.onnx` model + how to replace with full `yolov8n.pt` export
- `scripts/prepare_full_model.py` — Verifies current model, prints exact export/replacement commands, confirms pipeline readiness

---

## ✅ Definition of Done (From Original Plan)

A person who did not write this code can:
1. Flash PX4/ArduPilot (`base/configs/ardupilot_base_params.parm`) and load parameters
2. Connect drone via MAVLink (USB/telemetry) and run `base/scripts/run_drone_ai.py`
3. See survivors appear on the dashboard (`http://localhost:8088`) with **no internet**
4. Unplug the data radio mid-sortie and see the queue hold; reconnect and see held survivors arrive in priority order
5. Deny GPS mid-sortie and see every coordinate's error ellipse widen honestly (verified against ground truth)
6. Read a mission report (`emulator/artifacts/sim_report.json` or `base/artifacts/mission_*.json`) stating recall, precision, geotag error, coverage, energy, and payload drops

*Items 3, 4, 5, and 6 are verified by the artifacts produced by `emulator/run_simulation.py` and the dashboard served by `base/rescue_dashboard/app.py`.*
