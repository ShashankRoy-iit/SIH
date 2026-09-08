# Autonomous AI Drone for Search & Rescue — Project Plan

**SIH Problem Statement ID 26177** · Organisation: **Qualcomm Inc** · Theme: Robotics & Drones

> A deployable AI-powered autonomous drone that aids search-and-rescue operations by
> detecting people and hazards using RGB + thermal imaging with on-device edge AI,
> navigating with GPS and in GPS-denied environments, classifying hazards, building
> geo-tagged maps, operating with offline resilience, and reporting to a
> command-centre dashboard.

This document is the plan. `README.md` is the orientation; `artifacts/` holds the
measured results the claims below rest on.

---

## 1. How the problem is being read

Five requirements are easy to demo and hard to deploy. The distinction drives
every design decision in this plan, so it is stated first.

| Requirement | The demo version | The deployable version |
|---|---|---|
| Detect people | A YOLO model with good mAP on a benchmark | Recall at the ground sample distance the airframe can actually hold, with a false-alarm rate a controller can tolerate for eight hours |
| GPS-denied nav | "We ran with GPS off" | A position estimate whose *reported uncertainty* bounds its true error, so a coordinate sent to a rescue team is honest |
| Hazard classification | A classifier over hazard classes | Bi-directional anomaly detection — expected-hot-things-found-cold are as much a hazard as the reverse |
| Geo-tagged mapping | Pin dropped on a map | An error ellipse that grows correctly with time since last fix, so pins under denial are visibly less trustworthy |
| Offline resilience | It keeps flying | Store-and-forward with priority tiers, so a link outage costs latency for imagery and never costs a survivor |

The project is judged on the right-hand column. Everything below is organised to
produce evidence for it rather than for the left.

**A note on what "deployable" costs.** The reference flood basin in this project is
900 × 900 m. Surveying it at the ground sample distance where detection is best
(35 m AGL, 0.084 m/px) needs ~25 km of lane — over fifty minutes at survey speed.
A small multirotor on 6S does not have that. This is not a defect to be hidden; it
is the central operational constraint, and it is why the plan includes belief-weighted
lane ordering, plan truncation to the endurance budget, and a two-pass
coarse-then-confirm strategy rather than a single lawnmower over everything.

---

## 2. What already exists, and where the gaps are

### Detection
YOLOv8 / YOLO11 dominate thermal person detection in the 2024–2026 literature.
Published numbers on Jetson Orin NX put YOLOv8n INT8 at ~65 FPS; Qualcomm AI Hub
reports YOLOv8-Det FP16 at ~7 ms on the 8 Gen 2 NPU and YOLOv11-Det QNN FP16 at
~5.4 ms on QCS8550. The RB3 Gen 2 (QCS6490, 12 TOPS) runs 960 px at 37.3 ms.

**The gap:** small-object performance. AP50 for aerial thermal persons is weak —
around 0.28 in the datasets that measure it honestly. A person at 0.25 m/px is a
handful of pixels, and the published FPS figures are measured on frames where the
subject is large. Detection rate is therefore a function of *geometry* — altitude,
lens, target extent — before it is a function of model choice.

### Datasets
For SAR person detection: HERIDAL (68,750 RGB images, best published mAP 95.11%),
SARD, TinyPerson, AFO, SeaDronesSee, NII-CU multispectral, RGBTDronePerson,
A-VTSaR, C2A, AIResQ, WiSARD, ForestPersons.
For hazard/damage: xBD (850,736 building annotations over 19 disasters), RescueNet,
FloodNet, AIDER, D'RespNeT, xFBD.

**The gap:** almost none are multispectral *and* aerial *and* disaster-context
simultaneously. Cross-modal (RGB↔thermal) confirmation has to be trained on
co-registered pairs, which are scarce — this is why the cross-modal veto in this
project is physics-gated rather than learned end-to-end.

### Search planning
Boustrophedon with weighted regions achieves >96% coverage at up to 38% shorter
path length. Ergodic search and multi-UAV grid decomposition are established.
Bayesian belief with greedy re-tasking beats fixed IAMSAR patterns.
Probability of success decomposes as POC × POD = POS.

**The gap:** coverage is usually reported as a boolean "seen / not seen". A cell
imaged at 0.4 m/px through smoke and one imaged at 0.08 m/px in clear air both
count as covered, and the difference between them is whether a person lying there
would have been found.

### GPS-denied navigation
ArduPilot EKF3 supports external-nav source sets (`EK3_SRCn_POSXY/VELXY = 6`),
`VISO_TYPE = 1/3`, `MAV_CMD_SET_EKF_SOURCE_SET` (42007), and optical flow
(`EK3_SRC1_VELXY = 5`). ODOMETRY, VISION_POSITION_ESTIMATE and
VISION_SPEED_ESTIMATE are the relevant messages.

**The gap:** ArduPilot SITL has **no built-in visual-odometry simulator** (there is
no `SIM_VISO`). To exercise `VISO_TYPE=1` you must feed it externally, which is why
this project ships its own feeder rather than relying on the simulator.

### Companion compute
Qualcomm RB5 5G (QRB5165, 15 TOPS, PX4 + ROS2 + TFLite), VOXL 2 (16 g AI autopilot
with integrated flight controller), RB5 Gen 2 dev kit (QCS8550, 48 TOPS).

### Ground control
DroneKit is deprecated and unmaintained; the community recommends pymavlink for
ArduPilot or MAVSDK. **This project therefore exposes a DroneKit-compatible
`connect()` / `simple_takeoff()` / `goto()` layer over pymavlink** rather than
depending on DroneKit — the API familiarity without the dead dependency.

---

## 3. Innovation pillars

Eighteen, grouped by the requirement they serve. Each is implemented and testable,
not aspirational.

**Detection quality**
1. **Two-pass detect-then-confirm.** A coarse pass finds candidates; a low, slow
   confirmation pass resolves posture and movement. `PerceptionPipeline.confirmation_queue`
   and `gsd_confirm_threshold` implement the handoff.
2. **Physics-gated cross-modal ensemble.** RGB confirms thermal only when the
   radiometry says it should be able to — the veto is conditional, which is what
   makes it work at night instead of suppressing every night detection.
3. **Geometry-aware aperture and extent veto.** Detections whose projected ground
   extent is inconsistent with a human body at that range are rejected before
   classification.
4. **Energy-conserving radiometric renderer.** The simulator's thermal channel
   conserves energy, so sub-pixel radiometry experiments mean something
   (`scripts/experiment_subpixel_radiometry.py`).
5. **Bi-directional thermal anomaly detection.** Expected-hot-found-cold
   (a body in cold water) is treated as a hazard cue, not only the reverse.
6. **Per-class kinematic priors.** Motion gates differ for people, vehicles and
   animals, which is what separates a survivor from a distractor at low GSD.

**Navigation under denial**
7. **EKF3 source-set management with drift-inflated geo-tagging.** The source set
   switches on denial and refuses to switch to an unhealthy destination; the
   geo-tag sigma grows with time since last absolute fix.
8. **Online boresight self-calibration.** Mount misalignment is estimated in
   flight rather than assumed from the build.
9. **Drop-to-Confirm payload and LoRa relay.** A marker payload with a radio gives
   a survivor a physical, self-locating beacon independent of the aircraft's link.

**Reporting and command**
10. **Offline-first priority-tier store-and-forward.** See `sar/comms/link.py`.
11. **Zero-internet GCS.** The dashboard runs entirely on the local network;
    nothing in the chain requires a cloud round trip.
12. **Context-aware hazard severity.** Severity is a function of proximity to
    detected survivors, not an intrinsic property of the hazard.
13. **Physiology-driven triage clock.** Time-critical estimates come from a thermal
    physiology model (cold-water immersion, exposure) rather than a fixed priority
    table.
14. **Asymmetric single-modality confidence penalties.** A thermal-only detection
    is penalised differently from an RGB-only one, because their failure modes
    differ.

**Airframe and integration**
15. **TBS Lucid H743 + ELRS + analog FPV with MSP DisplayPort OSD.** AI overlays go
    onto the analog video the pilot is already watching, via `SERIALx_PROTOCOL=42`
    text OSD — no digital link required.
16. **Fine-GSD escape hatch with span waiver.** When detection needs it, the plan
    can drop altitude and explicitly waive the coverage-span requirement rather
    than silently trading one against the other.
17. **Belief-weighted lane ordering with endurance truncation.** What gets cut when
    the battery says stop is the ground least likely to hold anyone.
18. **Capability-weighted coverage rather than boolean coverage.** The grid stores
    cumulative P(detect) per cell, composed as `1 − Π(1 − pᵢ)`.

---

## 4. Architecture

```
                 ┌─────────────────────────────────────────────────┐
                 │  AIRFRAME  (TBS Lucid H743 Wing, 6S, ELRS)      │
                 │                                                 │
   RGB cam ─────►│  ┌───────────┐   ┌──────────────────────────┐   │
   LWIR cam ─────►│  │ companion │   │  ArduPilot  (flight ctrl)│   │
                 │  │ computer  │◄─►│  EKF3 source sets        │   │
   VIO/flow ─────►│  │ RB3 Gen2  │MAV│  GUIDED / RTL / LAND     │   │
                 │  │ or VOXL 2 │Link│                          │   │
                 │  └─────┬─────┘   └──────────┬───────────────┘   │
                 │        │                    │                   │
                 │   perception            servo latch             │
                 │   nav quality           (payload drop)          │
                 └────────┼────────────────────┼───────────────────┘
                          │                    │
              LoRa 900 ───┤              ELRS ─┤ (control + short status)
                          │                    │
                 ┌────────▼────────────────────▼───────────────────┐
                 │  GROUND  (zero internet)                        │
                 │  store-and-forward queue ─► dashboard           │
                 │  belief map, coverage, survivor list, hazards   │
                 └─────────────────────────────────────────────────┘
```

### Code map

| Path | Contents |
|---|---|
| `sar/core/` | geo conversions, frames, clock, events, config |
| `sar/vehicle/` | 6-DoF quadrotor plant, powertrain, autopilot, sensors (GPS, baro, VIO) |
| `sar/mavlink/` | MAVLink protocol constants, `MavConnection`, DroneKit-compatible shim |
| `sar/nav/` | external-nav feeder, VIO sources, nav quality monitor, EKF source-set manager |
| `sar/perception/` | detector ensemble, thermal model, cross-modal fusion, tracker, hazard map, pixel geo-tagger, pipeline |
| `sar/sim/` | world model, energy-conserving renderer, scenarios, **MiniSITL** vehicle simulator |
| `sar/decision/` | boustrophedon planner, capability-weighted coverage grid, Bayesian belief map |
| `sar/mission/` | mission runner: flies the plan, perceives, reports |
| `sar/comms/` | priority-tier store-and-forward link model |
| `sar/gcs/` | command-centre dashboard |
| `configs/` | ArduPilot parameter sets for SITL and for the TBS Lucid hardware |
| `scripts/` | detector evaluation, sub-pixel radiometry, SITL flight test, mission runner |
| `tests/` | behaviour tests, written against failure modes rather than internals |

---

## 5. Phases and distributed work packages

Each work package (WP) is sized for one person over one to two weeks, has a named
deliverable, and an acceptance test that can be run by someone other than its
author. Dependencies are explicit. WPs within a phase that share no dependency are
marked **∥** and can proceed in parallel.

### Phase 0 — Foundations  *(complete)*
Establish the interfaces everything else builds against, so parallel work does not
have to wait on a decision it cannot see.

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 0.1 | `sar/core` geo + frames | round-trip WGS84↔NED within 1 cm over 10 km | ✅ |
| 0.2 | `sar/vehicle` plant + powertrain | hover power matches mass/thrust budget; climb 0→30 m at 3.5 m/s | ✅ |
| 0.3 | `sar/mavlink` connection | shared-inbox reader survives concurrent pumps; ACK not lost | ✅ |
| 0.4 ∥ | `sar/sim/world` + renderer | energy-conserving thermal channel; sub-pixel radiometry reproduces measured contrast | ✅ |
| 0.5 ∥ | scenario presets | `build_reference_scenario(name, seed)` reproduces a sortie exactly | ✅ |

### Phase 1 — Flight stack and simulation  *(complete)*
Get a vehicle that flies over MAVLink, so perception and planning have something to
sit on. **No heavy 3D graphics** — library-based simulation only.

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 1.1 | ArduPilot SITL/HITL parameter sets | `configs/ardupilot_sitl.parm`, `configs/ardupilot_hardware.parm` for `TBS_LUCID_H7_WING` | ✅ |
| 1.2 | **MiniSITL** — pure-Python MAVLink vehicle wrapping `sar/vehicle` | wire-compatible with ArduPilot SITL on tcp:5760; arms, climbs, holds velocity, RTLs, lands, auto-disarms | ✅ |
| 1.3 | DroneKit-compatible layer | `connect()` / `simple_takeoff()` / `goto()` over pymavlink | ⬜ |
| 1.4 ∥ | Flight test harness | `scripts/sitl_flight_test.py` — climb, velocity, denial, restore, payload drop, RTL | ✅ |

**Why MiniSITL exists.** The ArduPilot SITL binary is a build artifact and cannot
always be present. MiniSITL is a pure-Python vehicle that speaks the same MAVLink
dialect on the same port, so `MavConnection`, the nav layer and the mission runner
are byte-identical against either. It publishes only the *estimate* — truth is
never put on the wire — which is what makes geotag error measurable.

### Phase 2 — GPS-denied navigation  *(complete)*

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 2.1 | External-nav feeder (ODOMETRY / VISION_POSITION_ESTIMATE) | 30 Hz, correct LOCAL_FRD(20)/BODY_FRD(12) frames, velocity in body frame | ✅ |
| 2.2 ∥ | VIO sources: simulated and telemetry-echo | `SimulatedVioSource` drifts realistically with texture; `TelemetryVioSource` keeps VisOdom healthy for arming | ✅ |
| 2.3 | Nav quality monitor + EKF source-set manager | switches set 1→2 on denial, refuses an unhealthy destination, returns on restore | ✅ |
| 2.4 | EKF emulator with honest variance | denial → CONST_POS_MODE + growing sigma; dead reckoning diverges ~t² | ✅ |
| 2.5 | Drift characterisation | during denial, reported sigma **bounds** true error | ✅ |

**Measured:** with `SimulatedVioSource` through a 15 s denial, true error grows to
7.2 m while reported sigma grows to 13.5 m. The uncertainty is conservative — which
is the property a rescue team depends on. With `TelemetryVioSource` the reported
sigma stays optimistic while error grows, because the source echoes the diverging
estimate back; that is a documented limitation, not a bug, and is why the denial
test uses the simulated source.

### Phase 3 — Perception  *(complete)*

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 3.1 | Detector ensemble (LWIR + RGB) | calibrated recall vs GSD; `recall_of_resolvable` reported separately from raw recall | ✅ |
| 3.2 ∥ | Thermal physiology model | triage clock driven by exposure, not a fixed table | ✅ |
| 3.3 ∥ | Cross-modal fuser with radiometric gating | night preset does not veto thermal-only detections | ✅ |
| 3.4 | Tracker with per-class kinematic priors | stable IDs across a lane; distractors not promoted | ✅ |
| 3.5 | Hazard map, bi-directional | downed powerline, fire, flood water, collapsed structure | ✅ |
| 3.6 | Pixel geo-tagger with uncertainty propagation | ellipse scales with nav sigma and off-nadir angle | ✅ |
| 3.7 | Detector evaluation harness | `scripts/eval_detector.py` aimed + survey modes → `artifacts/detector_aimed.json` | ✅ |

**Measured (aimed mode, ensemble):**

| Altitude | GSD m/px | resolvable / 36 | recall of resolvable | decoy FA | background FA/frame |
|---|---|---|---|---|---|
| 35 m | 0.084 | 12 | **1.00** | 3 | 0.017 |
| 50 m | 0.120 | 10 | **1.00** | 3 | 0.033 |
| 70 m | 0.168 | 9 | **1.00** | 5 | 0.117 |

The detector finds every survivor it can resolve, at any altitude. What altitude
changes is how many are resolvable at all, and how much junk appears beside them —
background false alarms rise **7×** from 35 m to 70 m. This table is why the survey
altitude ladder starts low and climbs only when endurance forces it.

### Phase 4 — Search planning  *(complete)*

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 4.1 | Boustrophedon planner | lane spacing from swath and side overlap; ground speed from perception rate and along-track footprint | ✅ |
| 4.2 ∥ | Capability-weighted coverage grid | cumulative P(detect) per cell, composed `1 − Π(1 − pᵢ)` | ✅ |
| 4.3 ∥ | Bayesian belief map | negative observation multiplies by `(1 − p_detect)`; priors from terrain/inundation/roads/damage, not from victim positions | ✅ |
| 4.4 | Belief-weighted lane ordering + endurance truncation | what gets cut is the lowest-belief ground | ✅ |

### Phase 5 — Mission execution and communications  *(complete)*

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 5.1 | Mission runner | flies by MAVLink command only; renders from truth, geo-tags from estimate | ✅ |
| 5.2 | Priority-tier store-and-forward link | a full queue sheds bulk and never sheds a survivor | ✅ |
| 5.3 ∥ | Coalescing and duplicate suppression | one survivor is one alert plus material updates, not one per frame | ✅ |
| 5.4 ∥ | Transport characterisation (ELRS / LoRa / Wi-Fi) | alert fits the narrowest alert-capable link | ✅ |
| 5.5 | Payload drop (servo latch / gripper) | `DO_SET_SERVO` on channel 9, recorded with position | ✅ |

### Phase 6 — Command centre  *(in progress)*

| WP | Deliverable | Acceptance | Status |
|---|---|---|---|
| 6.1 | Zero-internet dashboard | live map, survivor list, coverage, link status; no external calls | 🔶 |
| 6.2 ∥ | Survivor card with uncertainty | ellipse drawn to reported sigma, revision history, triage clock | 🔶 |
| 6.3 ∥ | Replay from a mission report artifact | a sortie can be reviewed after the fact | ⬜ |

### Phase 7 — Hardware integration  *(not started)*

| WP | Deliverable | Acceptance |
|---|---|---|
| 7.1 | Airframe: carbon tube frame, 3D-printed mounts, BLDC + 4-in-1 ESC on 6S | hover throttle < 55%, ≥ 15 min endurance at survey weight |
| 7.2 ∥ | TBS Lucid H743 flash + parameter load | `TBS_LUCID_H7_WING` hwdef; SERIAL2 GPS, SERIAL4 Telem1, SERIAL6 ELRS, SERIAL7 Telem2 |
| 7.3 ∥ | ELRS bind + failsafe | RadioMaster Pocket bound; failsafe → RTL; RSSI_TYPE=3 |
| 7.4 | Battery monitoring | BATT_MONITOR=4, VOLT_PIN=10, CURR_PIN=11, VOLT_MULT=11.0, AMP_PERVLT=40.0 |
| 7.5 | Analog FPV + MSP DisplayPort OSD | SERIALx_PROTOCOL=42, OSD_TYPE=5; AI overlay visible in goggles |
| 7.6 ∥ | Companion computer + cameras | RB3 Gen 2 or VOXL 2; RGB + LWIR boresighted |
| 7.7 | Servo latch payload drop | SERVO9_FUNCTION=59, GRIP_ENABLE=1, GRIP_TYPE=0, GRIP_GRAB=1100, GRIP_RELEASE=1900 |
| 7.8 | GPS + denial field test | 30+ satellite fix; denial under canopy/urban; measured drift vs reported sigma |
| 7.9 | End-to-end field sortie | full Phase 5 loop outdoors with the dashboard on a laptop |

### Phase 8 — Evaluation and handover

| WP | Deliverable | Acceptance |
|---|---|---|
| 8.1 ∥ | Monte Carlo robustness sweep | `stress` preset: rain, smoke, night, denial — degradation characterised, not hidden |
| 8.2 ∥ | Ablations | each pillar's contribution measured against the baseline |
| 8.3 | Deployment runbook | flash, bind, calibrate, fly, recover — by someone who did not write it |
| 8.4 | Final demonstration | live sortie, live dashboard, unplugged network |

---

## 6. Suggested team split

Sized for six people; the roles collapse gracefully to fewer.

| Role | Phases | Notes |
|---|---|---|
| **Flight stack / GNC** | 1, 2, 7.2–7.4 | ArduPilot parameters, EKF source sets, failsafes |
| **Perception / ML** | 3 | detector, fusion, tracker, hazard; owns the calibration artifacts |
| **Planning / autonomy** | 4 | coverage, belief, plan generation and truncation |
| **Systems / integration** | 5, 7.6–7.7 | mission runner, comms, payload, companion computer |
| **Ground / UI** | 6, 8.3 | dashboard, replay, runbook |
| **Airframe / electrical** | 7.1, 7.5, 7.8 | frame, wiring, VTX/OSD, field testing |
| **Evaluation** *(shared)* | 8 | owns the test harnesses so nobody marks their own homework |

The evaluation role is deliberately separate. Every acceptance test in this plan is
runnable by someone other than the WP author, because a number produced by the
person who needs it to be true is not evidence.

---

## 7. Simulation strategy

**Library-based, no heavy graphics.** The requirement is algorithm testing —
navigation, control loops, comms protocols — so the simulation is headless and
numeric.

Two interchangeable vehicle backends, same wire protocol:

| Backend | When | Notes |
|---|---|---|
| **ArduPilot SITL** | available | the reference; `configs/ardupilot_sitl.parm` |
| **MiniSITL** (`sar/sim/sitl.py`) | always | pure Python, no binary needed, wraps `sar/vehicle` |

MiniSITL is not a stub. It reproduces the behaviours the rest of the system depends
on: EKF source-set switching with refusal of unhealthy destinations, denial flags
(`CONST_POS_MODE`, `PRED_POS_HORIZ_REL`), ArduPilot's prearm strings, dimensionless
EKF variance ratios, and the ODOMETRY frame requirements (`LOCAL_FRD`/`BODY_FRD`,
velocity in body frame). It publishes only the estimate; truth is reachable from the
simulator object but never goes on the wire.

Rendering is a numpy energy-conserving model, not a graphics pipeline: two 320×240
frames (LWIR + RGB) plus the full perception stack costs ~200–320 ms, so a 2 Hz
perception rate is sustainable in real time alongside a 200 Hz physics loop.

### Reproducing a sortie

```bash
python scripts/run_mission.py --scenario flood --seed 7 --area 220 --duration 520
python scripts/run_mission.py --scenario earthquake --deny-gps 90 --deny-for 45
python scripts/run_mission.py --transport elrs_telemetry     # deliberately bad link
python scripts/run_mission.py --scenario flood_night         # cross-modal veto test
python -m pytest tests/ -q
```

Scenario + seed fully determines the world; the report is written to
`artifacts/mission_<scenario>_seed<seed>.json` so two runs can be diffed numerically.

---

## 8. Hardware mapping

Every part on hand is used, and each maps to a specific work package.

| Available | Used for | WP |
|---|---|---|
| BLDC motors, 4-in-1 ESC | quadrotor propulsion, 6S | 7.1 |
| TBS Lucid H743 Wing | ArduPilot flight controller (`TBS_LUCID_H7_WING`; STM32H743VIH6 480 MHz, 3–12S, 13 PWM, 7 UART) | 7.2 |
| ELRS RX + RadioMaster Pocket | control link and MAVLink telemetry uplink (modelled as `elrs_telemetry`: 2.4 kbit/s, 64 B payload, **control only**) | 7.3, 5.4 |
| GPS module (30+ sats) | primary position source; denial tested by masking | 7.8 |
| Analog camera + VTX + goggles | pilot view with MSP DisplayPort AI overlay | 7.5 |
| 6S LiPo + charger | endurance budget that drives plan truncation | 7.1, 4.4 |
| Servo motors | payload latch (`SERVO9_FUNCTION=59`, `DO_SET_SERVO`) | 7.7, 5.5 |
| Carbon rods/tubes, 3D printer | frame and sensor mounts, boresighting | 7.1, 7.6 |

**To acquire:** an LWIR module (the requirement is RGB + thermal), a companion
computer (Qualcomm RB3 Gen 2 or VOXL 2 — matching the problem statement's
organisation), and a LoRa data link. The LoRa is not optional: it is the minimum
alert-capable link, and without it survivor reports have no path off the aircraft
beyond ELRS control range.

**No built-in compass on the TBS Lucid H7 Wing** — an external magnetometer is
required, or EKF3 must run without one. This is noted because it changes the
parameter set and is easy to discover only at the field test.

---

## 9. Validation strategy

Three layers, because each catches a different class of failure.

1. **Component tests** (`tests/`) — written against failure modes, not internals.
   The comms tests assert what the ground station ends up knowing after a bad link.
2. **Calibration artifacts** (`artifacts/`) — detector recall vs GSD, sub-pixel
   radiometry, drift characterisation. These are the numbers the design decisions
   cite, so they are regenerated rather than quoted from memory.
3. **End-to-end sorties** — a full mission flown over MAVLink with a scored report
   comparing what reached the ground against the world's ground truth.

### Failure modes this project has already paid for

Recorded because each one was invisible in the test that preceded it, and each
would have shipped.

- **Tilt-sign inversion.** NED/ZYX: north needs *negative* pitch, east *positive*
  roll. Inverted, the tilt fights the demand and the result reads as a runaway, not
  as a sign error.
- **Missing NED→body rotation on the tilt demand.** The demand is in NED but thrust
  lies along the body axis. At yaw 0 the rotation is the identity, so every
  north-only test passes; at yaw 180° it is a sign flip on both axes and the
  aircraft accelerates away from its own target to the tilt limit. This produced a
  survey that flew 9.3 km on a 600 m plan.
- **MAVLink `YAW_IGNORE` not honoured.** Re-pointing the nose at the direction of
  travel on every velocity message turns each serpentine lane boundary into a 180°
  yaw slew at full speed.
- **Centimetre parameters read as metres.** `RTL_ALT=2500` is 25 m. Read as metres,
  RTL climbs to 2.5 km at a perfectly normal rate with a perfectly normal
  controller, and the only symptom is that it never completes.
- **`x or default` on an object with `__len__`.** An empty queue is falsy, so the
  caller's queue was silently replaced by a private one — a system that appeared to
  transmit nothing while working perfectly.
- **A batch budget that cannot be exceeded.** A 395-byte alert on a link granting 75
  bytes per window can never be sent; the queue wedges permanently at full capacity.
- **A wire format that does not fit the radio.** A readable, complete survivor alert
  serialised to 409 B against a 222 B LoRa MTU. Invisible in any demo with Wi-Fi.
- **Scoring against pre-compaction field names.** Reported 0% recall on a sortie
  that had delivered eight survivors — the worst kind of test failure, because the
  number looks plausible.
- **A belief floor set as an absolute constant.** 0.002 per cell over 22,500 cells
  is 45 expected survivors against a prior of 12, so the map started believing there
  were four times more people than the scenario contained and progress could never
  move.
- **EKF variance fields read as m².** ArduPilot publishes *dimensionless ratios*
  where ~1.0 is the fail threshold. Reading them as variance leaves sigma pinned at
  its worst-case fallback forever.
- **A filter that never propagates between measurements.** Invisible while a 30 Hz
  source is healthy; the moment a measurement goes stale the estimate freezes while
  the aircraft keeps flying, and sigma stays small because nothing told the filter
  it had stopped being corrected.

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Endurance too short to cover the operational area at useful GSD | **Certain** | High | Belief-weighted ordering, truncation, two-pass coarse-then-confirm (WP 4.4, 16) |
| Thermal small-object recall below operational need | High | High | Altitude ladder from measured calibration; confirmation pass; span waiver |
| GPS-denied drift exceeds geo-tag honesty | Medium | High | Sigma grows with time-since-fix; denial bounded by mission rule; WP 2.5 measures it |
| LoRa duty cycle starves reporting on a rich sortie | Medium | Medium | Coalescing, sent-fingerprint suppression, priority tiers; measured air time per sortie |
| Companion computer unavailable at demo time | Medium | Medium | MiniSITL + laptop-hosted perception reproduces the full stack |
| No external magnetometer on the Lucid H7 | **Certain** | Low | Parameter set runs EKF3 without compass; noted before the field test |
| Analog OSD overlay insufficient for pilot situational awareness | Medium | Low | MSP DisplayPort is text-only by design; dashboard carries the rich picture |

---

## 11. Definition of done

The project is done when a person who did not write it can:

1. flash the flight controller and load `configs/ardupilot_hardware.parm`,
2. arm, and fly a belief-weighted survey of a named area without touching the sticks,
3. watch survivors appear on a dashboard on a laptop with **no internet connection**,
4. unplug the data radio mid-sortie and see the queue hold, then reconnect and see
   the held survivors arrive in priority order,
5. deny GPS mid-sortie and see every reported coordinate's error ellipse widen
   honestly — verified afterwards against ground truth, not asserted,
6. read a sortie report that states recall, precision, geotag error, coverage of the
   searched box, energy, and what was dropped and why.

Item 5 and the "verified against ground truth" clause in item 6 are the ones that
distinguish this from a demonstration.
