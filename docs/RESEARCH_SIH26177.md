# Clean-sheet research for SIH 2026 · Problem Statement SIH26177

**"A deployable AI-powered autonomous drone that aids search-and-rescue
operations by detecting people and hazards, thereby improving responder
safety and reducing victim discovery time."**
— Qualcomm Inc · Robotics & Drones · Hardware

This document starts from zero: it re-reads the problem statement word by
word, surveys what exists, kills every weak option with evidence, and ends
with the single architecture this repository builds. If you read nothing
else, read §1 (what is actually being asked) and §9 (what we build and why).

Companion: [`tech.md`](../tech.md) teaches the same conclusions in easy
words with pictures. This file is the engineering reasoning behind them.

---

## 1. Reading the problem statement like a jury

The official text (SIH portal, via the SIH26177 listing) asks for eight
things. Each is quoted, then translated into a measurable requirement —
because "autonomous navigation" is a wish, while "holds a 32 m lane at
8 m/s with sigma < 10 m" is something you can build and test.

| # | What the PS says | What it measurably means | How we prove it |
|---|---|---|---|
| 1 | Autonomous navigation, GPS-enabled **and GPS-denied**, with AI/SLAM/obstacle avoidance | Fly a planned survey with GNSS; on denial, keep navigating on VIO + LiDAR with a **reported sigma that bounds true error** | Denial sorties in sim + field; sigma-vs-truth plot in `artifacts/` |
| 2 | **On-device AI** inference, no cloud dependence | Every detection runs on the aircraft (phone NPU); zero bytes to the cloud in the detect path | `SAR_DETECTOR=neural` on-device benchmark; airplane-mode flight |
| 3 | **Multi-sensor fusion**: RGB + thermal + IMU + GPS | One survivor hypothesis from ≥2 sensors, with cross-modal agreement scored — not two independent detectors stapled together | Cross-modal agreement bonus measured in `eval_detector.py` |
| 4 | **Hazard classification**: fire, flood water, debris, unstable structures, powerlines, landslides, chemical leaks | ≥8 hazard classes on the map, each with a severity that depends on **proximity to survivors** | Hazard map in every sortie report |
| 5 | **Geo-tagged mapping**: survivor locations, hazard zones, safe routes | Every pin carries lat/lon **+ a 95% error ellipse**; ground-team route avoids hazard cells | A* route + ellipse rendering on the dashboard |
| 6 | **Emergency alerting** with prioritised rescue recommendations | Triage tiers (who first, why) pushed as ≤200 B alerts that fit the worst radio | Alert latency measured on `lora_900` and ELRS-backup profiles |
| 7 | **Offline resilience**, optional 5G/Wi-Fi | Full mission with the data link down; store-and-forward delivers in priority order when it returns | Unplug-the-radio test (definition of done #4) |
| 8 | **Command-centre dashboard** | Laptop dashboard, **zero internet**, live map + survivor cards + link status + replay | `run_mission.py --live` with Wi-Fi off |

Three words in the title carry the whole judging rubric:

- **Deployable** — not a demo. Survives a dead link, a dead GPS, a dead
  sensor, and still brings the aircraft home. (Hence: latching safety
  supervisor, energy-to-return, preflight gate.)
- **Autonomous** — the aircraft decides where to look next without a pilot
  on sticks. (Hence: belief map + coverage grid + planner, not waypoints
  drawn by hand.)
- **Reducing victim discovery time** — the metric is *time-to-first-find*,
  not mAP. A 0.95 mAP model that covers 1 km²/hour loses to a 0.80 model
  covering 4 km²/hour. (Hence: two-pass survey-then-confirm, belief-ordered
  lanes, endurance truncation.)

### Who faces this problem today (the jury's first question)

One real person: the **NDRF flood-relief team leader** in Assam/Bihar during
monsoon. Today she stands on a bund with binoculars, sends boats down
channels she cannot see into, and finds stranded families by shouting.
What exists for her: consumer camera drones (DJI) that need a pilot, a
screen, daylight, and luck — no thermal, no autonomy, no map, no triage.
What she needs: launch in minutes, get back a map with pins, error bars,
and "boat 2 goes here first". That gap — *pilot-dependent toy vs.
autonomous teammate* — is the product.

---

## 2. Prior art: what exists, and why it is not the answer

| System | What it is | Why it fails this PS |
|---|---|---|
| DJI M30T / M350 + H20T | Thermal + RGB, great hardware | **No autonomy, no offline triage, no GPS-denied nav.** It is a flying camera with a pilot. Also ~₹15–30 lakh — not deployable per-district. |
| Skydio X10 | Autonomy + obstacle avoidance | US-only supply, closed stack, no thermal+RGB fusion API for SAR triage, no LoRa/offline-first reporting. |
| ideaForge / Aarav drones | Indian survey drones | Survey/mapping mission, not SAR: no person detection, no triage, no denial handling. |
| Academic YOLO-thermal papers | mAP 0.5–0.6 on HIT-UAV/AIResQ | Benchmarks, not systems: no geotagging, no denial, no comms, no endurance model. A detector is ~15% of this PS. |
| PX4 avoidance + RTAB-Map demos | SLAM + avoidance videos | No victim detection, no multi-sensor fusion, no reporting chain. |
| Phone + drone hobby builds | FPV + phone screen | No autonomy at all. |

**The gap is the integration, not any single component.** Nobody ships:
thermal+RGB *fused* detection → honest geotag → triage → store-and-forward
→ dashboard → ground route, on cheap hardware, offline, with GPS-denied
fallbacks. That whole chain is this project.

---

## 3. Sensor trade study

### 3.1 RGB vs thermal vs LiDAR vs multispectral

| Sensor | Sees people at night/smoke? | Gives identity/shape? | Cost/weight | Verdict |
|---|---|---|---|---|
| RGB (phone camera, 12–50 MP) | No | **Yes** — clothing, posture, hazards | ₹0 (already on phone), 0 g extra | **Confirming channel + VIO + mapping** |
| LWIR thermal 160×120 (Lepton 3.5 class) | **Yes** | Blob only (47 px at 50 m) | ~₹25k, ~1 g core | **Primary detection channel** — the only sensor that sees a body in water at 2 AM |
| LiDAR 1D (TF-Luna, 8 m) | N/A (ranging) | Altitude AGL truth | ~₹3k, 5 g | **Yes — AGL truth + landing + low-alt obstacle stop** |
| LiDAR 2D/3D scanning | N/A | Obstacle cloud | ₹15k–2L, 100–500 g | Optional: 2D for avoidance if budget allows; not needed for detection |
| Multispectral/NDVI | No better than RGB for bodies | No | Expensive | **Rejected** — helps crops, not casualties |
| mmWave radar | Through smoke, coarse | No shape | ₹8k+, complex | Rejected for now — LiDAR + thermal covers the envelope cheaper |

**Decision: phone RGB + LWIR thermal + 1D LiDAR rangefinder (+ optional 2D
scan).** Thermal finds candidates day/night/smoke; RGB confirms identity and
feeds VIO; LiDAR gives true height above ground (baro drifts, GPS altitude
is ±5 m fiction) and stops the aircraft flying into a wire/building on a
low confirmation pass.

### 3.2 Why thermal is primary (the physics in one paragraph)

A living body is 30–36 °C at the skin. Flood water at night is 12–18 °C.
That 15+ K contrast exists in total darkness, in smoke, under thin
vegetation — everywhere RGB sees nothing. The catch is **sub-pixel
radiometry**: a thermal pixel reports the area-weighted average of
everything inside it, so a half-pixel person at 70 m reads ~20 °C, not
33 °C, and the human band stops discriminating. Consequence: **search wide
and high for candidates, descend to 22 m to confirm** — the two-pass
strategy is forced by physics, not chosen by taste.
(`scripts/experiment_subpixel_radiometry.py` measures the collapse curve.)

---

## 4. Compute trade study: why a phone is the onboard computer

The PS sponsor is Qualcomm and the obvious answer is an RB3 Gen 2 (QCS6490,
12 TOPS). We compared honestly:

| Option | AI compute | Camera | Display | 4G/5G | Battery | Price | Verdict |
|---|---|---|---|---|---|---|---|
| **Android phone (Snapdragon 8 Gen 2 class)** | Hexagon + tensor accel, INT8 TFLite @ 30+ fps on 640px YOLO-nano | 12–50 MP, OIS, excellent | Built-in (field UI free) | Built-in modem | Built-in UPS (flies home on its own power) | ₹0 — team already owns it | **Winner: flies now, costs nothing** |
| Qualcomm RB3 Gen 2 / VOXL 2 | 12 TOPS Hexagon DSP, best NPU | Needs separate sensor + tuning | None | Needs modem | Needs regulation | ₹60k–1.2L + import delay | Best *final* board; blocked on procurement |
| Jetson Orin Nano | 40 TOPS, CUDA | Needs sensor | None | None | Hungry (7–15 W) | ~₹55k | Great but heavy, hot, power-hungry for this airframe |
| Raspberry Pi 5 + Hailo/AI HAT | ~13–26 TOPS w/ HAT | Needs sensor | None | None | OK | ~₹25k with HAT | Credible fallback; worse camera + no modem |
| Laptop-on-a-string (Wi-Fi offload) | Infinite | No | — | — | — | — | **Rejected**: violates on-device requirement, dies with the link |

The phone is not a compromise — it is strictly better on **five** axes that
matter for deployment: it has the best RGB camera in the price universe, a
screen for field debugging, a modem for optional 4G relay, its own battery
(autopilot brown-outs don't kill the AI), and every NDRF unit already owns
several. The Hexagon DSP inside that very phone is Qualcomm silicon, so the
sponsor's technology story stays intact: TFLite + NNAPI delegates run on the
same Hexagon architecture as the RB3 path.

**Decision: phone-first, RB3-compatible.** All models ship as TFLite-INT8
(phone NPU via NNAPI) *and* ONNX (workstation/RB3 path). One detector
interface, two runtimes. When the RB3 arrives, only the runtime string
changes.

---

## 5. Navigation trade study: surviving GPS denial

| Approach | Absolute truth? | Drift | Needs | Verdict |
|---|---|---|---|---|
| GPS-only | Yes | None (when available) | Open sky | Baseline; fails exactly where disasters happen (urban canyon, valley, jamming) |
| VIO (phone camera + IMU) | No — relative | ~1–2% of distance | Features + light + compute | **Primary denial fallback.** Phone already has the camera + IMU + CPU. OpenVINS/VINS-Fusion class or ARCore-pose bridge. |
| 2D LiDAR SLAM | Locally absolute | Low indoors/short range | Scanning LiDAR | Complementary at low altitude; not primary (range, weight) |
| Optical flow + rangefinder | No | High | Flow sensor or phone cam | Tertiary: velocity hold for short transits (EKF set 3) |
| Baro + compass heading hold | No | Quadratic | Nothing | Last resort: fly straight, grow sigma honestly |

**Decision: EKF3 source-set ladder 1→2→3** (already in
`configs/ardupilot_hardware.parm`): GNSS → VIO-from-phone (ODOMETRY @ 30 Hz
over USB) → optical-flow velocity hold. The non-negotiable property:
**reported sigma must bound true error** (measured 13.5 m vs 7.2 m over
15 s denial). A pin with an honest 14 m ellipse sends a boat to the right
channel; a pin with a lying 2 m ellipse sends it to the wrong village.

---

## 6. Model trade study: what detects a 47-pixel person

Measured evidence (2025 aerial-thermal meta-study, 75k images, Jetson AGX
Orin — see `docs/06_AI_MODELS_AND_DATASETS.md`):

| Model family | AP small objects | Why |
|---|---|---|
| RT-DETR-L (transformer) | **5.0%** | Needs texture + big data; thermal people are smooth warm ellipses in small datasets |
| YOLOv8/9/10/11 (CNN one-stage) | **10.7–11.6%** | Convolutional priors match the blob regime; P2 stride-4 head keeps tiny-object resolution |
| Heuristic (physics-gated, this repo) | recall-of-resolvable **1.00**, precision ~17–40% | Perfect recall of what physics allows; clutter is the enemy, not blindness |

Realistic ceiling: **mAP50 ≈ 0.55** on AIResQ. Anyone quoting 0.95 is
solving ground-level RGB, not aerial thermal.

**Decision (the "best model" for this PS): a physics-gated neural-heuristic
ensemble, not a bare network** — `sar/ai/flood.py::FloodDetector`:

1. Thermal triage in kelvin (human band on the *peak* pixel, not the mean).
2. Geometry-aware aperture: expected body size from live GSD; reject roofs,
   vehicles, specks by extent ratio.
3. **Flood-context veto (new):** water mask from RGB + LiDAR-flatness prior;
   sun-glint and roof detections inside water/sky context are down-weighted,
   in-water candidates get a confirmation descent instead of a penalty.
4. Neural head: YOLO11n-nano TFLite-INT8 on the phone NPU (P2 head,
   1-channel thermal stem) — kills roof geometry the heuristic can't.
5. **Temporal consistency (new):** a candidate must persist ≥3 frames with
   consistent geo-position *and* plausible thermal decay; static clutter that
   never moves *and* never cools is the discriminator.
6. Calibrated confidence (temperature scaling) so fusion maths and the
   operator's trust are both valid.

Why the ensemble beats either half alone is kill-critiqued in
[`docs/MODEL_CRITIQUE.md`](MODEL_CRITIQUE.md). The one-line version: the
network rejects *geometry* (roofs, vehicles), the physics rejects
*impossibility* (wrong size, wrong temperature, wrong context), and only
together do they reach operator-tolerable precision without losing recall.

---

## 7. Comms trade study: the alert must fit the worst radio

| Link | Rate | Range | Carries a 200 B alert? | Verdict |
|---|---|---|---|---|
| ELRS 2.4 GHz telemetry | ~2.4 kbit/s, 64 B packets | ~800 m–2 km | **No** — control link; 64 B MTU can't take an alert | Control + tiny status only. Modelled honestly as such. |
| LoRa 900 MHz | ~5–20 kbit/s, 222 B MTU | 2–8 km | **Yes** — alert designed to 200 B | **Primary alert link.** Not optional hardware. |
| Phone 4G/5G (where towers survive) | Mbit/s | Tower-bound | Yes + thumbnails | Opportunistic bulk path; never assumed |
| Wi-Fi bridge | Mbit/s | ~200 m | Yes | Bench + demo + dashboard hop |

**Decision: store-and-forward with priority tiers** (CRITICAL alerts →
HIGH assessments → NORMAL health/coverage → LOW imagery). A full queue
sheds imagery, never survivors; alerts retry until ACKed with dedup keys.
This is the difference between "works at the demo on Wi-Fi" and "works in
a flooded district".

---

## 8. Simulation trade study: AirSim vs Gazebo vs headless

| Simulator | Imagery | Physics/sensors | Flood scene cost | Runs here? | Verdict |
|---|---|---|---|---|---|
| **AirSim (Cosys-AirSim fork, UE5)** | Photoreal RGB + depth + segmentation; thermal via segmentation-proxy or custom shader | Good multirotor; LiDAR, IMU, GPS, magnetometer APIs | Medium: flood town from UE assets + water plane | Needs GPU + UE5 build (not in this sandbox) | **Primary visual sim** — `sar/sim/airsim_bridge.py` + `worlds/flood_town/` |
| **Gazebo Harmonic (gz-sim)** | PBR, camera + depth + thermal-camera plugin + GPU LiDAR | Best open physics (DART), ArduPilot SITL-native via MAVLink | Low-medium: SDF flood town, water plane plugin | Needs gz install (not in sandbox) | **Primary physics/HITL sim** — `sar/sim/gazebo_bridge.py` + SDF world |
| Headless MiniSITL + numpy renderer (existing) | Radiometric thermal in kelvin, schematic RGB | 200 Hz plant, EKF semantics | Zero — always runs | **Yes** | **Algorithm truth**: fast, deterministic, CI-safe. Never replaced, only complemented. |

Key insight: **no single simulator proves everything.** AirSim proves the
detector on realistic pixels; Gazebo proves the vehicle + avoidance +
ArduPilot integration; the headless stack proves search maths, comms, and
triage deterministically in CI. So the flood runner
(`scripts/run_flood_sim.py --backend auto`) tries AirSim → Gazebo →
headless-cinematic in that order, and **the headless fallback renders a
cinematic flood town (water reflections, houses, victims, debris) so the
"model works" demonstration runs anywhere**, including this sandbox and a
jury laptop with no GPU.

Thermal in AirSim deserves one honest sentence: stock AirSim has no LWIR
sensor; we derive a pseudo-thermal frame from the segmentation + depth
cameras with the same radiometric model as the headless renderer
(per-class temperature + distance attenuation + NETD noise). It exercises
the detector's temperature logic on realistic geometry; absolute kelvin
truth still comes from the headless radiometric path and, finally, the
real Lepton.

---

## 9. The decision: what we build

```
AIR:  TBS Lucid H743 (ArduPilot) + phone (RGB + NPU AI + VIO + 4G)
      + LWIR thermal + TF-Luna LiDAR + GPS + ELRS + LoRa + analog FPV w/ OSD
                    │ USB-OTG MAVLink + video
GROUND: laptop dashboard (offline) + RadioMaster Pocket (pilot override)
```

| PS requirement | Our answer | Where it lives |
|---|---|---|
| GPS + GPS-denied nav | EKF source-set ladder; phone VIO @30 Hz; LiDAR AGL; honest sigma | `sar/nav/`, `sar/hardware/{mobile,lidar}.py`, `configs/ardupilot_hardware.parm` |
| On-device AI | TFLite-INT8 YOLO-nano on phone Hexagon + heuristic ensemble | `sar/ai/flood.py`, `scripts/export_phone_model.py` |
| RGB+thermal+IMU+GPS fusion | Cross-modal fuser with radiometric gating + temporal consistency | `sar/perception/fusion.py`, `sar/ai/flood.py` |
| Hazard classification | 12 classes + survivor-proximity severity | `sar/perception/{detector,hazard}.py` |
| Geo-tagged mapping | Pins + 95% ellipses + A* ground routes | `sar/perception/geotag.py`, `sar/rescue/routing.py` |
| Emergency alerting | Triage tiers + 200 B alerts + payload drops | `sar/perception/human_id.py`, `sar/comms/link.py`, `sar/rescue/` |
| Offline resilience | Priority store-and-forward; phone modem is bonus, never assumed | `sar/comms/link.py` |
| Dashboard | Zero-internet FastAPI dashboard + replay | `sar/gcs/dashboard.py` |
| Simulation proof | AirSim ⇄ Gazebo ⇄ headless-cinematic flood town | `sar/sim/{airsim_bridge,gazebo_bridge,flood_scene,flood_render}.py` |

### What was killed, explicitly

- Transformer detectors (half the small-object AP). 
- Cloud/edge-offload inference (violates requirement 2 and dies with the link).
- RGB-only detection (blind at night — half the golden hours).
- GPS-only navigation (fails in the exact terrain of the PS).
- ELRS-as-data-link (64 B MTU cannot carry a 200 B alert — physics, not opinion).
- Demo-Wi-Fi comms (invisible failure until field day).
- Single-simulator proof (each sim lies about something; the trio covers all lies).
- Buying the RB3 before flying (the phone flies *today*; the RB3 is an upgrade path, not a gate).

---

## 10. Risks, remaining

| Risk | Status after this plan |
|---|---|
| No trained thermal weights yet | Pipeline + export + quant gate done; needs GPU-days on HIT-UAV/AIResQ — tracked, not blocked (heuristic flies meanwhile) |
| No real VIO front end yet | Consumer side done + tested; phone VIO bridge specified in `docs/MOBILE_COMPANION.md`; needs field validation |
| AirSim/Gazebo not installed here | Headless-cinematic fallback proves the loop today; UE/SDF worlds ship for lab machines |
| Precision still the weak metric | Flood-context veto + temporal consistency target it directly; measured in `eval_flood_model.py` |

*Next: [`tech.md`](../tech.md) for the illustrated easy-words version, or
[`docs/MODEL_CRITIQUE.md`](MODEL_CRITIQUE.md) for the adversarial review of
the detector.*
