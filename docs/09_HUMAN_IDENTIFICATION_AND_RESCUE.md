# Autonomous Human Identification, Precision Air-Drop & Rescue Routing

**SIH Problem Statement ID 26177** · Qualcomm Inc. (Robotics & Drones) · Theme: Autonomous SAR

This document details the theory, mathematical formulation, architecture, and validation of the autonomous human identification, precision ballistic air-drop targeting, safe evacuation routing, and interactive visualization subsystems of the SAHYOG drone stack.

---

## 1. Operational Overview

In large-scale disaster scenarios (flash floods, earthquakes, wildfires, landslides), rapid search and survival aid delivery are critical:
1. **Human Identification**: Autonomous multi-spectral identification that goes beyond a bounding box by classifying victim posture, active distress/SOS signals, estimated physiological state (hypothermia/heat stress), demographics, and immediate survival needs.
2. **Precision Air-Drop Delivery**: Autonomous calculation and execution of aerodynamic ballistic drop solutions to deploy targeted emergency payloads (lifebuoys, trauma kits, thermal blankets, locator beacons, water rations) with sub-15m accuracy in wind.
3. **Safe Evacuation Corridors**: Multi-modal $A^*$ pathfinding that computes safe access routes for first responders (foot teams, amphibious rescue boats, 4x4 vehicles, UGVs) while avoiding deep flood currents, steep rubble, fire fronts, and collapse zones.
4. **Interactive Tactical Command**: Real-time 60 FPS animated tactical dashboard with dual-spectrum synthetic sensors, rotating drone kinematics, waving survivors, parachute drops, locator beacons, and evacuation routes.

---

## 2. Multi-Spectral Human Identification Architecture

Located in `sar/perception/human_id.py`, the `HumanIdentifier` processes confirmed geographic tracks from the perception pipeline (`PerceptionPipeline`).

```
  RGB + Thermal Observations
            │
            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Multi-Attribute Human Identification Engine                 │
 ├──────────────────────────────────────────────────────────────┤
 │  1. Posture Classification (Standing, Lying, Waving, Water)  │
 │  2. SOS Wave Motion Spectral Analysis (0.8 - 3.5 Hz)         │
 │  3. Demographic & Group Size Estimation                      │
 │  4. Chromatic & Saliency Clothing Analysis                   │
 │  5. Thermal Physiology & Core Temp Proxy                     │
 │  6. Decoy Rejection (Hot Engines, Fast Vehicles, Rocks)      │
 │  7. Triage Priority & Rescue Equipment Mapping               │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                       `HumanProfile` Record
```

### 2.1 Posture Classification & Geometry
Human postures are classified into seven discrete operational categories:
- `STANDING`: Standard vertical projected footprint ($A \approx 0.35 - 0.75\,\text{m}^2$, aspect ratio $\approx 1.0 - 1.3$).
- `WAVING_SOS`: Dynamic upper-body periodic aspect ratio oscillation ($\Delta \text{aspect} > 0.35$, $f \in [0.6, 4.2]\,\text{Hz}$).
- `SITTING`: Compact stationary footprint, immobility $> 0.85$, elongation $< 1.35$.
- `LYING`: Prone/supine stationary footprint, immobility $> 0.88$, elongation $> 1.5$.
- `PRONE_PARTIAL_BURIAL`: Immobile survivor in rubble/collapse context with high elongation $> 1.6$.
- `IN_WATER_CLINGING`: Survivor immersed in water $> 0.5\,\text{m}$ depth.
- `WALKING`: Continuous ground speed $> 0.6\,\text{m/s}$.

### 2.2 Dynamic SOS Waving Gesture Detection
Survivors in distress frequently signal passing aerial drones by waving arms overhead. The system tracks the time series of bounding box aspect ratios $\alpha(t) = \frac{w(t)}{h(t)}$ and projected area $A(t)$:
$$\Delta \alpha = \max(\alpha) - \min(\alpha)$$
$$f_{\text{wave}} = \frac{N_{\text{zero\_crossings}}}{2 \cdot \Delta t}$$
When $\Delta \alpha > 0.35$ and $0.6\,\text{Hz} \le f_{\text{wave}} \le 4.2\,\text{Hz}$, the posture is elevated to `WAVING_SOS` and the distress level is tagged `CRITICAL_ACTIVE_SOS`.

### 2.3 Physiological Vitals & Core Temperature Proxy
Core temperature $T_{\text{core}}$ is estimated through environmental exposure modelling:
- **In-Water Immersion**: Convective heat loss pulls core temperature downward over time:
  $$T_{\text{core}}(t) = T_{\text{base}} - \dot{Q}_{\text{loss}} \cdot t_{\text{immersion}} \cdot \left(\frac{20 - T_{\text{water}}}{10}\right)$$
- **Cold Exposure on Land**: Peripheral shutdown model:
  $$T_{\text{core}} = \max\left(28.0,\, 35.5 - 0.4 \cdot (25.0 - T_{\text{apparent}})\right)$$
- **Heat Stress Risk**: Elevated surface radiometry ($T_{\text{apparent}} > 36.5^\circ\text{C}$):
  $$T_{\text{core}} = \min\left(41.5,\, 37.5 + 0.6 \cdot (T_{\text{apparent}} - 36.5)\right)$$

### 2.4 Biological Decoy Discrimination
To avoid deploying critical relief kits to non-human targets, four physics checks reject decoys:
1. **Biothermal Upper Bound**: $T_{\text{apparent}} > 48^\circ\text{C} \implies$ non-biological (generator, vehicle engine, solar reflection).
2. **Biothermal Lower Bound**: $T_{\text{apparent}} < 15^\circ\text{C}$ with ambient $> 22^\circ\text{C} \implies$ cold wet inanimate debris.
3. **Kinematic Velocity Bound**: $v > 6.5\,\text{m/s}$ in disaster terrain $\implies$ moving vehicle or fast animal.
4. **Context Extent Ratio**: Hot region background area $> 7\times$ human body $\implies$ industrial thermal source.

---

## 3. Aerodynamic Ballistic Air-Drop Simulator

Located in `sar/rescue/payload.py`, the ballistic calculator models the full descent trajectory of emergency relief payloads.

### 3.1 Equations of Motion
The state vector $\vec{x} = [r_N, r_E, r_D]^T$ and $\vec{v} = [v_N, v_E, v_D]^T$ are integrated with:
$$\frac{d\vec{r}}{dt} = \vec{v}$$
$$m \frac{d\vec{v}}{dt} = m \vec{g} - \frac{1}{2} \rho(z) C_d A \|\vec{v} - \vec{w}(z)\| (\vec{v} - \vec{w}(z))$$

Where:
- $\vec{g} = [0, 0, 9.80665]^T\,\text{m/s}^2$ (Downwards).
- $\rho(z) = \rho_0 e^{-z / 8500}$ (Barometric density lapse).
- $\vec{w}(z)$ is the 3D altitude-dependent wind vector field.
- $C_d A$ switches from $C_d A_{\text{freefall}}$ to $C_d A_{\text{canopy}}$ at $t = t_{\text{release}} + \tau_{\text{parachute\_delay}}$.

### 3.2 Calibrated Payload Specifications

| Payload Type | Mass ($kg$) | $C_d A_{\text{freefall}}$ ($m^2$) | $C_d A_{\text{canopy}}$ ($m^2$) | Chute Delay ($s$) | Target Survivor Condition |
|---|---|---|---|---|---|
| `FLOTATION_BUOY` | 0.85 | 0.035 | 0.85 | 0.40 | In-water immersion |
| `FIRST_AID_TRAUMA_KIT` | 0.65 | 0.025 | 0.75 | 0.35 | Trauma, lacerations, crush |
| `THERMAL_EMERGENCY_BLANKET`| 0.35 | 0.015 | 0.60 | 0.30 | Hypothermia risk |
| `LORA_LOCATOR_BEACON` | 0.25 | 0.012 | 0.50 | 0.25 | Trapped under rubble |
| `WATER_RATIONS_PACK` | 0.90 | 0.030 | 0.80 | 0.40 | Stranded / heat stress |
| `EXTRICATION_MARKER` | 0.40 | 0.020 | 0.65 | 0.30 | Extrication assistance |

### 3.3 Inverse Ballistic Targeting Lead Solution
Given the aircraft's current ground speed $V_g$, altitude $h_{\text{AGL}}$, and approach heading $\psi$, the simulator computes the horizontal lead offset $\Delta \vec{r}_{\text{lead}}$:
$$\Delta \vec{r}_{\text{lead}} = \vec{r}_{\text{impact}} - \vec{r}_{\text{release}}$$
$$\vec{r}_{\text{release\_target}} = \vec{r}_{\text{survivor}} - \Delta \vec{r}_{\text{lead}}$$

This ensures the payload impacts within $\le 15\,\text{m}$ of the target survivor.

---

## 4. Multi-Modal Safe Access Transit Routing ($A^*$)

Located in `sar/rescue/routing.py`, the routing subsystem plans obstacle-avoidant access paths for response teams.

### 4.1 Modality Constraints

| Team Modality | Nominal Speed ($m/s$) | Max Wading Depth ($m$) | Min Water Depth ($m$) | Max Slope (deg) | Rubble Passable |
|---|---|---|---|---|---|
| `FOOT_RESCUE_TEAM` | 1.2 | 0.50 | 0.00 | 28.0° | Yes |
| `AMPHIBIOUS_RESCUE_BOAT` | 4.5 | $\infty$ | 0.25 | 4.0° | No |
| `OFFROAD_4X4_VEHICLE` | 7.0 | 0.40 | 0.00 | 18.0° | No |
| `ALL_TERRAIN_UGV` | 2.2 | 0.60 | 0.00 | 35.0° | Yes |

### 4.2 Cost Function
The graph search evaluates movement cost along edges $(u, v)$:
$$C(u, v) = d(u, v) \cdot \left[ \frac{1}{\mu_{\text{speed}}(v)} + 4.0 \cdot S_{\text{hazard}}(v) \right]$$
Where $S_{\text{hazard}}(v)$ incorporates fire severity, smoke density, and flood current velocities.

---

## 5. Tactical Command Centre Dashboard & Animation Engine

Located in `sar/gcs/dashboard.py` and `scripts/animate_rescue_mission.py`:

- **60 FPS Animated Canvas**:
  - Rotating quadrotor propellers synced to vehicle ground speed.
  - Active scanning FOV spotlight cone with sweeping radar beam.
  - Animated SOS waving arms for distressed survivors.
  - Parachute deployment and descent trajectories.
  - Expanding omnidirectional LoRa radar locator pulses.
  - Animated emergency rescue squads and boats moving along safe access corridors.
- **Dual-Spectrum Synthetic Feeds**:
  - LWIR Thermal Ironbow false-color stream with temperature crosshairs.
  - Visible RGB camera feed with AI detection bounding boxes and posture tags.
- **Interactive Control Deck**:
  - Live triage metrics (Immediate, Delayed, Minor, Airdrops).
  - One-click **"Deploy Precision Air-Drop"** and **"Dispatch Rescue Squad"** actions.
- **Zero-Internet Self-Contained Deployment**:
  - Zero external CDN dependencies — runs on air-gapped disaster laptops over local Wi-Fi or Ethernet.

---

## 6. Verification and Acceptance Tests

Automated test suites in `tests/`:
1. `tests/test_human_id.py`: Validates standing, waving SOS, water clinging, prone burial, group clustering, core temp, and decoy rejection.
2. `tests/test_rescue.py`: Validates aerodynamic ballistic drops, inverse targeting lead calculation, A* routing, and coordinator lifecycle.
3. `tests/test_animation.py`: Validates mission animation data generation, standalone HTML5 player building, and Matplotlib GIF export.
4. `tests/test_comms.py`: Validates store-and-forward link budgets under radio dropouts.
