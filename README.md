# Autonomous AI Drone for Search & Rescue

**SIH Problem Statement 26177** · Qualcomm Inc · Robotics & Drones

An autonomous search-and-rescue drone that detects people and hazards from RGB +
thermal imagery with on-device edge AI, navigates with GPS *and* through GPS
denial, geo-tags what it finds with honest uncertainty, and reports to a
command-centre dashboard over a radio link that is assumed to be intermittent.

**[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) is the plan** — research review,
innovation pillars, phase-wise work packages sized for distributed work, hardware
mapping, risks, and the definition of done. This file is the orientation.

---

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy pymavlink fastapi uvicorn pyyaml matplotlib scipy pytest

# A complete sortie: fly, search, detect, report to a live dashboard
python scripts/run_mission.py --area 220 --duration 460 --live

# The same sortie with GNSS denied mid-flight for 45 s
python scripts/run_mission.py --area 220 --duration 460 --deny-gps 190 --deny-for 45 --live

# A deliberately bad link, to watch store-and-forward work
python scripts/run_mission.py --transport elrs_telemetry

# Night preset, where the cross-modal veto has to be conditional rather than total
python scripts/run_mission.py --scenario flood_night

# Vehicle-only flight test: climb, velocity, denial, restore, payload drop, RTL
python scripts/sitl_flight_test.py

# Detector calibration and evaluation
python scripts/eval_detector.py --mode both

python -m pytest tests/ -q
```

`--live` serves the command centre on `http://0.0.0.0:8088`. Scenario plus seed
fully determines the world, so any sortie is reproducible and the report in
`artifacts/` can be diffed numerically against a previous one.

---

## What a sortie produces

`artifacts/mission_<scenario>_seed<seed>.json` and a one-screen summary:

```
  flight      armed=True  landed=True  crashed=False  522 s  4272 m  44.3 Wh  batt 77%
  plan        8 lanes, 30 m spacing, 1760 m total, alt 55 m AGL
  perception  482 frames, 241 cycles, 860 ms mean (2304 ms p95)
  COVERAGE
    effective coverage  11.7%   <- of the whole 900 x 900 m basin
    of the search box  100.0%   <- the sortie did what it planned
      box n 259-479 m, e 481-701 m, 4 victims inside it of 12 in the world
  DETECTION vs GROUND TRUTH
    correctly matched  4   recall 33%   precision 17%
    geotag error       mean 29.1 m, worst 54.2 m
  DATA LINK  (lora_900)
    delivered          655 packets / 120339 B   packet success 97.8%
    coalesced          379 updates folded into queued messages
    suppressed         299 no-change updates not sent
    evicted            1009 (bulk shed: 1.3 MB)
```

Read the two coverage numbers together. The sortie searched 40,000 m² of a
810,000 m² basin and covered **100%** of what it set out to search. Recall against
the whole world is 33% because only 4 of the 12 survivors were inside the box —
the other 8 were never flown over. Reporting only the first number would flatter
the system; reporting only the second would condemn it. Both are in the report.

---

## Simulation approach

**Library-based, headless, no 3D graphics** — built for algorithm testing:
navigation, control loops, comms protocols.

Two interchangeable vehicle backends on the same MAVLink wire protocol:

| Backend | Notes |
|---|---|
| **ArduPilot SITL** | the reference, when the binary is available |
| **MiniSITL** (`sar/sim/sitl.py`) | pure Python, always available, wraps `sar/vehicle` |

MiniSITL is not a stub. It reproduces what the rest of the system depends on: EKF
source-set switching with refusal of unhealthy destinations, denial flags
(`CONST_POS_MODE`, `PRED_POS_HORIZ_REL`), ArduPilot's prearm strings, dimensionless
EKF variance ratios, and the ODOMETRY frame requirements (`LOCAL_FRD`/`BODY_FRD`
with body-frame velocity). **It publishes only the estimate — truth never goes on
the wire** — which is what makes geotag error measurable rather than assumed.

Rendering is a numpy energy-conserving model, not a graphics pipeline. Two 320×240
frames plus the whole perception stack costs 200–320 ms, so a 2 Hz perception rate
runs in real time alongside a 200 Hz physics loop.

The mission runner flies by MAVLink command only and never touches the vehicle
plant. A runner that moved the aircraft by assigning to its state would still
produce a convincing log and would tell you nothing about whether guidance, link
budget or prearm checks work.

---

## Layout

| Path | Contents |
|---|---|
| `sar/core/` | geo conversions, frames, clock, events, config |
| `sar/vehicle/` | 6-DoF quadrotor plant, powertrain, autopilot, GPS/baro/VIO sensors |
| `sar/mavlink/` | protocol constants, `MavConnection`, DroneKit-compatible shim |
| `sar/nav/` | external-nav feeder, VIO sources, quality monitor, EKF source-set manager |
| `sar/perception/` | detector ensemble, thermal physiology, cross-modal fusion, tracker, hazard map, pixel geo-tagger, pipeline |
| `sar/sim/` | world model, renderer, scenarios, MiniSITL |
| `sar/decision/` | boustrophedon planner, coverage grid, Bayesian belief map |
| `sar/mission/` | mission runner |
| `sar/comms/` | priority-tier store-and-forward link |
| `sar/gcs/` | zero-internet command-centre dashboard |
| `configs/` | ArduPilot parameter sets for SITL and for the TBS Lucid H743 |
| `scripts/` | mission runner, flight test, detector evaluation, radiometry experiment |
| `tests/` | behaviour tests, written against failure modes rather than internals |
| `artifacts/` | measured results the design decisions cite |
| `docs/` | the project plan |

---

## Three design decisions worth reading before the code

**Coverage is a probability, not a boolean.** The grid stores cumulative
P(detect) per cell, composed as `1 − Π(1 − pᵢ)`, where each pass's `pᵢ` comes from
the ground sample distance it actually achieved. A cell imaged at 0.4 m/px through
smoke and one imaged at 0.08 m/px in clear air both count as "covered" in a boolean
grid, and the difference between them is whether a person lying there would have
been found.

**The alert must fit the narrowest radio.** A complete, readable survivor alert
serialises to 409 bytes. The LoRa MTU is 222. The most important message the system
produces could not be sent over the radio carried precisely for when nothing else
works — and this is invisible in any demonstration with Wi-Fi. The alert is now 200
bytes, with the rich detail as a separate lower-tier message that waits for a wider
link. ELRS is modelled honestly as a *control* link that cannot carry alerts at all.

**Report locations, not track ids.** At 8 m/s with a ~0.9 s perception cycle the
tracker re-acquires identities routinely, and per-track reporting put eight markers
on one person. Reports are merged spatially with a radius scaled to the reported
uncertainty, and bounded so transitive merging cannot walk a site away from its
anchor. 43 tracks became 17 distinct locations.

---

## Known limitations

Stated plainly, because a report that only lists successes is not evidence.

- **Precision is the weak number.** 17% on the reference flood sortie. The real
  detections geotag well — V09 at 8.8 m error, V07 at 23.9 m — but two physical
  objects (building rooftops) generate repeated person detections, and a handful of
  geotags hit the 25 m sigma ceiling. The detector's own calibration shows
  `recall_of_resolvable = 1.00` at 35, 50 and 70 m, so this is not a recall problem;
  it is the false-alarm side, and it is the first thing to fix.
- **Perception is slower than the platform.** 860 ms mean against a 500 ms target
  at 2 Hz, with a 2.3 s p95. The control loop is perception-bound, so lane tracking
  degrades under load even though it holds 0.3 m lateral error at the measured rate.
  The landing-zone search dominates and is already cached; it needs to move off the
  control thread.
- **Endurance bounds coverage hard.** The full basin at the best detection altitude
  needs ~25 km of lane — over fifty minutes. A 6S multirotor does not have that.
  The plan truncates to the budget and flies the highest-belief lanes first, but the
  honest operational answer is a coarse pass followed by low confirmation passes,
  which is planned (WP 16) and not yet built.
- **`TelemetryVioSource` cannot navigate a real denial.** It echoes the estimate
  back to itself, so error grows while the reported sigma stays optimistic. That is
  documented behaviour, not a bug — `SimulatedVioSource`, which measures the plant
  with its own drift model, is what the denial tests use, and with it the reported
  sigma (13.5 m) correctly bounds the true error (7.2 m) over a 15 s denial.

---

## Failure modes this project has already paid for

Each was invisible in the test that preceded it, and each would have shipped.

- **Missing NED→body rotation on the tilt demand.** The demand is in NED; thrust
  lies along the body axis. At yaw 0 the rotation is the identity, so every
  north-only test passed. At yaw 180° it is a sign flip on both axes and the
  aircraft accelerates away from its own target to the tilt limit. A 600 m plan
  flew 9.3 km.
- **`YAW_IGNORE` not honoured.** Re-pointing the nose at the direction of travel on
  every velocity message turns each serpentine lane boundary into a 180° yaw slew
  at full speed.
- **Centimetre parameters read as metres.** `RTL_ALT=2500` is 25 m. Read as metres,
  RTL climbs to 2.5 km at a perfectly normal rate with a perfectly normal
  controller, and the only symptom is that it never completes.
- **`x or default` on an object with `__len__`.** An empty queue is falsy, so the
  caller's queue was silently replaced by a private one — a system that appeared to
  transmit nothing while working perfectly.
- **A batch budget that cannot be exceeded.** A 395-byte alert on a link granting 75
  bytes per window can never be sent; the queue wedges permanently at capacity.
- **Scoring against pre-compaction field names.** Reported 0% recall on a sortie
  that had delivered eight survivors — the worst kind of test failure, because the
  number looks plausible.
- **A belief floor set as an absolute constant.** 0.002 per cell over 22,500 cells
  is 45 expected survivors against a prior of 12, so the map started believing there
  were four times more people than the scenario contained and progress could never
  move.
- **EKF variance fields read as m².** ArduPilot publishes dimensionless *ratios*
  where ~1.0 is the fail threshold. Reading them as variance pins sigma at its
  worst-case fallback forever.
- **A filter that never propagates between measurements.** Invisible at 30 Hz; the
  moment a measurement goes stale the estimate freezes while the aircraft keeps
  flying, and sigma stays small because nothing told the filter it had stopped being
  corrected.
- **`Telemetry.armed` is an `ArmState` dataclass, not a bool.** `not tel.armed` is
  never true, so an RTL loop waiting for disarm simply runs to its timeout — which
  looks like a slow landing rather than a condition that can never be met.

---

## Hardware

The plan uses what is on hand: BLDC motors and 4-in-1 ESC on 6S, **TBS Lucid H743
Wing** (`TBS_LUCID_H7_WING`), **ELRS** with a RadioMaster Pocket, a 30+ satellite
GPS module, analog camera/VTX/goggles with MSP DisplayPort OSD for AI overlay,
servo-driven payload latch, carbon tube frame with 3D-printed mounts.

To acquire: an **LWIR module** (the requirement is RGB *and* thermal), a **companion
computer** (Qualcomm RB3 Gen 2 or VOXL 2), and a **LoRa data link** — the last is
not optional, it is the minimum alert-capable link.

Note: the TBS Lucid H7 Wing has **no built-in compass**, so an external
magnetometer is required or EKF3 must run without one. This changes the parameter
set and is easy to discover only at the field test.

Full mapping in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §8.
