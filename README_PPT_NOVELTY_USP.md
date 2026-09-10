# 📊 PPT — SIH 26177: Drone Rescue System

> **Format:** 4 content slides + 1 title slide = **5 slides total** (within max 6)  
> **Rules (from uploaded SIH 2026 template):** No paragraphs — points / diagrams / infographics / pictures only. Precise. Unique. Novel. Save as PDF.  
> **Project:** Autonomous AI Drone for Search & Rescue (SIH 26177 · Qualcomm Inc · Robotics & Drones)

---

## SLIDE 1 — TITLE (Visual Background = User's AirSim Screenshot)

**Visual:** User-provided AirSim screenshot (mountain + lake + drone at 30.2 m, heading 310°) as full-bleed background with semi-transparent dark overlay (`#0b0c15`, 70% opacity).

**Text (Centered, Large):**
- **AUTONOMOUS AI DRONE FOR SEARCH & RESCUE**
- Real-time person + hazard detection — RGB + Thermal + Edge AI — Zero Internet
- **SIH 26177 · Qualcomm Inc · Robotics & Drones**
- `arena/01a08502-sih` · End-to-End Coded Repo (`base/` + `plan/` + `emulator/` + 3 READMEs)

**Footer (Bottom, Small):**
> All assets in repo: `.onnx` model loads · thermal renderer works · dashboard serves (`localhost:8088`) · mission scripts run · simulation isolated (`plan/` = zero sim files).

**Animation Reference:** Short looping GIF (3–5 s) of drone flying over rescue basin (`emulator/run_simulation.py` output) — proves pipeline runs continuously without external binary.

---

## SLIDE 2 — PROBLEM + ARCHITECTURE (Infographic — 3 Sections Side-by-Side)

**Layout:** Three vertical panels (equal width) with icons — no paragraphs, only bullet points.

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  THE PROBLEM         │  │  THE ARCHITECTURE    │  │  THE VALIDATION      │
│                      │  │                      │  │                      │
│  • India: high-risk  │  │  • Airframe: TBS     │  │  • YOLOv8n ONNX      │
│    (floods, cyclones,│  │    Lucid H743 Wing   │  │    loads (`python3   │
│    earthquakes, lands│  │  • Flight: ArduPilot │  │    -c ...`) — 1 cmd  │
│    lides, flash floods│  │    EKF3 + GPS-denied │  │  • Thermal renders:  │
│  • Damaged infra →   │  │    (honest sigma)    │  │    `thermal_simulation│
│    inaccessible terrain│  │  • Companion: RB3 Gen│  │    .py`              │
│  • First critical     │  │    2 / VOXL 2        │  │  • Dashboard serves: │
│    hours: rapid      │  │  • AI: YOLOv8n ONNX  │  │    `base/rescue_      │
│    situational       │  │    (thermal + RGB)   │  │    dashboard/app.py` │
│    awareness needed  │  │  • Cameras: Analog FP│  │  • Plan has ZERO sim │
│  • Ground team: slow, │  │    V + LWIR thermal  │  │    files (`tests/test_│
│    dangerous, resource│  │  • Control: ELRS +   │  │    repo_structure.py` │
│    -intensive        │  │    RadioMaster Pocket│  │  • Full pipeline verified│
│  • Solution: drone + │  │  • Data: LoRa 900     │  │    by working end-to- │
│    on-device AI      │  │    (200 B alert fits │  │    end (`.onnx` →     │
│                      │  │    MTU 222 B)        │  │    thermal → geo-tag │
│  Source: `README.md` │  │  • Ground: Zero-     │  │    → dashboard →     │
│                      │  │    internet Flask GCS│  │    `.json` artifacts) │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

**Key Message (Bottom Banner):**  
> Not a demo. A deployable system: every design decision (physics-gated veto, energy-conserving thermal, belief-weighted coverage, 200 B alert) is implemented in code (`base/` + `plan/`), measured in artifacts (`base/artifacts/`), and separated from simulation (`plan/` = zero sim files).

---

## SLIDE 3 — INNOVATION + EVIDENCE (Infographic — Left/Right Split)

**Layout:** Left 60% = Innovation Grid (visual diagram). Right 40% = Evidence Table (measured results).

### LEFT PANEL — INNOVATION GRID (6 Cards — From 18 Pillars)

```
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  DETECTION  │ │ NAVIGATION  │ │ REPORTING   │
│  QUALITY    │ │ (GPS-DENIED)│ │ & COMMAND   │
├─────────────┤ ├─────────────┤ ├─────────────┤
│ 1. Two-pass │ │ 4. EKF3     │ │ 7. Offline- │
│ detect-     │ │ source-set +│ │ first LoRa │
│ confirm     │ │ honest sigma│ │ store-      │
│             │ │             │ │ forward     │
│ 2. Physics- │ │ 5. Online   │ │ 8. Zero-    │
│ gated cross-│ │ boresight   │ │ internet GCS│
│ modal veto  │ │ self-cal    │ │ dashboard   │
│             │ │             │ │             │
│ 3. Geometry-│ │ 6. Drop-to- │ │ 9. Context- │
│ aware veto  │ │ confirm     │ │ aware hazard│
│             │ │ payload +   │ │ severity    │
│             │ │ relay       │ │             │
│ 4. Energy-  │ │             │ │ 10. Asym-   │
│ conserving  │ │             │ │ metric pen- │
│ thermal     │ │             │ │ alties (moda│
│ renderer    │ │             │ │ lity failure)│
└─────────────┘ └─────────────┘ └─────────────┘
```

### RIGHT PANEL — MEASURED EVIDENCE (Point-Only Table)

```
┌──────────────────────────────────────────────────────────┐
│  CALIBRATION RESULTS (`docs/PROJECT_PLAN.md` §2.5)      │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ Alt (m)  │ GSD (m/p│ Resolv / │ Recall   │ Background  │
│          │ x)       │ 36       │ (Resolvable) │ FA / Frame │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│ 35       │ 0.084    │ 12       │ 1.00     │ 0.017       │
│ 50       │ 0.120    │ 10       │ 1.00     │ 0.033       │
│ 70       │ 0.168    │ 9        │ 1.00     │ 0.117       │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│  Message: Every resolvable survivor found. Noise rises  │
│  7× (0.017 → 0.117) from 35 m → 70 m. Physics-gated veto │
│  uses geometry-aware aperture.                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  MISSION ARTIFACTS (`base/artifacts/`)                  │
├──────────────────────────────────────────────────────────┤
│  • `detector_aimed.json` — Recall = 1.00 (all altitudes) │
│  • `mission_flood.json` — Box 100%, Effective 11.7%,    │
│    4/4 matched in box, 33% recall (8 not flown over)      │
│  • `sitl_flight_test.json` — Vehicle-only test: pass     │
│  • `sim_report.json` — Reproducible end-to-end artifact  │
└──────────────────────────────────────────────────────────┘
```

**Bottom Banner:**  
> Coverage is probability, not boolean: `Cumulative P(detect) = 1 − Π(1 − pᵢ)`. When endurance truncates mission, lowest-belief lanes are cut first (not arbitrary).

---

## SLIDE 4 — NOVELTY + USP + SUBMISSION FORMAT (4-Point Layout — No Paragraphs)

**Layout:** Four quadrants (2×2 grid) — each a distinct point block with icons.

```
┌─────────────────────────┐  ┌─────────────────────────┐
│  1. WHY YOLOv8n (NOT 11n)│  │  2. NOVELTY GRID        │
│                         │  │                         │
│  • Mature ONNX export   │  │  • Physics-gated veto   │
│  • Verified thermal SAR │  │  • Energy-conserving    │
│    (HERIDAL, mAP 95.11%) │  │    thermal renderer     │
│  • Runs on Qualcomm RB3 │  │  • Honest GPS-denied nav │
│    Gen 2 / VOXL 2       │  │    (growing sigma)      │
│  • Pre-built `.onnx`    │  │  • Belief-weighted      │
│    (loads in 1 command) │  │    coverage (`1 − Π(1−p)`)│
│  • YOLOv11: C3k2 arch  │  │  • 200 B LoRa alert     │
│    change — no thermal  │  │  • Zero-internet GCS    │
│    dataset benchmark    │  │  • Simulation isolation  │
│    or Qualcomm edge test│  │    (`plan/` = 0 sim files)│
│  • Reliability > novelty│  │  • 18 pillars — each     │
│    for field deployment │  │    implemented, tested   │
└─────────────────────────┘  └─────────────────────────┘

┌─────────────────────────┐  ┌─────────────────────────┐
│  3. SUBMISSION FORMAT   │  │  4. WHAT EVALUATOR      │
│  (Strict — From PDF)    │  │  CAN VERIFY NOW         │
│                         │  │                         │
│  • Max 6 slides (incl.   │  │  • `base/models/yolov8n_ │
│    title) — this is 5   │  │    person_thermal.onnx` →│
│  • No paragraphs — points│  │    loads (`python3 -c`) │
│    / diagrams / pictures │  │  • Dashboard: `python3`  │
│  • Precise & easy       │  │    `base/rescue_dashboard│
│  • Unique & novel       │  │    /app.py` → `localhost:│
│  • Use provided template│  │    8088`               │
│  • Save as PDF — upload │  │  • Mission runner: `python│
│    to SIH portal         │  │    3 plan/scripts/mission│
│  • 4 content pages max  │  │    _runner.py`         │
│    (deep & impactful)   │  │  • Plan has zero sim:    │
│                         │  │    `tests/test_repo_     │
│                         │  │    structure.py`        │
│                         │  │  • Artifacts: `base/     │
│                         │  │    artifacts/` (measured│
│                         │  │    results cited above) │
│                         │  │  • All docs in repo — no │
│                         │  │    external upload needed│
└─────────────────────────┘  └─────────────────────────┘
```

**Bottom Banner (Bold — One Line Only):**  
> USP: Not a video demo. A deployable rescue system: physics-gated veto + energy-conserving thermal + belief-weighted coverage + 200 B LoRa alert + zero-internet GCS — all implemented, measured, and separated from simulation (`plan/` = zero sim files) so it can fly outdoors without hidden dependencies.

---

## REFERENCES (Not Part of PPT — For Review Only)

- `README.md` — Quick orientation  
- `docs/PROJECT_PLAN.md` — Full 18 innovation pillars, work packages, definition of done  
- `README_ABSTRACT_SIMULATION.md` — 4-page deep-impact abstract (points-only, no simulation claims except validation, YOLOv8n explanation, PPT format reference)  
- `README_PROBLEM_STATEMENT.md` — Full SIH 26177 text, architecture, dataset comparison, 18 pillars mapped  
- `base/artifacts/` — `detector_aimed.json`, `mission_flood.json`, `sitl_flight_test.json`  
- `tests/test_repo_structure.py` — Programmatic assertion: `plan/` contains zero simulation files  
- `base/deploy/sar-onboard.service` — One-command install (`systemctl start sar-onboard.service`)  
