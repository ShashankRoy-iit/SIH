# 05 · Search theory: how the aircraft decides where to look

Reference for `sar/decision/coverage.py` — `BoustrophedonPlanner`, `SurveyPlan`,
`CoverageGrid`, `BeliefMap`, `PriorSource` — and for the planning half of
`sar/mission/runner.py`.

The one-line thesis: **a search is not a geometry problem, it is an allocation
problem.** Endurance, not area, is the scarce resource, so every decision here
is "what is the next joule worth".

---

## 1. Three quantities, deliberately kept separate

| Quantity | Question it answers | Whose question | Object |
|---|---|---|---|
| **Coverage** | "Have we searched it, and how well?" | The operator's | `CoverageGrid` |
| **Belief** | "Where is someone still likely to be?" | The analyst's | `BeliefMap` |
| **Plan** | "What do we fly next?" | The aircraft's | `SurveyPlan` |

Merging them is the classic mistake. Coverage is a statement about the
*aircraft's sensors*; belief is a statement about the *world*. A cell can be
fully covered and still hold high belief (searched badly, through smoke, at
120 m) — and that is exactly the cell you must revisit.

---

## 2. Coverage is a probability, not a tick-box

`CoverageGrid` stores, per cell, the cumulative probability that a survivor
present there **would have been detected**:

```
P_detected(cell) = 1 − Π_i (1 − p_i)
```

`p_i` is the detection probability of look *i*, derived from that frame's actual
ground sample distance, range, viewing angle and conditions — not from the fact
that the cell fell inside a camera frustum.

Consequences that fall out of this and matter operationally:

* A **second pass at better GSD is worth more than repeating the first**, and
  the arithmetic shows it, rather than a boolean grid saying "already done".
* A pass through smoke at 100 m raises `P_detected` by very little, so the
  planner keeps the cell on the list. The operator sees a pale cell, not a
  green one.
* The grid also keeps `n_looks`, `best_gsd_m` and `last_look_s` per cell, so
  a report can state *how* an area was searched, which is what a handover to a
  ground team actually needs.

**Coverage is always reported twice** — against the search box, and against the
whole area of operations:

```
effective coverage  11.7%   <- of the whole 900 x 900 m basin
of the search box  100.0%   <- the sortie did what it planned
```

Quoting either alone is misleading in opposite directions.

---

## 3. Belief: expected survivors per cell, not a probability distribution

`BeliefMap` holds **the expected number of survivors that are present and not
yet detected** per cell. That choice is deliberate:

* A normalised probability distribution says *"the survivor is somewhere"*.
  Finding them collapses the map and it stops being useful.
* An expected count says *"we believe about nine people are in this basin, and
  here is where"*. Finding three reduces the remaining expectation by three.
  The map stays composable with detections, which is what a multi-survivor,
  multi-sortie operation requires.

### Priors

`PriorSource(name, field, weight, description)` folds in spatial knowledge:
building edges and rooftops, road lines, water edges, high ground, last-known
positions, mobile-phone pings. They are combined **multiplicatively over a
floor**:

```python
belief *= (0.25 + 0.75 * clip(field, 0, 1) ** weight)
belief *= expected_survivors / belief.sum()      # renormalise to the count
belief  = maximum(belief, floor_per_cell)
```

The `0.25 +` and the floor are not cosmetic. A prior built from a damage map is
wrong *somewhere*, and a cell driven to exactly zero becomes permanently
unsearchable — the map would confidently hide a survivor. The floor is scaled to
**2% of the uniform prior**, small enough not to outweigh the evidence and large
enough that nothing is written off. (An earlier absolute constant of 0.002 per
cell summed to 45 expected survivors over a 900 m basin against a true prior of
12 — the map believed in four times more people than existed and search progress
could never move. That bug is in the code comments as a warning.)

### Updates

```python
belief.observe(north, east, p_detect, detected)
```

* **Negative observation** — a look that found nothing multiplies by
  `(1 − p_detect)`. A poor look barely reduces belief. This is the coupling
  between the renderer's physics and the search strategy, and it is the whole
  reason coverage is stored as probability.
* **Positive observation** — a detection multiplies the cell by `0.15` rather
  than zeroing it, because group size is an estimate and a second person in the
  same cell may have been occluded in that frame.

`expected_remaining()` is the number an operator should be shown next to
"survivors found": *the map still thinks there are 5.2 people out there*.

---

## 4. The plan: everything is derived, nothing is chosen

`BoustrophedonPlanner.plan(...)` produces serpentine lanes. Three numbers that
projects usually hard-code are computed here.

### 4.1 Lane spacing from the swath

```
spacing = swath_width_m × (1 − side_overlap)          side_overlap = 0.30 default
```

At 35 m AGL with a 57° HFOV the thermal swath is ≈ 38 m, so lanes go at ≈ 27–32 m.
Every point is imaged at least once and lane boundaries twice, which is what
makes a detection near a lane edge recoverable from the adjacent pass while the
aircraft is rolling.

### 4.2 Along-track speed from the perception rate

```
v ≤ perception_hz × along_track_footprint × (1 − forward_overlap)
```

At 2 Hz with a 25 m along-track footprint and 60% forward overlap, that is 20
m/s — clipped to the airframe's 8 m/s. Fly faster than this bound and the
coverage grid records cells as *seen* between two frames that never looked at
them. Coverage would then be a lie in the direction that kills people.

### 4.3 Altitude from the required GSD

Altitude is bounded by `[min_altitude_agl_m, max_altitude_agl_m]` (25–120 m by
default; the safety supervisor holds the hard limits) and, when unspecified, is
derived from the swath the plan needs. In practice the sortie uses the two-pass
pair: **45 m survey, 22 m confirmation** (`configs/onboard.yaml`).

### 4.4 Lane *ordering* is where belief enters

The lanes themselves never move — a lawnmower with varying spacing is not a
complete search. What changes is the **order**:

```python
lane.weight = belief.mass_at(points_along_lane)
lanes.sort(key=descending_weight, tie_break=low_transit_serpentine)
```

This is a real trade, not a free optimisation: flying high-belief lanes first
makes the transit legs cross low-belief ground, so the total path is slightly
longer. It is worth it because **the binding constraint is endurance**. A sortie
that covers the most promising 60% before it must return home finds more people
than one that covers 100% of the least promising ground in index order.

### 4.5 Transit time is counted

`SurveyPlan.total_duration_s` adds the inter-lane transits. Ignoring them
understates a many-short-lane sortie by 10–20%, and that error lands directly on
the endurance estimate — the number that decides whether the aircraft comes
home.

---

## 5. The confirmation pass

```python
plan2 = planner.refine(plan, north, east, radius_m=60, altitude_agl_m=22)
```

A survivor detected at survey altitude gets a second look from **lower and
slower**: four short lanes over a 120 m box at roughly half the altitude and 60%
of the speed. `refine()` returns a *new* plan instead of mutating the survey, so
the survey is resumable afterwards.

Why this exists at all is radiometry, not tidiness. A thermal pixel reports the
area-weighted average of its footprint, so below about one pixel of target the
apparent temperature collapses toward the background and the human thermal band
stops discriminating. `scripts/experiment_subpixel_radiometry.py` measures the
curve. Search wide and high for *candidates*; descend to *decide*.

---

## 6. Endurance is the real constraint

The 900 × 900 m reference basin at 32 m lane spacing needs about **25 km of
lane**. At 8 m/s that is 52 minutes of flying before transit, climb, or the
confirmation passes — beyond one battery. This is not a defect in the planner;
it is the physics of the problem, and it drives three things:

1. **Belief-ordered lanes** (§4.4) — cover the best ground first.
2. **Reachability-based abort** — the safety supervisor computes the energy to
   climb, cruise home and land, applies a **1.35 margin**, and compares it with
   the energy remaining. A fixed 20% floor is meaningless at 800 m out and
   wasteful at 50 m out.
3. **Multi-sortie / multi-aircraft handoff** — the coverage grid and belief map
   are serialisable precisely so a second battery or a second aircraft resumes
   rather than restarts. *Designed, not yet implemented — see `STATUS.md`.*

---

## 7. What a search progress number should mean

```python
belief.expected_remaining()      # survivors the map still expects to find
grid.p_detected.mean()           # mean detection capability achieved
```

Search progress is reported as **belief mass removed**, not as area swept. The
distinction matters at the end of a sortie: sweeping the last 20% of low-prior
ground moves the area number a lot and the belief number almost not at all,
which is precisely when the operator should be told the next battery is better
spent elsewhere.

---

## 8. Reproducing this

```bash
python3 scripts/run_mission.py --area 220 --duration 460 --live
make assets           # regenerates docs/assets/02_search_coverage.gif from this code
```

The coverage animation in [`teach.md`](../teach.md#3-search-coverage-is-a-probability)
is driven by the real `CoverageGrid`; the panel on the right is the trade this
document describes — coverage rising linearly, survivors found in steps, and the
battery running out first.
