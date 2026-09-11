# Kill-critique: the detector, adversarially reviewed

This document tries to **kill** every detector candidate for the flood
search-and-rescue task — including our own — and keeps only what survives
with evidence. Read it as: *claim → attack → verdict*. The survivor is
`FloodDetector` (`sar/ai/flood.py`), and §7 lists exactly what changed in
the code because of this review.

Test everything here with:

```bash
python3 scripts/eval_flood_model.py --scenes 40 --altitudes 35,50,70
python3 scripts/eval_detector.py --mode both   # baseline cross-check
```

---

## 1. Candidate A — bare YOLO (RGB, COCO weights)

**Claim:** "YOLO finds people, ship it."

**Attack:**
- COCO people are standing pedestrians at eye level. Our people are
  20–100 px warm blobs, often lying, often chest-deep in water. Domain gap
  is total: published aerial-thermal mAP50 for YOLO ≈ 0.55 *after*
  fine-tuning on thermal data, far lower zero-shot.
- RGB is blind at night, in smoke, in rain. Half of all golden hours are
  dark. A detector that sleeps at night is not a SAR detector.
- COCO YOLO happily fires on 4 m warm roof panels — it has no notion of
  ground sample distance, so "person-sized" is unknowable to it.

**Verdict: KILLED.** Useful only as the RGB *confirming* channel after
aerial fine-tuning (VisDrone/HERIDAL), never as the primary.

## 2. Candidate B — detection transformer (RT-DETR / RF-DETR)

**Claim:** "Transformers are state of the art, use the best."

**Attack:** the 2025 aerial-thermal meta-study (75k images, Jetson AGX
Orin) measured AP-small: RT-DETR-L **5.0%** vs YOLO family **10.7–11.6%**.
Less than half. Causes are structural: self-attention needs texture (a
thermal person is a smooth ellipse) and needs data (public aerial-thermal
sets are ~10⁴ images, not 10⁶). Slower, hungrier, worse — on exactly our
target class.

**Verdict: KILLED.** Revisit in 2–3 years if thermal SAR datasets grow 10×.

## 3. Candidate C — heuristic thermal detector alone (repo baseline)

**Claim:** "Recall of resolvable is 1.00 — we're done."

**Attack:**
- Recall is perfect; **precision is ~17%** on the reference flood sortie.
  Five false alarms per real survivor exhausts an operator in an hour.
  Dominant sources, measured: sun-heated rooftops, vehicles, water-glint
  edges, warm debris.
- The geometry veto (extent ratio ≤ 6.5) catches *big* clutter but not
  roof ridges and car roofs at 50–70 m where the warm patch itself is
  person-sized in pixels.
- Single-frame: a glint that exists for one frame is reported with the
  same confidence as a body seen for ten seconds.

**Verdict: WOUNDED, KEPT AS HALF OF THE ENSEMBLE.** Its recall (1.00 of
resolvable at 35/50/70 m) is the floor we refuse to go below. Everything
below fixes precision *without touching that recall* — verified by the
`recall_of_resolvable == 1.00` assertion in `eval_flood_model.py`.

## 4. Candidate D — thermal-only neural (YOLO11n, 1-channel, P2 head)

**Claim:** "A trained network beats hand rules on clutter."

**Attack (honest — this one is strong):**
- True on *geometry*: learned features reject roof shapes and vehicle
  outlines better than any extent ratio. Expected precision gain: large.
- But: needs GPU-days + dataset access (HIT-UAV/AIResQ approvals) we may
  not have before field day; INT8 quantisation can silently eat small-object
  recall (hence the ≥0.97 recall-retained gate); a network alone still has
  no GSD reasoning, no water context, no temporal memory.
- Failure mode is *silent*: a quantised model that regresses on the day
  looks confident while missing chest-deep children.

**Verdict: KEPT, BUT GATED.** It flies inside the ensemble behind the
physics veto and the quantisation gate — never alone, never uncalibrated.

## 5. Candidate E — RGB + thermal naive fusion ("both must agree")

**Claim:** "Require both sensors to agree; false alarms vanish."

**Attack:** at night RGB sees nothing, so "both must agree" = "detect
nothing at night". In-water survivors show no clothing cue even by day
(only head/shoulders visible). Naive AND-fusion deletes the hardest real
cases along with the clutter. Measured consequence in early code: night
recall collapsed.

**Verdict: KILLED.** Replaced by **conditional fusion**: agreement raises
confidence; disagreement penalises *only when RGB is usable*
(`rgb_frame_quality`), and in-water candidates trigger a confirmation
descent instead of a penalty.

## 6. Candidate F — single-frame scoring (no temporal memory)

**Claim:** "Per-frame confidence is enough."

**Attack:** static clutter (rooftop, rock) is *perfectly* persistent, so
naive "persist ≥ N frames" doesn't discriminate either — it just delays
both. The discriminator must be **movement + thermal behaviour**: a body
cools/warms plausibly and shifts subtly; a roof never moves and tracks the
sun; a glint flickers frame-to-frame. Single-frame scoring cannot see any
of this.

**Verdict: KILLED.** Replaced by temporal consistency with a thermal-decay
term (`TemporalFilter` in `sar/ai/flood.py`): persist ≥3 frames with
consistent geo-position **and** non-flickering contrast; flicker vetoes,
stillness + plausible temperature confirms.

---

## 7. The survivor: `FloodDetector` — what the critique changed in code

| # | Attack that forced it | Code change | Where |
|---|---|---|---|
| 1 | A→ roofs fire the network | Geometry veto *before* the network: projected extent must be human at live GSD | `sar/ai/flood.py::geometry_gate` |
| 2 | C→ 17% precision, roofs/vehicles | Flood-context veto: RGB water mask + LiDAR-flat prior; roof/warm-vehicle context down-weighted, never auto-deleted | `sar/ai/flood.py::flood_context` |
| 3 | C→ water glints | Glint flicker test: single-frame high-chroma specks with no thermal partner are suppressed | `sar/ai/flood.py::glint_test` |
| 4 | E→ night collapse | Conditional fusion: `rgb_quality < 0.30` ⇒ LWIR-only is *expected*, not penalised | `sar/ai/flood.py::conditional_fuse` |
| 5 | F→ static clutter | Temporal filter: ≥3 frames, geo-consistent, thermal-decay plausibility, flicker veto | `sar/ai/flood.py::TemporalFilter` |
| 6 | D→ silent quant regression | Quant gate: INT8 ships only at ≥0.97 recall retained vs FP32; else w8a16 | `scripts/export_phone_model.py --validate` |
| 7 | D→ uncalibrated argmax | Temperature-scaled confidence; fusion uses probabilities, not logits | `sar/ai/calibration.py` + flood stack |
| 8 | All→ "trust me" | Every rejection carries a *reason code*; the sortie report lists what was vetoed and why | `Detection.attributes["veto"]` |

### Operating points (measured, headless-cinematic flood scenes)

| Altitude | GSD | Recall of resolvable | Precision (baseline → flood) | FA/frame (baseline → flood) |
|---|---|---|---|---|
| 35 m | 0.084 | 1.00 | 0.31 → **0.58** | 0.017 → **0.008** |
| 50 m | 0.120 | 1.00 | 0.22 → **0.47** | 0.033 → **0.015** |
| 70 m | 0.168 | 1.00 | 0.11 → **0.29** | 0.117 → **0.052** |

Recall of resolvable stays 1.00 at all altitudes (the floor holds);
precision roughly doubles and false alarms halve. The 70 m row is still
weak — which is why the planner surveys at 35–45 m and only climbs when
endurance forces it, and why confirmation descends to 22 m. Numbers
regenerate with `python3 scripts/eval_flood_model.py`.

### What would kill even this model (stated plainly)

1. **A casualty at ambient temperature** (long cold-water immersion,
   covered by debris): no thermal contrast exists; only RGB shape/movement
   can find them, by day. Mitigation: RGB person channel + movement cue stay
   active even with zero thermal score.
2. **Heavy rain on the lens + night + denial simultaneously**: all three
   sensing modes degraded at once. Mitigation: honest sigma + "sensors
   degraded" flag to the operator beats a confident lie.
3. **Adversarial clutter**: a sun-heated mannequin-shaped object at body
   temperature that moves (flag/branch). No per-frame system solves this;
   the confirmation pass (posture + movement + RGB identity) is the answer,
   and the system says "needs confirmation" rather than guessing.
