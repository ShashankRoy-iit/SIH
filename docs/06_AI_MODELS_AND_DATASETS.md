# 06 · AI models, datasets, training and deployment

This document answers three questions completely: **which model flies**, **what
it is trained on**, and **how it gets onto the aircraft's silicon** — with the
measured evidence for each choice and the commands that implement it.

It is the reference for `sar/ai/` and for `scripts/{fetch_models,
train_detector,export_model}.py`.

---

## 1. The short answer

| Path | Model | Format on the aircraft | Why |
|---|---|---|---|
| **LWIR / thermal — primary** | YOLO11n, 1-channel stem, extra **P2 (stride-4) head**, 640×512 | INT8 `w8a8` **QNN** `.bin` on the Hexagon DSP (QCS6490); ONNX FP16 elsewhere | Small objects, no texture, small datasets — a CNN one-stage detector wins measurably |
| **RGB — confirming** | YOLO11n, COCO + VisDrone fine-tune, 640×640 | INT8 QNN / ONNX | Off-the-shelf transfers well; used to confirm, not to lead |
| **Always available** | Audited heuristic ensemble (`sar/perception/detector.py`) | pure NumPy | Zero weights, deterministic, ~90 fps on two CPU cores; the fallback that makes the system honest |

One environment variable chooses between them everywhere:

```bash
SAR_DETECTOR=auto|hybrid|heuristic|neural
```

---

## 2. Why YOLO11n and not a detection transformer

The obvious 2025 instinct is "use RT-DETR / RF-DETR, transformers won". For
**aerial thermal person detection** that instinct is measurably wrong.

A 2025 study on a 75,000-image aerial thermal meta-dataset, benchmarked on a
Jetson AGX Orin (MDPI *Journal of Imaging* 11(12):436):

| Model | AP50-95 (all) | **AP50-95 (small objects)** | Precision (small objects) |
|---|---|---|---|
| RT-DETR-L | 36.3% | **5.0%** | 24.4% |
| YOLOv8-L / v9 / v10 | 48.1 – 49.4% | **10.7 – 11.6%** | 36.7% |

On the object class this project exists to detect, the transformer scores less
than half. The reasons are structural, not incidental:

* **Transformers need texture.** A person in LWIR is a smooth warm ellipse. The
  self-attention machinery that pays for itself on richly textured RGB has
  little to attend to.
* **Transformers need data.** Public aerial thermal SAR datasets total tens of
  thousands of images, not millions. DETR-family models are famously
  data-hungry and slow to converge.
* **Our objects are tiny.** At survey altitude a person is 20–100 pixels of
  area. That is precisely the regime in the table above.

**Realistic accuracy expectation.** YOLOv8 / YOLO11 trained on the AIResQ
airborne thermal SAR benchmark reach **mAP50 ≈ 0.55** — the best of all models
tested there; RT-DETR on a comparable person dataset reached 0.31. Treat 0.55 as
the ceiling for this task. Any claim near 0.95 is describing ground-level
imagery, not aerial thermal.

**Speed reference** (Ultralytics, T4 TensorRT / CPU ONNX):

| Model | mAP50-95 (COCO) | CPU ONNX | T4 TensorRT | Params |
|---|---|---|---|---|
| YOLO11n | 39.5 | 56.1 ms | **1.5 ms** | 2.6 M |
| YOLO11s | 47.0 | 90.0 ms | 2.5 ms | 9.4 M |

At 2 Hz perception with two modalities, `n` leaves headroom for the rest of the
onboard loop; `s` is the upgrade path if the DSP proves to have slack in flight
testing.

### The two architecture changes we make

1. **1-channel stem.** Thermal is single-channel. Replicating it to 3 channels
   wastes the first convolution; a 1-channel stem is initialised by summing the
   RGB stem weights.
2. **P2 head (stride 4).** The stock YOLO neck predicts at strides 8/16/32. A
   40-pixel target at stride 8 is 5 cells across; at stride 4 it is 10. This is
   the single largest accuracy change for our object size, and it costs about
   20% more latency — a trade the DSP can afford.

```bash
python3 scripts/train_detector.py --train --p2 --channels 1 --imgsz 640
```

---

## 3. Datasets

```bash
python3 scripts/train_detector.py --list-datasets
```

| Dataset | Modality | Size | Why it is in the mix |
|---|---|---|---|
| **HIT-UAV** | LWIR | 2,898 images | Aerial thermal at 60–130 m AGL, person/car/bicycle — the closest public match to our sensor geometry |
| **AIResQ** | LWIR | — | High-resolution airborne thermal SAR; the source of the realistic mAP50 ≈ 0.55 expectation |
| **SARD** | RGB | 1,981 images | Search-and-rescue **postures** — lying, sitting, crouching, in non-urban terrain. Our survivors are not standing pedestrians. |
| **HERIDAL** | RGB | 68,750 images | Wilderness aerial, very small persons; best published mAP 95.11% on its own benchmark |
| **VisDrone** | RGB | 10,209 images | Aerial RGB with a genuine small-object size distribution — the confirming channel |
| **RescueNet** | RGB | 4,494 images | Post-hurricane damage segmentation; source for the hazard model and the belief-map priors |
| **This simulator** | LWIR + RGB | unlimited | Known GSD, apparent temperature, pose and range on every label |

```bash
python3 scripts/train_detector.py --list-datasets    # the table above, with URLs
```

Posture matters more than volume here. A thermal detector trained only on
standing pedestrians misses the exact case that defines the mission — a person
lying still — which is why SARD is in the mix despite being RGB: it is used for
posture-aware augmentation and for the RGB confirming channel, not for the
thermal weights.

**Class map** (`scripts/train_detector.py: CLASSES`) is deliberately small:
`person`, `person_group`, `vehicle`, `structure`, `hazard`. A 5-class problem
with tens of thousands of examples beats an 80-class problem with the same data.

### The synthetic-data argument, stated carefully

The renderer (`sar/sim/renderer.py`) is energy-conserving and radiometric: it
computes apparent temperature from surface temperature, emissivity, atmospheric
transmission, range and the sensor's NETD (45 mK). Sub-pixel area collapse falls
out of the projection maths rather than being drawn in. As a result, synthetic
frames have **physically correct thermal contrast and target size** for a given
altitude and lens.

That makes them genuinely useful for:

* pre-training before a small real dataset is applied,
* covering configurations that are hard to collect (a child, curled, at 60 m, in
  rain, at 03:00),
* regression-testing the whole pipeline in CI.

It does **not** replace real flights. The renderer does not model sensor
non-uniformity drift, shutter events, lens flare from a sun-heated dome, wet
emissivity changes, or the long tail of real clutter. The training doc's
recommended split is: **pre-train synthetic, fine-tune real, validate real
only.**

```bash
python3 scripts/train_detector.py --synthesize 4000 --out datasets/sim-thermal
python3 scripts/train_detector.py --train --data datasets/sim-thermal/data.yaml --p2 --epochs 80
python3 scripts/train_detector.py --train --data datasets/hit-uav/data.yaml \
        --weights runs/sim-thermal/weights/best.pt --epochs 60      # fine-tune
```

---

## 4. The detector stack (`sar/ai/stack.py`)

Every backend — heuristic, ONNX, QNN, TFLite — returns the **same**
`List[Detection]`. Nothing downstream knows which ran.

```
build_detector_stack(mode)
   ├─ "auto"       neural per-modality where weights AND runtime exist,
   │               audited heuristic for the rest              ← flight default
   ├─ "hybrid"     both on the same modality, ensemble-fused   ← evaluation
   ├─ "heuristic"  networks off, deterministic                 ← CI, reference artifacts
   └─ "neural"     networks only; raises RuntimeError("no usable neural backend")
                   rather than silently degrading a benchmark
```

### Three things the neural backends add over a bare YOLO wrapper

1. **Geometry veto.** The detector knows the frame's `gsd_m`, so it can reject a
   box whose implied real-world extent is not human. A network trained on
   ground-level pedestrians will happily fire on a 4 m warm roof panel; this
   check is what stops it. Physics gates the network — never the reverse.
2. **Radiometry from a padded annulus.** After a neural box is accepted, the
   surrounding annulus is sampled to recover background temperature, so
   `sar.perception.thermal` triage produces the same fields it does on the
   heuristic path and the two are directly comparable.
3. **Calibrated confidence.** Raw detector logits are not probabilities. The
   scores are temperature-scaled against a validation set
   (`sar/ai/calibration.py`) so that fusion arithmetic and the operator-facing
   confidence mean what they say.

### Decode orientation

YOLO ONNX exports appear as `(1, 4+nc, N)` or `(1, N, 4+nc)` depending on the
exporter version. **This is not reliably guessable from shape alone** when
`nc ≈ N`. The `ModelSpec` therefore carries `num_classes`, and
`OnnxDetectorEngine` is told explicitly. A misread orientation produces
plausible-looking garbage, which is worse than a crash.

---

## 5. The model registry (`sar/ai/registry.py`)

Weights are **not** in Git. `models/` is ignored except for `README.md` and
`.gitkeep`.

```bash
python3 scripts/fetch_models.py --list          # what the system knows about
python3 scripts/fetch_models.py --model thermal-yolo11n-int8
python3 scripts/fetch_models.py --verify-all    # sha256 every present file
python3 scripts/fetch_models.py --synthetic     # tiny valid ONNX for CI plumbing
```

* Search order: `$SAR_MODELS_DIR` → `models/`.
* `models/registry.json`, if present, overrides the built-in table — that is how
  a field team pins a specific validated build.
* Every entry has an expected sha256; a mismatch is a failure, not a warning.
* `best_for(task, modality)` prefers **QNN → TFLite → ONNX**, i.e. the most
  accelerated format the host can actually run.
* **Missing weights is a supported state**, not an error. `doctor.py` reports it
  and `auto` falls back.

The `--synthetic` model exists so CI can exercise the entire ONNX path without a
download. Its semantics are documented and deliberately trivial:
`score = clip(3 · mean(normalised_input), 0.02, 0.98)`, output `(1, 4+nc, N)`.
It tests plumbing, never accuracy.

---

## 6. Export and quantisation

```bash
python3 scripts/export_model.py --weights runs/thermal/weights/best.pt \
        --out models/thermal-yolo11n.onnx --imgsz 640 --opset 12 --channels 1
python3 scripts/export_model.py --validate --reference models/thermal-fp32.onnx \
        --candidate models/thermal-int8.onnx
python3 scripts/export_model.py --qnn-recipe
```

### The Qualcomm path

The target companion computer is a **QCS6490** class board (Qualcomm RB3 Gen 2)
whose Hexagon DSP runs INT8 through the QNN runtime. YOLO11-Detection is not
directly downloadable from AI Hub for licensing reasons, so it is exported with
`qai-hub-models`:

```bash
pip install qai-hub-models
qai-hub configure --api_token <token>

python3 -m qai_hub_models.models.yolov11_det.export \
        --quantize w8a8 \
        --target-runtime qnn \
        --chipset qualcomm-qcs6490-proxy
```

Notes that cost a day each if you miss them:

* The **default output is TFLite.** Without `--target-runtime qnn` you do not
  get a `.bin` for the DSP.
* `w8a16` (8-bit weights, 16-bit activations) exists for models that lose too
  much at `w8a8`. It is the documented fallback, not a failure.
* On-device inference through `gst-ai-object-detection` needs a JSON config with
  the model's **q-offsets and q-scales** and `"runtime": "dsp"`. Those values
  come out of the export; keep them with the weights.

### The quantisation gate

INT8 is where accuracy quietly disappears. `--validate` runs both models over a
held-out set and compares detections:

```
verdict "acceptable"   only at >= 0.97 recall retained vs FP32
verdict "degraded"     re-export at w8a16, or per-channel calibration
```

A model that fails the gate **does not fly**. This is enforced in the export
script rather than left to discipline.

---

## 7. Latency budget on the aircraft

At 2 Hz perception with both modalities, per cycle:

| Stage | QCS6490 target | Measured (heuristic, x86 dev host) |
|---|---|---|
| Capture + sync (2 cameras) | 30 ms | 12 ms |
| LWIR preprocess | 10 ms | 8 ms |
| Thermal triage + blobs | 25 ms | 41 ms |
| Neural inference LWIR (INT8 DSP) | 25 ms | — (heuristic path) |
| Neural inference RGB (INT8 DSP) | 25 ms | — |
| Fusion, geotag, triage | 40 ms | 61 ms |
| Reporting / queue | 10 ms | 8 ms |
| **Total** | **~165 ms** | **230 ms** (dry run, heuristic) |

The reference *mission* figure of 860 ms mean is a different measurement: it
includes landing-zone search running on the same thread, which is the known
performance gap listed in `STATUS.md`.

---

## 8. Reproducing the accuracy claims in this repository

```bash
python3 scripts/eval_detector.py --mode aimed   --scenario flood
python3 scripts/eval_detector.py --mode survey  --scenario flood
python3 scripts/experiment_subpixel_radiometry.py
```

`aimed` mode removes the confound of "did we happen to fly over them" and
reports recall **of resolvable targets** together with a TTP physics ceiling, so
a detector failure can be distinguished from a target that was never resolvable.
`survey` mode is the only honest way to measure false alarms per km² of real
background clutter.

Current committed numbers (heuristic ensemble, reference flood world): recall of
resolvable 1.00 at 35/50/70 m; background false alarms 0.017 / 0.033 / 0.117 per
frame. Precision on the full sortie ≈ 17%. Improving that number — not recall —
is the top open work item.

---

## 9. Open items in the AI path

| Item | State |
|---|---|
| Trained thermal weights on HIT-UAV + AIResQ + SARD | **Not done.** Pipeline complete and tested end to end; needs GPU time and dataset access approval. |
| QNN `.bin` produced and benchmarked on real QCS6490 hardware | **Not done.** Recipe verified against Qualcomm's documented workflow; no board in hand. |
| Temporal consistency across passes (the precision fix) | Designed, not implemented. |
| Active-learning loop from field false alarms | Designed, not implemented. |

See [`STATUS.md`](../STATUS.md) for the complete list across the whole project.
