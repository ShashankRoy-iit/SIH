# 12 · Model training & running — final results (sandbox + Colab path)

**Date:** 2026-09-13 · **Host:** 2 vCPU, ~3.8 GB RAM, **no GPU**, ~20 GB disk

This document answers the question "why can't you train/run the models here, and
what are the final results?" with **measured numbers**, not intentions. It is the
result record for the detector training work; the *how* lives in
[`docs/06_AI_MODELS_AND_DATASETS.md`](06_AI_MODELS_AND_DATASETS.md) and the
runnable GPU path is [`docs/colab/SAHYOG_SAR_training_GPU_QNN.ipynb`](colab/SAHYOG_SAR_training_GPU_QNN.ipynb).

---

## 1. Why this sandbox cannot run the full training — and what we did instead

| Constraint | This sandbox | What the full fine-tune needs |
|---|---|---|
| GPU | **none** (`nvidia-smi` absent, `torch.cuda.is_available() == False`) | 1× GPU, ≥12 GB VRAM |
| CPU | 2 vCPU @ 2.6 GHz | 8+ cores (nice-to-have) |
| RAM | ~3.8 GB | 16–32 GB (dataset caching, dataloaders) |
| Disk | ~20 GB | 50+ GB (HIT-UAV ≈ 3 GB, AIResQ ≈ 9 GB, plus cache) |

A full fine-tune of YOLO11n on HIT-UAV + AIResQ + SARD (tens of thousands of
frames at 640×512) is **hours on a GPU but ~a day per epoch on this CPU** — and
it would exhaust the 3.8 GB of RAM mid-epoch. So the honest split is:

1. **Here (CPU):** a *complete, end-to-end* train → export → register → detect
   cycle on a simulator-rendered dataset, producing **real trained checkpoints,
   real held-out metrics, and a real end-to-end detection score**. The rendered
   data comes from the project's own energy-conserving radiometric renderer, so
   the physics (thermal contrast, GSD, clutter) is the same — only the volume is
   small.
2. **Colab / Jupyter (GPU):** the full-scale fine-tune on real datasets, with
   every command in a ready notebook (below). Colab's free T4 GPU makes this a
   ~1–2 hour run; the same commands work on a local GPU workstation via
   `scripts/train_detector.py --train`.

---

## 2. The environment that runs the model (rebuilt and verified)

```
torch 2.2.2+cu121 (CPU, cuda=False)   torchvision 0.17.2
ultralytics 8.1.0                      numpy 1.26.4
onnx 1.15.0  onnxruntime 1.30.0        opencv-python-headless 4.9.0.80
```

`onnxruntime` runs the exported detector on `CPUExecutionProvider`. Nothing in
the runtime needs CUDA — inference on the aircraft is NPU/DSP anyway.

---

## 3. Two training runs, and why the first one failed

This is the most useful number in the document. The first run trained the
**stock** `yolov8n.yaml` (stride-8 head) on targets rendered at survey altitude
(30–90 m). The second used the **P2 (stride-4) head** (`yolov8n-p2.yaml`) and
near-pass altitude (15–45 m). Both at imgsz 256, batch 4, 2 CPU threads.

| Run | Head | Targets | Thermal mAP50 | Result |
|---|---|---|---|---|
| 1 | stride-8 (no P2) | 64% of boxes < 8 px | **0.0009** | blind — physically cannot localise sub-8 px objects |
| 2 | **P2 stride-4** | 83% of boxes ≥ 8 px | **0.2338** | learns; ~260× improvement |

Aerial survivors are 4–10 px at survey altitude. A stride-8 detector's finest
grid cell is 8 px, so a 4 px body is less than one cell — the model *cannot*
see it, regardless of epochs. This is exactly why the deployment spec
([`docs/06_AI_MODELS_AND_DATASETS.md`](06_AI_MODELS_AND_DATASETS.md) §2) uses a
P2 head and 640×512 input, and why the full GPU run below is configured with
`--p2`.

---

## 4. Datasets (rendered by the radiometric simulator)

| Dataset | Frames (train/val) | Boxes | Persons | Histogram `<8 / 8–16 / 16–32 / >32` px |
|---|---|---|---|---|
| `datasets/sim-thermal` | 800 (666/134) | 370 | 271 | 64 / 144 / 75 / 87 |
| `datasets/sim-rgb` | 500 (416/84) | 241 | 169 | 40 / 91 / 49 / 61 |

Generated with `scripts/train_detector.py --synthesize … --alt-min 15 --alt-max 45`
(modality `lwir` and `rgb`). 60% of frames are aimed near a survivor (an
all-empty aerial dataset trains a model that predicts "nothing").

---

## 5. Trained checkpoints and held-out metrics

YOLOv8n-**P2**, 30 epochs, imgsz 256, batch 4, CPU.

| Model | Best checkpoint | mAP50 | mAP50-95 | Precision | Recall | Wall time |
|---|---|---|---|---|---|---|
| Thermal (LWIR) | `runs/sar-thermal3/weights/best.pt` | **0.2338** | 0.0603 | **0.9175** | 0.2027 | 34.8 min |
| RGB (phone cam) | `runs/sar-rgb2/weights/best.pt` | **0.1614** | 0.0638 | 0.5976 | 0.1745 | 22.1 min |

Full record: `artifacts/train_results.json`. These are the unbiased held-out
numbers (random split, empty frames included, IoU@0.5). mAP50-95 is low because
8–16 px boxes are IoU-sensitive — the same reason the repo's geometry veto and
search-theory fusion operate on the *projected extent*, not raw IoU.

Exported ONNX (opset 12), registered in `models/registry.json` so the flight
default `SAR_DETECTOR=auto` picks them up:

| ONNX | Size | Input | Output (raw P2 head) | sha256 (prefix) |
|---|---|---|---|---|
| `models/yolov8n-thermal-sim.onnx` | 11.85 MB | `(1,3,256,256)` | `(1,10,5440)` | `15e3619f…` |
| `models/yolov8n-rgb-sim.onnx` | 11.85 MB | `(1,3,256,256)` | `(1,10,5440)` | `5d9e6407…` |

> Both are 3-channel inputs: ultralytics feeds single-channel LWIR as a
> grayscale-replicated 3-channel tensor. The deployment 1-channel stem is the
> GPU/`--channels 1` path in `scripts/train_detector.py` + `yolo11n-thermal-sar`.

---

## 6. End-to-end result: the trained detector on the aircraft runtime

`datasets/eval_neural_end2end.py` renders survey frames from the reference
flood scenario (20–40 m, 70% aimed near a survivor), pushes them through the
**same `build_detector_stack("neural")` runtime the aircraft uses** (ONNX →
decode → geometry veto), and scores against the renderer's geometric truth.

| Modality | Recall | Precision | F1 | TP / FP / FN | Mean latency |
|---|---|---|---|---|---|
| Thermal (LWIR) | **0.697** | **1.000** | 0.821 | 23 / 0 / 10 | 23.5 ms |
| RGB (phone cam) | **0.667** | **1.000** | 0.800 | 22 / 0 / 11 | 23.1 ms |

**Zero false alarms in 60 frames per modality** — the trained detector plus the
geometry veto delivers the precision the heuristic ensemble could not (its
precision was ≈17% on the same sortie, the #1 open gap in `STATUS.md` §2.1).
The recall gap (≈30%) is the honest cost of the small dataset + imgsz 256; the
GPU run's 640×512 P2 model on real data is the path to closing it. Full report:
`artifacts/neural_eval.json`.

Stack assembly, verified for all four modes (models auto-selected via
`best_for()` → `models/registry.json`):

```
auto      -> neural_lwir:yolov8n-thermal-sim + neural:yolov8n-rgb-sim + heuristic_rgb(hazards)
hybrid    -> neural both + heuristic both
neural    -> neural_lwir:yolov8n-thermal-sim + neural:yolov8n-rgb-sim
heuristic -> heuristic_lwir + heuristic_rgb
```

---

## 7. The phone is the RGB camera (integration state)

The RGB detector above is **the phone's camera model**, not a flight-controller
camera. The phone path is exercised end to end:

- `mobile/phone_onboard.py --mode bench --duration 15` →
  **frames 225 · VIO 444 · rgb_fps 15.0 · vio_hz 29.6 · det_events 225 ·
  link streaming** (`artifacts/phone_bench.json`).
- Phone NPU model: `scripts/export_phone_model.py --emit-stub` →
  `models/person-nano-int8.onnx`; the trained RGB weights replace the stub via
  `scripts/export_phone_model.py --recipe` (TFLite-INT8 → NNAPI/Hexagon).
- Build doc updated with the measured bench numbers:
  [`docs/MOBILE_COMPANION.md`](MOBILE_COMPANION.md) §6.

---

## 8. The GPU solution — Colab / Jupyter notebook (ready to run)

[`docs/colab/SAHYOG_SAR_training_GPU_QNN.ipynb`](colab/SAHYOG_SAR_training_GPU_QNN.ipynb)
contains the complete full-scale pipeline, cell by cell:

1. Clone repo + install (`pip install -e '.[sim,dev]'` + `ultralytics`).
2. Real datasets — HIT-UAV (2,898 aerial LWIR frames), AIResQ (best published
   aerial-thermal mAP), SARD (postures, RGB) — or synthesise 4k frames.
3. Train YOLO11n **`--p2`** thermal (1-channel, 640×512) and RGB (640×640),
   120 epochs, batch 16, `--device 0`.
4. Export ONNX (opset 12) + TFLite-INT8 (phone Hexagon DSP).
5. QNN `w8a8` quantisation for the Qualcomm QCS6490 (RB3 Gen 2) NPU, gated by
   the repo's ≥97% recall-retention check (`export_model.py --validate`).
6. Register + `fetch_models.py --verify-all` + `eval_detector.py` + a live
   mission with `SAR_DETECTOR=auto`.

The equivalent local-workstation commands (no notebook) are in
`scripts/train_detector.py --train --p2 …` and `scripts/export_model.py`.

**Expected full-scale ceiling (published baselines, for calibration):**
aerial-thermal person detection reaches mAP50 ≈ 0.55 on AIResQ-trained YOLO;
small objects are the hard subset (RT-DETR-L ≈ 5% AP vs YOLO ≈ 11%). The
sandbox's 0.23 mAP50 on 370 boxes / 30 epochs is consistent with that curve at
tiny data volume; the Colab run is what reaches the published ceiling.

---

## 9. Reproduce everything

```bash
# 1. environment
python3 -m venv .venv && .venv/bin/pip install -e '.[sim,dev]' onnx onnxruntime
.venv/bin/pip install "ultralytics==8.1.0"

# 2. data + train (CPU, ~1 h) + register + evaluate
.venv/bin/python scripts/train_detector.py --synthesize 800 --modality lwir --alt-min 15 --alt-max 45 --out datasets/sim-thermal
.venv/bin/python scripts/train_detector.py --synthesize 500 --modality rgb  --alt-min 15 --alt-max 45 --out datasets/sim-rgb
.venv/bin/python datasets/train_all.py          # trains + exports both ONNX
.venv/bin/python datasets/postprocess.py         # registers + smoke-tests
.venv/bin/python datasets/eval_neural_end2end.py --json artifacts/neural_eval.json

# 3. fly the trained stack
SAR_DETECTOR=auto .venv/bin/python scripts/run_mission.py --scenario flood --duration 460

# 4. full-scale (GPU / Colab)
#    open docs/colab/SAHYOG_SAR_training_GPU_QNN.ipynb
```
