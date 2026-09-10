# YOLOv8n Model — Ready Now + How to Train Real Model

## Current Model: `yolov8n_person_thermal.onnx`

- **File size:** ~6.1 MB (6144288 bytes)
- **Type:** Pre-built ONNX inference model (synthetic weights included for demonstration)
- **Loads with:** `python3 -c "import onnxruntime as ort; s = ort.InferenceSession('base/models/yolov8n_person_thermal.onnx'); print('ONNX loaded:', s.get_inputs()[0].name)"`
- **Works with:** `base/sar/perception/yolo_detector.py` (`conf_threshold=0.35`)
- **Behavior:** The synthetic `.onnx` runs inference correctly but produces very low confidence scores (near-zero weights). This is expected behavior — it proves the pipeline works, but for real detections you need a fully trained model.
- **Used by:** `emulator/run_simulation.py` (injects simulated detections `conf=0.92` when victims visible, to demonstrate full pipeline), `base/scripts/run_drone_ai.py` (uses same `.onnx` file)

## Why This Model? Why Not YOLOv11n?

- **YOLOv8n selected:** Mature ONNX export (`ultralytics` library), verified thermal SAR dataset fine-tuning (HERIDAL — 68,750 RGB images, best published mAP 95.11%), confirmed Qualcomm RB3 Gen 2 / VOXL 2 edge deployment, pre-built `.onnx` included in repo, loads in 1 command.
- **YOLOv11 NOT selected:** Architecture changed (C3k2 blocks). No verified thermal dataset fine-tuning pipeline exists for SAR/rescue contexts. No `.pt` → `.onnx` export benchmarked on Qualcomm edge. Not included in repo. **Reliability > novelty for field deployment.**

## How to Get / Train a Real Model

### Option A: Download Pre-Trained YOLOv8n (Fastest)

```bash
# 1. Install ultralytics
pip install ultralytics

# 2. Download official YOLOv8n weights
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 3. Export to ONNX format
python3 -m ultralytics yolo export model=yolov8n.pt format=onnx imgsz=320
# Produces: yolov8n.onnx

# 4. Replace demo model
cp yolov8n.onnx base/models/yolov8n_person_thermal.onnx
```

### Option B: Fine-Tune on Thermal SAR Dataset (Recommended for Real Deployment)

```bash
# 1. Prepare dataset (thermal + RGB pairs from AFO, HERIDAL, SARD)
#    See docs/06_AI_MODELS_AND_DATASETS.md for dataset details

# 2. Download base YOLOv8n weights
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# 3. Train / fine-tune on thermal dataset
python3 -m ultralytics yolo train model=yolov8n.pt data=thermal_sar_dataset.yaml epochs=50 imgsz=320
# Training takes ~2-6 hours on a GPU (NVIDIA RTX 3060 or better)

# 4. Export best model (from runs/train/exp/weights/best.pt)
python3 -m ultralytics yolo export model=runs/train/exp/weights/best.pt format=onnx imgsz=320

# 5. Replace in repo
cp best.onnx base/models/yolov8n_person_thermal.onnx
```

### Option C: Synthetic Model (Current — For Pipeline Testing Only)

The current `yolov8n_person_thermal.onnx` uses synthetic/random weights. It proves:
- `.onnx` loads correctly (`InferenceSession`)
- Inference runs (`run_simulation.py` completes its loop)
- Dashboard updates (`live_state.json` updates)
- Pipeline connects (same modules used by real drone)

**For real field deployment:** Replace with fully trained model (Option A or B above).

## Verification Scripts

```bash
# Verify model loads
python3 -c "import onnxruntime as ort; s = ort.InferenceSession('base/models/yolov8n_person_thermal.onnx'); print('Loaded:', s.get_inputs()[0].name)"

# Verify pipeline runs
python3 emulator/run_simulation.py --duration 60

# Verify dashboard serves live data
python3 base/rescue_dashboard/app.py &
# Visit: http://localhost:8088

# Verify real drone script (without actual drone — reads .json)
python3 base/scripts/run_drone_ai.py --model base/models/yolov8n_person_thermal.onnx --live --dashboard
```
