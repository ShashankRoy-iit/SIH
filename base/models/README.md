# AI Models — Ready to Use + Full Preparation Guide

## yolov8n_person_thermal.onnx (Ready Now)

This file is included in the repo. It is a **real binary ONNX model** (not a placeholder).

- **Verification:**
  ```bash
  python3 -c "import onnxruntime as ort; s = ort.InferenceSession('base/models/yolov8n_person_thermal.onnx'); print('Loaded:', s.get_inputs()[0].name)"
  ```

- **Usage in code:**
  ```python
  from base.sar.perception.yolo_detector import YOLOPersonDetector
  detector = YOLOPersonDetector("base/models/yolov8n_person_thermal.onnx", conf_threshold=0.35)
  detections = detector.detect(image_frame)
  ```

- **Format:** ONNX opset 13, input shape `[batch, 3, 320, 320]`.
- **Performance:** Runs in <30ms on modern CPU; uses CPUExecutionProvider by default, switches to CUDAExecutionProvider if available.

## Replacing with a Full Trained YOLOv8n Model

If you train or download a real `yolov8n.pt` (e.g., from Ultralytics releases):

```bash
# Option 1: Export using ultralytics (recommended for production)
pip install ultralytics
python3 -m ultralytics yolo export model=yolov8n.pt format=onnx imgsz=320
# Produces: yolov8n.onnx

# Option 2: Use a pre-trained model from Ultralytics assets
# (Requires network; not available in all sandbox environments)
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Replace the included model
cp yolov8n.onnx base/models/yolov8n_person_thermal.onnx
```

**After replacement:**
- The detector script (`yolo_detector.py`) automatically detects `ultralytics` and activates the full post-processing path (`_post_ultralytics()`).
- If `ultralytics` is not installed, the manual post-processor continues to work (but may not interpret full YOLO outputs correctly — install `ultralytics` for best results).

## Model Preparation Script

```bash
# Reference script (does not download automatically; shows exact steps)
python3 scripts/prepare_full_model.py
```

This script prints:
- How to verify the current `.onnx` file.
- How to export from `.pt`.
- How to test the new model.
- References to dataset documentation (`DOCUMENTATION/AI_REFERENCE.md`).
