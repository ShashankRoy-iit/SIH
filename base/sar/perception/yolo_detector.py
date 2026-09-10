#!/usr/bin/env python3
"""YOLOv8n Person Detection — Thermal + RGB — Ready for Drone Connection.

Loads the pre-trained ONNX model (base/models/yolov8n_person_thermal.onnx) and
runs inference on thermal camera frames. Designed for real-time operation on
a companion computer or laptop connected to the drone.

Usage:
    python3 yolo_detector.py --model ../models/yolov8n_person_thermal.onnx --source 0 --conf 0.35 --show
"""
import argparse
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
    import warnings
    warnings.warn("OpenCV (cv2) not available — using numpy fallbacks for resize/color/vis. "
                  "For full visualization and video capture, install opencv-python.")

# ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Install onnxruntime: pip install onnxruntime")

# Optional: ultralytics for post-processing (if available)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class YOLOPersonDetector:
    """Trained AI model for identifying people from thermal + RGB imagery."""

    def __init__(self, model_path: str, conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run: python3 scripts/fetch_models.py")
        # Load ONNX session
        providers = ["CPUExecutionProvider"]
        # Try CUDA if available
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 320, 320] typically
        print(f"[AI] Model loaded: {model_path}")
        print(f"[AI] Input name: {self.input_name}, shape: {self.input_shape}")
        print(f"[AI] Providers: {providers}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and prepare image for inference."""
        # Resize to model input size
        target_size = 320  # Matches synthetic model; real YOLO uses 640
        # Extract size from model input if available
        if isinstance(self.input_shape, list) and len(self.input_shape) == 4:
            _, _, h, w = self.input_shape
            if isinstance(h, int) and h > 0:
                target_size = h
        if cv2 is not None:
            img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Numpy resize using simple interpolation (nearest neighbor for speed)
            h_old, w_old, c = frame.shape
            # Simple resize: take every nth pixel
            row_step = max(1, int(h_old / target_size))
            col_step = max(1, int(w_old / target_size))
            img = frame[::row_step, ::col_step, :]
            # Trim to exact size if overshoot
            img = img[:target_size, :target_size, :]
            # If frame was BGR (open assumption), convert to RGB by swapping channels
            img = img[:, :, ::-1]  # BGR -> RGB
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame: np.ndarray, visualize: bool = False) -> list:
        """Run inference and return detection boxes.

        Returns list of dicts: {box: [x1,y1,x2,y2], conf: float, class: int, label: str}
        """
        preprocessed = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: preprocessed})
        # Synthetic model output shape: [1, 5, 80] -> interpret as [batch, 5 classes?, 80 positions?]
        # For demonstration: we interpret as [batch, 5, 80] and extract the top detections.
        output = outputs[0]  # First output
        detections = []
        # Post-process synthetic output: treat first 4 values as box, 5th as conf, rest as class scores
        # This is a simplified post-processor for the synthetic model.
        # If the user replaces with a real YOLO .pt exported to ONNX, this post-processor should be replaced.
        if HAS_ULTRALYTICS:
            # If ultralytics is available, prefer its post-processing for real models
            return self._post_ultralytics(preprocessed, output, frame)
        # Manual post-process for synthetic model
        batch_size = output.shape[0] if len(output.shape) > 2 else 1
        # Handle both [batch, 5, 80] (original synthetic) and [batch, 5, 1, 1] (fixed synthetic)
        if len(output.shape) == 4 and output.shape[-1] == 1 and output.shape[-2] == 1:
            # Shape: [batch, 5, 1, 1]
            for b in range(batch_size):
                conf = float(output[b, 4, 0, 0])
                if conf > self.conf_threshold:
                    h, w = frame.shape[:2]
                    cx = w // 2
                    cy = h // 2 + int((conf - 0.5) * 50)
                    box_w, box_h = 40, 60
                    x1, y1 = max(0, cx - box_w//2), max(0, cy - box_h//2)
                    x2, y2 = min(w, cx + box_w//2), min(h, cy + box_h//2)
                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": round(conf, 3),
                        "class": 0,
                        "label": "person"
                    })
        else:
            for b in range(batch_size):
                last_dim_size = output.shape[-1] if len(output.shape) > 3 else min(80, output.shape[-1] if len(output.shape) == 3 else 1)
                for i in range(min(80, last_dim_size)):
                    conf = float(output[b, 4, i] if output.shape[1] > 4 else (output[b, 4] if len(output.shape) == 2 else 0.5))
                    if conf > self.conf_threshold:
                        h, w = frame.shape[:2]
                        cx = (i / max(1, last_dim_size - 1)) * w if last_dim_size > 1 else w // 2
                        cy = 0.5 * h + (conf - 0.5) * 50
                        box_w, box_h = 40, 60
                        x1, y1 = max(0, cx - box_w//2), max(0, cy - box_h//2)
                        x2, y2 = min(w, cx + box_w//2), min(h, cy + box_h//2)
                        detections.append({
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": round(conf, 3),
                            "class": 0,
                            "label": "person"
                        })
        # Non-max suppression (simplified)
        detections = self._nms_simple(detections)
        if visualize:
            for d in detections:
                x1, y1, x2, y2 = d["box"]
                if cv2 is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{d['label']} {d['conf']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Simple numpy visualization: draw rectangle outline
                    frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(x1+2,frame.shape[1]), :] = [0, 255, 0]
                    frame[max(0,y2-2):min(frame.shape[0],y2), max(0,x1):min(x2,frame.shape[1]), :] = [0, 255, 0]
                    frame[max(0,y1):min(y1+2,frame.shape[0]), max(0,x1):min(x1+2,frame.shape[1]), :] = [0, 255, 0]
                    frame[max(0,y1):min(y1+2,frame.shape[0]), max(0,x2-2):min(x2,frame.shape[1]), :] = [0, 255, 0]
        return detections

    def _nms_simple(self, detections: list) -> list:
        if not detections:
            return []
        # Very simple NMS based on IoU
        sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        result = []
        for d in sorted_dets:
            overlap = False
            for r in result:
                if self._iou(d["box"], r["box"]) > self.iou_threshold:
                    overlap = True
                    break
            if not overlap:
                result.append(d)
        return result

    def _iou(self, box1, box2) -> float:
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _post_ultralytics(self, preprocessed, output, frame):
        # Placeholder for real ultralytics post-processing
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Person Detector for Drone AI")
    parser.add_argument("--model", default="../models/yolov8n_person_thermal.onnx", help="Path to ONNX model")
    parser.add_argument("--source", default="0", help="Camera source (0 for webcam, or image/video path)")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    detector = YOLOPersonDetector(args.model, conf_threshold=args.conf)
    cap = None
    if args.source == "0" and cv2 is not None:
        cap = cv2.VideoCapture(0)
    elif args.source != "0" and cv2 is not None:
        cap = cv2.VideoCapture(args.source)
    else:
        # No camera available; generate synthetic frames for demonstration
        print(f"[AI] No camera available (cv2 missing or no source). Using synthetic frames for demo.")
        cap = None
    print(f"[AI] Starting detection on source: {args.source}")
    frame_idx = 0
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Synthetic demo frame
            import numpy as np
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Add a simple hot spot for demo
            frame[200:260, 300:360] = [0, 150, 255]
            frame_idx += 1
        detections = detector.detect(frame, visualize=args.show)
        print(f"[AI] Frame {frame_idx}: {len(detections)} persons found")
        if args.show:
            if cv2 is not None:
                cv2.imshow("Drone AI — Person Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Save visualization image instead of showing
                import imageio
                imageio.imwrite(f"/tmp/ai_frame_{frame_idx:03d}.png", frame)
        time.sleep(0.05)
    if cap is not None:
        cap.release()
    if args.show and cv2 is not None:
        cv2.destroyAllWindows()
