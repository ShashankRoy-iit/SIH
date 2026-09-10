#!/usr/bin/env python3
"""AI Simulation Pipeline — Runs YOLO against simulated thermal frames."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from base.sar.perception.yolo_detector import YOLOPersonDetector
from emulator.thermal_sim.thermal_renderer import ThermalRenderer

def main():
    detector = YOLOPersonDetector("../base/models/yolov8n_person_thermal.onnx")
    renderer = ThermalRenderer(mode="flood")
    print("[Sim AI] Starting AI inference on simulated thermal feed...")
    # Example inference loop
    import time
    for i in range(5):
        frame = renderer.render(victims=[{"pos": (320+i*20, 240)}], smoke=0.2)
        detections = detector.detect(frame)
        print(f"[Sim AI] Frame {i}: {len(detections)} detections")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
