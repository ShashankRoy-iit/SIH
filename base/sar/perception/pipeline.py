#!/usr/bin/env python3
"""Perception Pipeline — Cross-Modal Fusion + Geo-Tagging.

Integrates RGB, thermal, AI detector, tracker, and geo-tagger. Designed for
onboard companion computer (RB3 Gen 2 / VOXL 2) connected to PX4 via MAVLink.
"""
import time
from typing import List, Dict, Any, Optional

# Local imports
try:
    from .yolo_detector import YOLOPersonDetector
    from .thermal_simulation import ThermalCameraSimulation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from sar.perception.yolo_detector import YOLOPersonDetector
    from sar.perception.thermal_simulation import ThermalCameraSimulation


class PerceptionPipeline:
    def __init__(self, detector_path: str, conf: float = 0.35):
        self.detector = YOLOPersonDetector(detector_path, conf_threshold=conf)
        self.thermal_sim = ThermalCameraSimulation()
        print("[Pipeline] Perception pipeline initialized.")

    def process_thermal_frame(self, frame, visualize: bool = False) -> List[Dict[str, Any]]:
        detections = self.detector.detect(frame, visualize=visualize)
        # Enrich with thermal info
        for d in detections:
            d["thermal"] = True
            d["source"] = "thermal_ai"
        return detections

    def geo_tag(self, detection: Dict, drone_lat: float = 0.0, drone_lon: float = 0.0, drone_alt: float = 30.0) -> Dict:
        """Simple geo-tagger. In real deployment uses MAVLink global_position_int."""
        # Placeholder geo-tagging
        box = detection.get("box", [0, 0, 1, 1])
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        # Rough offset from center of image (pixel -> meters approximation)
        offset_x = (cx - 320) * 0.05  # meters
        offset_y = (cy - 240) * 0.05
        detection["geo"] = {
            "latitude": drone_lat + offset_y / 111320,
            "longitude": drone_lon + offset_x / (111320 * 0.99),
            "altitude_m": drone_alt,
            "uncertainty_sigma_m": 15.0 + (1.0 - detection.get("conf", 0.5)) * 20
        }
        return detection
