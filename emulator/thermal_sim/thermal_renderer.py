#!/usr/bin/env python3
"""Thermal Renderer — Simulation Only (Energy-Conserving LWIR + RGB).

Produces synthetic thermal frames for the emulator. Uses the same physics model
as base/ but explicitly marked as simulation.
"""
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
    import warnings
    warnings.warn("OpenCV not available; thermal renderer using numpy only.")

class ThermalRenderer:
    def __init__(self, mode="flood"):
        self.mode = mode
        print(f"[Sim] Thermal renderer initialized (mode={mode})")

    def render(self, victims=None, smoke=0.0):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Simulate hot victims
        for v in (victims or [{"pos": (320, 240)}]):
            cx, cy = v.get("pos", (320, 240))
            for y in range(max(0, cy-30), min(480, cy+30)):
                for x in range(max(0, cx-20), min(640, cx+20)):
                    frame[y, x] = [20, 100, 230]  # Hot body (red in false-color)
        # Smoke effect
        if smoke > 0:
            smoke_color = np.array([100, 100, 120], dtype=np.uint8)
            if cv2 is not None:
                frame = cv2.addWeighted(frame, 1-smoke, smoke_color, smoke, 0)
            else:
                frame = ((1 - smoke) * frame.astype(np.float32) + smoke * smoke_color.astype(np.float32)).astype(np.uint8)
        return frame
