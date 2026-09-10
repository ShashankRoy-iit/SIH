#!/usr/bin/env python3
"""Thermal Camera Simulation — Energy-Conserving LWIR Renderer.

Produces synthetic thermal images for testing the AI pipeline without a real
LWIR camera. Uses a simple energy-conservation model: hot objects emit more,
cold objects emit less, smoke reduces contrast.

Usage:
    python3 sim_thermal.py --output thermal_feed.jpg --mode night --smoke 0.3
"""
import argparse
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


class ThermalCameraSimulation:
    def __init__(self, width: int = 640, height: int = 480, mode: str = "day"):
        self.width = width
        self.height = height
        self.mode = mode  # day, night, smoke, flood
        self.base_temp = 300.0  # Kelvin

    def render_frame(self, detections: list = None, smoke_level: float = 0.0):
        """Generate a synthetic thermal frame.

        Args:
            detections: List of dicts with 'box' for simulated people.
            smoke_level: 0.0 (clear) to 1.0 (heavy smoke reducing contrast).
        Returns:
            np.ndarray (H, W, 3) in false-color thermal (red = hot, blue = cold).
        """
        # Base thermal gradient
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        # Add noise
        noise = np.random.randint(-10, 10, (self.height, self.width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add simulated people (hot spots)
        for d in (detections or []):
            x1, y1, x2, y2 = d.get("box", [self.width//2-20, self.height//2-30, self.width//2+20, self.height//2+30])
            # Hot body: red/yellow gradient
            for y in range(max(0, y1), min(self.height, y2)):
                for x in range(max(0, x1), min(self.width, x2)):
                    # Gradient: hotter at center
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    intensity = max(0, 255 - int(dist * 3))
                    frame[y, x] = [0, intensity // 2, 255 - intensity]  # Red-ish

        # Add smoke effect (reduces contrast globally)
        if smoke_level > 0:
            smoke_mask = np.ones((self.height, self.width), dtype=np.float32) * smoke_level
            # Apply as blend
            smoke_color = np.array([100, 100, 120], dtype=np.uint8)  # Grayish
            if cv2 is not None:
                frame = cv2.addWeighted(frame, 1 - smoke_level, smoke_color, smoke_level, 0)
            else:
                frame = ((1 - smoke_level) * frame.astype(np.float32) + smoke_level * smoke_color.astype(np.float32)).astype(np.uint8)

        # Mode adjustments
        if self.mode == "night":
            # Darker overall, people still hot
            frame = np.clip(frame.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
        elif self.mode == "flood":
            # Cold water background (blue)
            frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) - 50, 0, 255).astype(np.uint8)
            frame[:, :, 2] = np.clip(frame[:, :, 2].astype(np.int16) + 40, 0, 255).astype(np.uint8)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Camera Simulation")
    parser.add_argument("--output", default="thermal_feed.jpg")
    parser.add_argument("--mode", default="day", choices=["day", "night", "smoke", "flood"])
    parser.add_argument("--smoke", type=float, default=0.0)
    args = parser.parse_args()

    sim = ThermalCameraSimulation(width=640, height=480, mode=args.mode)
    # Simulate a person in the center
    detections = [{"box": [280, 200, 360, 320]}]
    frame = sim.render_frame(detections, smoke_level=args.smoke)
    if cv2 is not None:
        cv2.imwrite(args.output, frame)
    else:
        import imageio
        imageio.imwrite(args.output, frame)
    print(f"[Thermal] Rendered thermal frame to {args.output} (mode={args.mode}, smoke={args.smoke})")
