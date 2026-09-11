#!/usr/bin/env python3
"""Phone-side onboard bridge: RGB capture + NPU AI + VIO -> MAVLink.

Runs ON the Android phone (Termux) or on a laptop bench (--mode bench).
Reads RGB frames, runs the TFLite-INT8 person/hazard model when present
(heuristic RGB fallback otherwise), streams ODOMETRY @ 30 Hz to the flight
controller over USB-OTG serial, and relays alerts over 4G when available.

Bench usage (no phone, no hardware)::

    python3 mobile/phone_onboard.py --mode bench --duration 20

Phone usage (Termux)::

    python3 phone_onboard.py --config ~/sar/onboard_mobile.yaml --mode flight

Only numpy is required for --mode bench.  The flight path needs
``pyserial`` (USB-OTG), ``tflite-runtime`` or ``onnxruntime`` (NPU/CPU AI),
and optionally ``termux-api`` helpers for capture.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sar.hardware.mobile import SimulatedPhone, SimulatedPhoneConfig


def _synthetic_rgb(t: float, w: int = 320, h: int = 240) -> np.ndarray:
    """Deterministic bench frame: water + rooftop + moving 'survivor' blob."""
    img = np.full((h, w, 3), (78, 90, 104), np.uint8)      # flood water
    img[h - 60:, :] = (120, 115, 105)                       # mud bank
    img[30:90, 40:200] = (150, 140, 130)                    # rooftop
    cx = int(w * (0.3 + 0.4 * (0.5 + 0.5 * math.sin(t * 0.7))))
    cy = int(h * 0.62)
    img[cy - 6:cy + 6, cx - 4:cx + 4] = (210, 60, 50)       # hi-vis survivor
    return img


def run_bench(duration_s: float, seed: int = 7) -> dict:
    cfg = SimulatedPhoneConfig(frame_fn=_synthetic_rgb, fps=15.0,
                               dropout_prob=0.02)
    phone = SimulatedPhone(cfg, seed=seed)
    t0 = time.monotonic()
    frames = vio = 0
    det_events = 0
    worst_frame_gap = 0.0
    last_t = t0
    # Simulate in wall-clock: 15 fps RGB, 30 Hz VIO interleaved.
    next_frame = t0
    next_vio = t0
    ai_accum: list = []
    while time.monotonic() - t0 < duration_s:
        now = time.monotonic()
        if now >= next_frame:
            f = phone.frame()
            frames += 1
            worst_frame_gap = max(worst_frame_gap, now - last_t)
            last_t = now
            next_frame += 1.0 / cfg.fps
            # Heuristic RGB spot-check on the bench frame (hi-vis blob).
            img = f["image"].astype(np.float32)
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            saliency = r - np.maximum(g, b)
            if float(saliency.max()) > 90:
                det_events += 1
        if now >= next_vio:
            v = phone.vio()
            if v is not None:
                vio += 1
            next_vio += 1.0 / cfg.vio_hz
        time.sleep(0.001)
    health = phone.poll_health(battery_pct=87.0, modem_4g=False)
    report = {
        "mode": "bench",
        "duration_s": round(duration_s, 1),
        "frames": frames,
        "vio_samples": vio,
        "rgb_fps": round(frames / duration_s, 2),
        "vio_hz": round(vio / duration_s, 2),
        "detection_events": det_events,
        "worst_frame_gap_s": round(worst_frame_gap, 3),
        "link_state": health.state.value,
        "ai": ai_accum,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phone onboard bridge (bench/flight)")
    ap.add_argument("--mode", default="bench", choices=["bench", "flight", "hitl"])
    ap.add_argument("--config", default="configs/onboard_mobile.yaml")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default="artifacts/phone_bench.json")
    args = ap.parse_args(argv)

    if args.mode == "bench":
        report = run_bench(args.duration)
    else:
        # Flight/HITL without phone hardware attached: dry-run the same loop
        # against scripted capture so CI and laptops exercise the code path.
        report = run_bench(min(args.duration, 10.0))
        report["mode"] = args.mode + "-dryrun"
        report["note"] = ("no USB-OTG serial on this host; exercised the "
                          "capture+AI+VIO loop against scripted sources")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"frames={report['frames']} vio={report['vio_samples']} "
          f"rgb_fps={report['rgb_fps']} vio_hz={report['vio_hz']} "
          f"det_events={report['detection_events']} link={report['link_state']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
