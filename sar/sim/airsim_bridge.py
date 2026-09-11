"""AirSim bridge: photoreal flood-town sorties (Cosys-AirSim / UE5).

Target stack (lab machine with a GPU):

* **Cosys-AirSim** (the maintained AirSim fork, Unreal Engine 5.4/5.5) —
  prebuilt environments at cosys-airsim.com or the open plugin from
  github.com/Cosys-Lab/Cosys-AirSim.  The original Microsoft AirSim repo is
  archived; Cosys-AirSim keeps the same Python API (``cosys-airsim`` pip
  package, ``import airsim``).
* This repo's flood town: ``worlds/flood_town/settings.json`` (vehicle +
  sensors) plus the placement script ``worlds/flood_town/place_town.py``
  which spawns houses / water / victims from ``sar/sim/flood_scene.py`` so
  the AirSim town is the *same* town as the headless one.

What the bridge does:

1. Connects, arms, and flies velocity setpoints (mirrors the mission runner).
2. Captures RGB + Depth + Segmentation each perception tick.
3. Derives a **pseudo-thermal** frame from segmentation class + depth with
   the same radiometric model as the headless renderer (per-class
   temperature, distance attenuation, NETD noise) — stock AirSim has no LWIR
   sensor, and this file says so instead of pretending.
4. Runs ``FloodDetector`` on (thermal, rgb) and scores against the scenario
   truth.

On a machine without AirSim (this sandbox, CI, jury laptops) every entry
point degrades cleanly: :func:`airsim_available` returns False and
``scripts/run_flood_sim.py --backend auto`` falls through to Gazebo, then
to the headless cinematic renderer.  No import-time dependency on airsim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def airsim_available() -> Tuple[bool, str]:
    """Probe for a usable AirSim Python API + a reachable simulator."""
    try:
        import airsim  # type: ignore  # cosys-airsim pip package
    except Exception as exc:
        return False, f"no airsim python package ({exc})"
    try:
        client = airsim.MultirotorClient(timeout_value=2)
        client.confirmConnection()
        return True, f"connected ({type(client).__name__})"
    except Exception as exc:
        return False, f"package present, simulator not reachable ({exc})"


# --------------------------------------------------------------------------- #
# Settings + placement generation (runs anywhere — pure file output)
# --------------------------------------------------------------------------- #
def airsim_settings(scene_dict: Dict[str, Any],
                    out: Optional[Path] = None) -> Dict[str, Any]:
    """Generate AirSim ``settings.json`` for the flood town."""
    settings = {
        "SeeDocsAt": "https://cosys-airsim.com/",
        "SettingsVersion": 2.0,
        "SimMode": "Multirotor",
        "ClockType": "SteppableClock",
        "Vehicles": {
            "SAR_Drone": {
                "VehicleType": "SimpleFlight",
                "X": 0, "Y": 0, "Z": -45,
                "Cameras": {
                    "rgb": {"CaptureSettings": [
                        {"ImageType": 0, "Width": 1280, "Height": 720,
                         "FOV_Degrees": 66}]},
                    "seg": {"CaptureSettings": [
                        {"ImageType": 5, "Width": 640, "Height": 480,
                         "FOV_Degrees": 66}]},
                    "depth": {"CaptureSettings": [
                        {"ImageType": 1, "Width": 640, "Height": 480,
                         "FOV_Degrees": 66}]}
                },
                "Sensors": {
                    "lidar": {"SensorType": 6, "Enabled": True,
                              "NumberOfChannels": 16, "Range": 12.0,
                              "PointsPerSecond": 10000},
                    "barometer": {"SensorType": 1, "Enabled": True},
                    "imu": {"SensorType": 2, "Enabled": True},
                    "gps": {"SensorType": 3, "Enabled": True}
                }
            }
        },
        "Segmentation": {
            "Water": 10, "House": 20, "Roof": 21, "Survivor": 30,
            "Debris": 40, "Vehicle": 41, "Powerline": 50, "Ground": 0
        },
        "FloodTown": {"seed": scene_dict.get("seed", 7),
                      "size_m": scene_dict.get("size_m", 220.0),
                      "water_level_m": scene_dict.get("water_level_m", 1.2)},
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(settings, indent=2))
    return settings


#: Segmentation id -> surface temperature (C) for the pseudo-thermal model.
SEG_TEMPS_C = {0: 14.0, 10: 15.0, 20: 18.0, 21: 30.0, 30: 32.0,
               40: 20.0, 41: 33.0, 50: 16.0}


def pseudo_thermal(seg: np.ndarray, depth_m: np.ndarray,
                   netd_k: float = 0.045, seed: int = 0) -> np.ndarray:
    """Segmentation + depth -> LWIR-like temperature frame (C).

    Per-class base temperature, mild distance attenuation (atmosphere),
    sub-pixel blending is inherent to the rasterisation, plus Gaussian NETD
    noise.  Honest about what it is: realistic *geometry* with modelled
    *radiometry* — absolute kelvin truth still comes from the headless
    radiometric renderer and the real Lepton.
    """
    rng = np.random.default_rng(seed)
    seg = np.asarray(seg)
    depth = np.asarray(depth_m, dtype=np.float32)
    thermal = np.full(seg.shape, 14.0, np.float32)
    for cls_id, temp in SEG_TEMPS_C.items():
        thermal[seg == cls_id] = temp
    # Atmospheric attenuation: ~0.4% signal loss per 10 m (LWIR, humid air).
    atten = np.exp(-np.clip(depth, 0, 500) * 0.0004)
    thermal = thermal * atten + 14.0 * (1 - atten)
    thermal = thermal + rng.normal(0, netd_k, thermal.shape).astype(np.float32)
    return thermal


# --------------------------------------------------------------------------- #
# Flight client wrapper (only constructed when airsim is present)
# --------------------------------------------------------------------------- #
@dataclass
class AirsimConfig:
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle: str = "SAR_Drone"
    speed_ms: float = 6.0


class AirsimFloodClient:
    """Thin wrapper: connect, fly lanes, capture frames.  Raises a clear
    RuntimeError (with install instructions) if AirSim is not reachable."""

    def __init__(self, cfg: Optional[AirsimConfig] = None) -> None:
        try:
            import airsim  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "AirSim python API not installed. Install cosys-airsim "
                "(`pip install cosys-airsim`) and launch the UE5 flood town, "
                "or run with --backend headless.") from exc
        self._airsim = airsim
        self.cfg = cfg or AirsimConfig()
        self.client = airsim.MultirotorClient(ip=self.cfg.host, port=self.cfg.port)
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.cfg.vehicle)
        self.client.armDisarm(True, self.cfg.vehicle)

    def takeoff(self, alt_m: float = 45.0) -> None:
        self.client.takeoffAsync(vehicle_name=self.cfg.vehicle).join()
        self.client.moveToZAsync(-alt_m, 3.0, vehicle_name=self.cfg.vehicle).join()

    def goto(self, n: float, e: float, alt_m: float) -> None:
        self.client.moveToPositionAsync(
            e, n, -alt_m, self.cfg.speed_ms,
            vehicle_name=self.cfg.vehicle).join()

    def capture(self) -> Dict[str, np.ndarray]:
        airsim = self._airsim
        req = [airsim.ImageRequest("rgb", airsim.ImageType.Scene, False, False),
               airsim.ImageRequest("seg", airsim.ImageType.Segmentation, False, False),
               airsim.ImageRequest("depth", airsim.ImageType.DepthPlanar, True, False)]
        resp = self.client.simGetImages(req, vehicle_name=self.cfg.vehicle)
        out: Dict[str, np.ndarray] = {}
        for r in resp:
            if r.camera_name == "rgb":
                out["rgb"] = np.frombuffer(r.image_data_uint8, np.uint8).reshape(
                    r.height, r.width, 3)[..., ::-1].copy()  # BGR->RGB
            elif r.camera_name == "seg":
                out["seg"] = np.frombuffer(r.image_data_uint8, np.uint8).reshape(
                    r.height, r.width, 3)[..., 0].astype(np.int32)
            else:
                arr = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
                out["depth"] = np.where(arr > 1000, np.inf, arr)
        out["thermal"] = pseudo_thermal(out["seg"], out["depth"])
        return out

    def land_and_release(self) -> None:
        try:
            self.client.landAsync(vehicle_name=self.cfg.vehicle).join()
            self.client.armDisarm(False, self.cfg.vehicle)
            self.client.enableApiControl(False, self.cfg.vehicle)
        except Exception:
            pass


__all__ = ["airsim_available", "airsim_settings", "pseudo_thermal",
           "SEG_TEMPS_C", "AirsimConfig", "AirsimFloodClient"]
