"""Cinematic headless flood renderer: the demo that runs anywhere.

Renders a nadir (straight-down) RGB + LWIR frame pair from a ``FloodScene``
for a camera at (n, e, agl_m) — water with sun glints, houses with warm
roofs, debris, powerlines, and survivors as thermal disks + clothing blobs.
It is schematic, not photoreal: its contract is *radiometric honesty*
(temperatures in C, sub-pixel area blending, NETD noise) and *determinism*
(seed + pose = pixels), so detector numbers measured on it are comparable
across runs and machines.

This is the fallback behind AirSim and Gazebo in ``run_flood_sim.py``: the
jury-laptop path with zero installs beyond numpy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from sar.sim.flood_scene import FloodScene


def render_flood_frame(scene: FloodScene, cam_n: float, cam_e: float, agl_m: float,
                       lwir_size: Tuple[int, int] = (320, 240),
                       rgb_size: Tuple[int, int] = (640, 480),
                       hfov_deg: float = 57.0, seed: int = 0) -> Dict[str, Any]:
    """Render one nadir frame pair.  Returns dict with lwir (C), rgb, gsd_m,
    water_frac and the survivors inside the footprint (ground truth)."""
    rng = np.random.default_rng(seed)
    lw, lh = lwir_size
    rw, rh = rgb_size
    gsd = 2.0 * agl_m * math.tan(math.radians(hfov_deg / 2.0)) / lw
    swath_n = gsd * lh
    swath_e = gsd * lw
    n0, e0 = cam_n - swath_n / 2.0, cam_e - swath_e / 2.0

    def to_px(n: float, e: float, w: int, h: int):
        u = (e - e0) / swath_e * w
        v = (n - n0) / swath_n * h
        return u, v

    # --- LWIR: ground + water + roofs ---------------------------------- #
    lwir = np.full((lh, lw), scene.ground_temp_c, np.float32)
    water = _water_field(scene, cam_n, cam_e, swath_n, swath_e, lw, lh)
    lwir[water] = scene.water_temp_c
    # Large-scale ground texture (smooth patches, NOT pixel noise: pixel noise
    # would trip the RGB texture rules and the water-smoothness gate).
    yy_t, xx_t = np.mgrid[0:lh, 0:lw]
    tex = (np.sin(xx_t / 23.0 + seed) + np.sin(yy_t / 17.0)
           + np.sin((xx_t + yy_t) / 31.0)) / 3.0  # -1..1, smooth
    lwir += (tex * 0.5).astype(np.float32)
    # Sun glints on water: sparse hot specks (the flicker test's prey).
    glint = water & (rng.random((lh, lw)) < 0.0004)
    lwir[glint] = 34.0
    rgb = np.full((rh, rw, 3), (108, 102, 94), np.uint8)
    rgb_water = _water_field(scene, cam_n, cam_e, swath_n, swath_e, rw, rh)
    rgb[rgb_water] = (78, 90, 104)
    # Smooth brightness patches + faint pixel grain (kept small so the
    # water-smoothness gate, the debris-texture rule and the water mask all
    # still behave: local std stays well under their thresholds).
    yy_r, xx_r = np.mgrid[0:rh, 0:rw]
    patch = (np.sin(xx_r / 47.0 + seed * 0.7) + np.sin(yy_r / 39.0)) / 2.0
    grain = rng.normal(0, 1.6, (rh, rw))
    shade = (patch * 7.0 + grain).astype(np.float32)
    shade[rgb_water] *= 0.4  # water stays glassy
    rgb = np.clip(rgb.astype(np.float32) + shade[..., None], 0, 255).astype(np.uint8)
    # Glints in RGB too (bright specks, kept sparse like the LWIR ones).
    rgb_glint = rgb_water & (rng.random((rh, rw)) < 0.0003)
    rgb[rgb_glint] = (235, 240, 245)

    for h in scene.houses:
        u, v = to_px(h.n, h.e, lw, lh)
        hw, hh = h.w_m / gsd / 2.0, h.d_m / gsd / 2.0
        x0, x1 = int(max(u - hw, 0)), int(min(u + hw, lw))
        y0, y1 = int(max(v - hh, 0)), int(min(v + hh, lh))
        if x1 > x0 and y1 > y0:
            roof = h.roof_temp_c * (0.55 if scene.night else 1.0)
            lwir[y0:y1, x0:x1] = roof + rng.normal(0, 0.4, (y1 - y0, x1 - x0))
            ru, rv = to_px(h.n, h.e, rw, rh)
            rhw, rhh = h.w_m / (swath_e / rw) / 2.0, h.d_m / (swath_n / rh) / 2.0
            rx0, rx1 = int(max(ru - rhw, 0)), int(min(ru + rhw, rw))
            ry0, ry1 = int(max(rv - rhh, 0)), int(min(rv + rhh, rh))
            if rx1 > rx0 and ry1 > ry0:
                col = (150, 142, 132) if not scene.night else (40, 42, 50)
                rgb[ry0:ry1, rx0:rx1] = col
                # Roof ridge + outline so houses read as structures, not slabs.
                rgb[ry0:ry0 + 2, rx0:rx1] = (110, 104, 96)
                rgb[ry1 - 2:ry1, rx0:rx1] = (110, 104, 96)
                rgb[ry0:ry1, rx0:rx0 + 2] = (110, 104, 96)
                rgb[ry0:ry1, rx1 - 2:rx1] = (110, 104, 96)
                mid = (ry0 + ry1) // 2
                rgb[mid:mid + 1, rx0:rx1] = (128, 120, 110)
    for d in scene.debris:
        u, v = to_px(d.n, d.e, lw, lh)
        r = max(d.size_m / gsd / 2.0, 1.0)
        yy, xx = np.mgrid[0:lh, 0:lw]
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
        lwir[disk] = d.temp_c

    # --- survivors ------------------------------------------------------ #
    visible = []
    for s in scene.survivors:
        u, v = to_px(s.n, s.e, lw, lh)
        if not (-10 <= u < lw + 10 and -10 <= v < vh(h := lh) + 10):
            continue
        frac = {"lying": 0.9, "sitting": 0.7, "standing": 1.0,
                "wading": 0.18, "clinging": 0.15}[s.posture]
        area_m2 = 0.62 * frac
        r_px = max(math.sqrt(area_m2 / math.pi) / gsd, 0.8)
        yy, xx = np.mgrid[0:lh, 0:lw]
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= r_px * r_px
        if not disk.any():
            continue
        bg_here = scene.water_temp_c if s.in_water else scene.ground_temp_c
        cover = min(1.0, math.pi * r_px * r_px / max(disk.sum(), 1))
        peak = s.body_temp_c * cover + bg_here * (1 - cover) if r_px < 1.6 else s.body_temp_c
        lwir[disk] = peak
        if not s.in_water or s.posture == "standing":
            ru, rv = to_px(s.n, s.e, rw, rh)
            # Clothing fills the body footprint (factor 1.1, not 0.8): a smaller
            # patch under-fills the geometry-aware aperture and is wrongly gated.
            rr = max(int(r_px * rw / lw * 1.1), 3)
            x0, x1 = int(max(ru - rr, 0)), int(min(ru + rr, rw))
            y0, y1 = int(max(rv - rr, 0)), int(min(rv + rr, rh))
            if x1 > x0 and y1 > y0 and not scene.night:
                # Shaded, not flat: real clothing has folds and a lit centre.
                # A flat plateau peaks at its CORNER, which under-fills the
                # geometry aperture; shading centres the peak honestly.
                patch = np.ones((y1 - y0, x1 - x0, 1), np.float32)
                py, px = np.mgrid[0:y1 - y0, 0:x1 - x0]
                cy, cx = (y1 - y0 - 1) / 2.0, (x1 - x0 - 1) / 2.0
                rr = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                shade = 1.0 - 0.18 * np.clip(rr / max(max(cx, cy), 1), 0, 1)
                rgb[y0:y1, x0:x1] = (np.asarray(s.clothing, np.float32)
                                     * shade[..., None]).astype(np.uint8)
        visible.append({"sid": s.sid, "u": round(float(u), 1), "v": round(float(v), 1),
                        "in_water": s.in_water, "on_roof": s.on_roof,
                        "posture": s.posture, "peak_c": round(float(peak), 2),
                        "area_px": int(disk.sum()),
                        "resolvable": bool(peak >= 23.0 and disk.sum() >= 2)})

    lwir += rng.normal(0, 0.045, lwir.shape).astype(np.float32)  # NETD 45 mK
    if scene.night:
        rgb = (rgb.astype(np.float32) * 0.12).astype(np.uint8)
    truth = [s for s in visible]
    return {"lwir": lwir, "rgb": rgb, "gsd_m": gsd,
            "footprint": (n0, n0 + swath_n, e0, e0 + swath_e),
            "water_frac": float(water.mean()), "truth": truth}


def vh(x: int) -> int:  # tiny helper kept module-level for pickling safety
    return x


def _water_field(scene: FloodScene, cam_n: float, cam_e: float,
                 swath_n: float, swath_e: float, w: int, h: int) -> np.ndarray:
    """Water covers low ground: a flooded river band + pond noise, fixed seed."""
    yy, xx = np.mgrid[0:h, 0:w]
    n = cam_n - swath_n / 2.0 + (yy / h) * swath_n
    e = cam_e - swath_e / 2.0 + (xx / w) * swath_e
    # A meandering flooded channel + broad flood sheet on the east side.
    band = np.abs(e - 18.0 * np.sin(n / 55.0)) < (26.0 + 8.0 * np.sin(n / 31.0))
    sheet = (e > 30.0) & (n > -scene.size_m / 2.0)
    rng = np.random.default_rng(scene.seed)
    ponds = ((np.sin(n / 23.0 + 1.7) + np.sin(e / 19.0)) > 1.15)
    return (band | sheet | ponds)


__all__ = ["render_flood_frame"]
