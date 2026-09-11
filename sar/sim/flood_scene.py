"""Flood-town scene spec: one description, three simulators.

``FloodScene`` describes a flooded town procedurally — water level, houses,
roads, powerlines, debris, survivors with postures and positions — with a
fixed seed, so AirSim, Gazebo and the headless cinematic renderer all build
*the same town* and results can be compared across backends instead of
argued about.

Coordinate frame: local NED metres about the town centre (home).  x=north,
y=east, z=down (negative = above ground).  Water surface at ``water_level_m``
above the datum ground plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class House:
    n: float
    e: float
    w_m: float = 8.0
    d_m: float = 6.0
    h_m: float = 3.2
    roof_temp_c: float = 30.0     # sun-heated roof (the classic false alarm)
    wall_temp_c: float = 18.0
    flooded: bool = True


@dataclass
class Survivor:
    sid: str
    n: float
    e: float
    posture: str = "lying"        # lying | sitting | standing | wading | clinging
    in_water: bool = True
    on_roof: bool = False
    body_temp_c: float = 32.0
    clothing: Tuple[int, int, int] = (205, 70, 55)  # hi-vis by default
    group: int = 1


@dataclass
class Debris:
    n: float
    e: float
    size_m: float = 1.2
    temp_c: float = 20.0
    kind: str = "timber"          # timber | vehicle | barrel | vegetation


@dataclass
class Powerline:
    n0: float
    e0: float
    n1: float
    e1: float
    h_m: float = 7.0


@dataclass
class FloodScene:
    seed: int = 7
    size_m: float = 220.0         # town is size_m x size_m
    water_level_m: float = 1.2    # flood depth over datum ground
    water_temp_c: float = 15.0
    ground_temp_c: float = 14.0
    night: bool = False
    houses: List[House] = field(default_factory=list)
    survivors: List[Survivor] = field(default_factory=list)
    debris: List[Debris] = field(default_factory=list)
    powerlines: List[Powerline] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "size_m": self.size_m,
                "water_level_m": self.water_level_m, "night": self.night,
                "houses": [h.__dict__ for h in self.houses],
                "survivors": [{**s.__dict__, "clothing": list(s.clothing)}
                              for s in self.survivors],
                "debris": [d.__dict__ for d in self.debris],
                "powerlines": [p.__dict__ for p in self.powerlines]}


def build_flood_town(seed: int = 7, size_m: float = 220.0, night: bool = False,
                     n_survivors: int = 12) -> FloodScene:
    """Procedural flood town.  Deterministic in (seed, size_m, night)."""
    rng = np.random.default_rng(seed)
    scene = FloodScene(seed=seed, size_m=size_m, night=night)
    half = size_m / 2.0

    # Street grid of houses with jitter; ~40% flooded at ground floor.
    step = 34.0
    n0 = -half + 20.0
    while n0 < half - 20.0:
        e0 = -half + 20.0
        while e0 < half - 20.0:
            if rng.random() < 0.78:
                # Most roofs cool (shaded/morning); ~1 in 3 sun-heated into
                # the human band — the adversarial clutter case, kept present
                # but not universal.
                hot = bool(rng.random() < 0.34)
                scene.houses.append(House(
                    n=float(n0 + rng.normal(0, 2.0)),
                    e=float(e0 + rng.normal(0, 2.0)),
                    roof_temp_c=float((26.0 + rng.random() * 6.0) if hot
                                      else (15.0 + rng.random() * 7.0)),
                    flooded=bool(rng.random() < 0.55)))
            e0 += step
        n0 += step * 0.9

    # Survivors: water (wading/clinging), roofs, high ground.
    postures = ["lying", "sitting", "standing", "wading", "clinging"]
    for i in range(n_survivors):
        r = rng.random()
        if r < 0.45:      # in water
            n = float(rng.uniform(-half + 10, half - 10))
            e = float(rng.uniform(-half + 10, half - 10))
            in_water, on_roof = True, False
            posture = "wading" if rng.random() < 0.5 else "clinging"
            temp = float(27.0 + rng.random() * 3.5)
        elif r < 0.65 and scene.houses:   # on a roof
            h = scene.houses[int(rng.integers(0, len(scene.houses)))]
            n, e = h.n + float(rng.normal(0, 1.5)), h.e + float(rng.normal(0, 1.5))
            in_water, on_roof = False, True
            posture = "sitting" if rng.random() < 0.6 else "lying"
            temp = float(31.0 + rng.random() * 2.0)
        else:             # high ground / debris pile
            n = float(rng.uniform(-half + 10, half - 10))
            e = float(rng.uniform(-half + 10, half - 10))
            in_water, on_roof = False, False
            posture = postures[int(rng.integers(0, 3))]
            temp = float(30.0 + rng.random() * 3.0)
        scene.survivors.append(Survivor(
            sid=f"V{i:02d}", n=n, e=e, posture=posture, in_water=in_water,
            on_roof=on_roof, body_temp_c=temp,
            clothing=(205, 70, 55) if rng.random() < 0.7 else (60, 90, 200)))

    # Debris: warm-ish clutter, including two hot decoys (engine bay class).
    for i in range(int(rng.integers(14, 22))):
        scene.debris.append(Debris(
            n=float(rng.uniform(-half, half)), e=float(rng.uniform(-half, half)),
            size_m=float(rng.uniform(0.6, 2.4)),
            temp_c=float(33.0 if i < 2 else 16.0 + rng.random() * 6.0),
            kind="vehicle" if i < 2 else "timber"))

    # Two powerline spans across the town (avoidance + hazard demo).
    scene.powerlines.append(Powerline(-half + 15, -20.0, half - 15, -20.0))
    scene.powerlines.append(Powerline(30.0, -half + 15, 30.0, half - 15))
    return scene


def survivors_in_box(scene: FloodScene, n0: float, n1: float, e0: float, e1: float):
    return [s for s in scene.survivors if n0 <= s.n <= n1 and e0 <= s.e <= e1]


__all__ = ["FloodScene", "House", "Survivor", "Debris", "Powerline",
           "build_flood_town", "survivors_in_box"]
