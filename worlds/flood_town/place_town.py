#!/usr/bin/env python3
"""Spawn the shared flood town inside AirSim (Cosys-AirSim / UE5).

Reads the deterministic town from sar/sim/flood_scene.py and spawns one
static actor per house / victim / debris / powerline pole via simSpawnObject,
assigning segmentation IDs that match airsim_bridge.SEG_TEMPS_C.  Run AFTER
starting the UE5 map with worlds/flood_town/settings.json active:

    python3 worlds/flood_town/place_town.py --seed 7

Requires the `cosys-airsim` pip package and a running simulator.  Pure
placement — no flight; fly with scripts/run_flood_sim.py --backend airsim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ASSET = {"house": "House_Simple", "victim": "Victim_Capsule",
         "debris": "Debris_Timber", "vehicle": "Sedan",
         "pole": "PowerlinePole", "water": "WaterPlane_Flood"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spawn flood town in AirSim")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true",
                    help="print placements without connecting")
    args = ap.parse_args(argv)

    from sar.sim.flood_scene import build_flood_town
    scene = build_flood_town(seed=args.seed)
    placements = []
    for i, h in enumerate(scene.houses):
        placements.append(("house", f"house_{i:02d}", h.n, h.e, 0.0, 20))
    for s in scene.survivors:
        z = scene.water_level_m + (0.4 if s.in_water else (3.6 if s.on_roof else 0.5))
        placements.append(("victim", f"victim_{s.sid}", s.n, s.e, z, 30))
    for i, d in enumerate(scene.debris):
        kind = "vehicle" if d.kind == "vehicle" else "debris"
        placements.append((kind, f"debris_{i:02d}", d.n, d.e,
                           scene.water_level_m, 41 if kind == "vehicle" else 40))
    print(f"{len(placements)} placements (seed {args.seed})")
    for p in placements[:8]:
        print("  ", p)
    print("   ...")
    if args.dry_run:
        return 0
    try:
        import airsim
    except Exception:
        print("cosys-airsim not installed; dry-run placements printed above.")
        return 2
    client = airsim.VehicleClient()
    client.confirmConnection()
    for kind, name, n, e, z, seg in placements:
        pose = airsim.Pose(airsim.Vector3r(float(e), float(n), float(-z)),
                           airsim.Quaternionr())
        try:
            obj = client.simSpawnObject(name, ASSET.get(kind, "Debris_Timber"),
                                        pose, airsim.Vector3r(1, 1, 1), False)
            client.simSetSegmentationObjectID(obj, seg, True)
        except Exception as exc:
            print(f"  !! {name}: {exc}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
