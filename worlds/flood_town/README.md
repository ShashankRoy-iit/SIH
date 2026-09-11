# `worlds/flood_town/` — one town, three simulators

The flood town is defined **once** in `sar/sim/flood_scene.py`
(`build_flood_town(seed)`) and materialised per backend:

| File | Backend | How |
|---|---|---|
| `settings.json` | AirSim (Cosys-AirSim, UE5) | Copy to `~/Documents/AirSim/settings.json`, launch the map, then `place_town.py` spawns the town |
| `place_town.py` | AirSim | Spawns houses/victims/debris + segmentation IDs; `--dry-run` prints placements anywhere |
| `flood_town.sdf` | Gazebo Harmonic | `gz sim flood_town.sdf -r`, then ArduPilot SITL or MiniSITL |
| *(generated at runtime)* | headless-cinematic | `sar/sim/flood_render.py` — the zero-install fallback |

Regenerate after changing the scene spec:

```bash
python3 -c "
from sar.sim.flood_scene import build_flood_town
from sar.sim.airsim_bridge import airsim_settings
from sar.sim.gazebo_bridge import gazebo_world_sdf
s = build_flood_town()
airsim_settings(s.to_dict(), 'worlds/flood_town/settings.json')
gazebo_world_sdf(s, 'worlds/flood_town/flood_town.sdf')
print('regenerated')"
```

Fly it: `python3 scripts/run_flood_sim.py --backend auto`.
