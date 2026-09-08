"""Named disaster scenarios and sensor presets.

Everything that varies between a demonstration, a regression test and a Monte
Carlo study lives here as *data*, not as code paths.  A scenario is a
:class:`~sar.sim.world.ScenarioSpec` plus the two camera specs, so one line
reproduces an entire sortie - which is what makes the evaluation numbers in
``docs/`` and ``artifacts/`` repeatable.

Presets
-------
``flood``        river-basin inundation, late afternoon, mixed sun and cloud.
                 The reference case: survivors in water, on rooftops, in mud.
``flood_night``  same basin at 02:00.  RGB is near-blind, so this is the preset
                 that proves the cross-modal veto is conditional rather than
                 unconditional.
``earthquake``   dense urban, collapsed structures, dust, GPS canyon between
                 buildings.  Cold-on-hot survivors: rubble is sun-baked.
``wildfire``     active fire front, heavy smoke, strong thermal contrast and
                 severe RGB degradation.
``landslide``    remote, no roads, steep terrain, scattered survivors.
``stress``       everything at once - rain, smoke, night, GPS denial.  Used by
                 the robustness sweep; it is supposed to look bad, and the
                 point is to show *how* the system degrades rather than that it
                 does not.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

from sar.core.geo import GeoPoint
from sar.sim.renderer import CameraSpec
from sar.sim.world import DisasterWorld, ScenarioSpec

__all__ = [
    "SCENARIOS", "LWIR_SPEC", "RGB_SPEC", "build_reference_scenario",
    "make_specs", "scenario_names",
]


# --------------------------------------------------------------------------- #
# Sensor presets (match config/perception.yaml)
# --------------------------------------------------------------------------- #
LWIR_SPEC = CameraSpec(
    name="lwir", kind="lwir",
    width=320, height=240,                 # simulated; native core is 640x512
    hfov_deg=42.0, mount_pitch_deg=0.0,
    netd_mk=45.0, fps=8.0,
    lwir_extinction_k=0.00035, atm_temp_c=22.0, max_range_m=400.0,
    native_width=640, native_height=512,
)

RGB_SPEC = CameraSpec(
    name="rgb", kind="rgb",
    width=320, height=240,
    hfov_deg=42.0, mount_pitch_deg=0.0,
    rgb_noise_dn=3.0, fps=15.0,
    haze_k=0.0016, max_range_m=400.0,
    native_width=1920, native_height=1080,
)

#: Narrow-FOV "confirmation" lens used by the two-pass search strategy: the
#: broad pass finds candidates, the low narrow pass confirms them.  See
#: docs/05_SEARCH_THEORY.md for the HFOV x altitude trade table.
LWIR_NARROW_SPEC = replace(LWIR_SPEC, name="lwir_narrow", hfov_deg=24.0)
RGB_NARROW_SPEC = replace(RGB_SPEC, name="rgb_narrow", hfov_deg=24.0)


def make_specs(hfov_deg: float | None = None, scale: float = 1.0,
               kind: str = "both") -> Dict[str, CameraSpec]:
    """Build camera specs, optionally overriding FOV and resolution scale.

    ``scale`` multiplies the simulated pixel count relative to the 320x240
    reference.  ``scale=2.0`` reproduces the real 640x512 core at ~5x the
    render cost - useful for a final validation run, not for the loop.
    """
    out: Dict[str, CameraSpec] = {}
    for spec in (LWIR_SPEC, RGB_SPEC):
        if kind != "both" and spec.kind != kind:
            continue
        s = spec
        if hfov_deg:
            s = replace(s, hfov_deg=float(hfov_deg))
        if scale != 1.0:
            s = replace(s, width=max(32, int(round(spec.width * scale))),
                        height=max(32, int(round(spec.height * scale))))
        out[s.kind] = s
    return out


# --------------------------------------------------------------------------- #
# Scenario presets
# --------------------------------------------------------------------------- #
_KOTA = GeoPoint(25.1850, 75.8357, 250.0)      # reference mission origin
SCENARIOS: Dict[str, ScenarioSpec] = {
    # ---- reference flood basin ------------------------------------------- #
    "flood": ScenarioSpec(
        name="flood_basin_pm", disaster="flood", origin=_KOTA,
        extent_north_m=900.0, extent_east_m=900.0, grid_res_m=2.0,
        n_victims=12, n_distractors=8, n_buildings=44,
        flood_fraction=0.46, fire_sites=1, landslide_sites=0,
        downed_powerlines=2, wind_speed_ms=5.0, wind_direction_deg=210.0,
        time_of_day_h=16.5, cloud_cover=0.35, rain=0.0, smoke_density=0.22,
        vegetation_fraction=0.18, base_satellites=34,
        victim_prior_weights={"water_edge": 1.6, "rooftop": 1.4, "road": 1.2},
    ),
    # ---- night: RGB is blind, LWIR must carry ---------------------------- #
    "flood_night": ScenarioSpec(
        name="flood_basin_night", disaster="flood", origin=_KOTA,
        extent_north_m=900.0, extent_east_m=900.0, grid_res_m=2.0,
        n_victims=12, n_distractors=8, n_buildings=44,
        flood_fraction=0.46, fire_sites=2, landslide_sites=0,
        downed_powerlines=2, wind_speed_ms=3.0, wind_direction_deg=200.0,
        time_of_day_h=2.0, cloud_cover=0.55, rain=0.0, smoke_density=0.10,
        vegetation_fraction=0.18, base_satellites=30,
        victim_prior_weights={"water_edge": 1.6, "rooftop": 1.4, "road": 1.2},
    ),
    # ---- dense urban earthquake, GPS canyon ------------------------------ #
    "earthquake": ScenarioSpec(
        name="urban_quake", disaster="earthquake", origin=_KOTA,
        extent_north_m=800.0, extent_east_m=800.0, grid_res_m=2.0,
        n_victims=14, n_distractors=10, n_buildings=120,
        flood_fraction=0.02, fire_sites=3, landslide_sites=1,
        downed_powerlines=4, wind_speed_ms=2.5, wind_direction_deg=90.0,
        time_of_day_h=11.0, cloud_cover=0.10, rain=0.0, smoke_density=0.45,
        vegetation_fraction=0.06, base_satellites=32,
        gps_canyon_zones=[(150.0, 150.0, 550.0, 550.0)],
        victim_prior_weights={"structure": 1.9, "road": 1.2},
    ),
    # ---- active wildfire -------------------------------------------------- #
    "wildfire": ScenarioSpec(
        name="wildfire_front", disaster="wildfire", origin=_KOTA,
        extent_north_m=1200.0, extent_east_m=1200.0, grid_res_m=2.0,
        n_victims=9, n_distractors=7, n_buildings=18,
        flood_fraction=0.01, fire_sites=7, landslide_sites=0,
        downed_powerlines=3, wind_speed_ms=9.0, wind_direction_deg=250.0,
        time_of_day_h=15.0, cloud_cover=0.20, rain=0.0, smoke_density=0.75,
        vegetation_fraction=0.48, base_satellites=33,
        victim_prior_weights={"road": 1.8, "dry_refuge": 1.4},
    ),
    # ---- remote landslide ------------------------------------------------- #
    "landslide": ScenarioSpec(
        name="hillside_landslide", disaster="landslide", origin=_KOTA,
        extent_north_m=1000.0, extent_east_m=1000.0, grid_res_m=2.0,
        n_victims=8, n_distractors=6, n_buildings=10,
        flood_fraction=0.05, fire_sites=0, landslide_sites=4,
        downed_powerlines=1, wind_speed_ms=6.0, wind_direction_deg=180.0,
        time_of_day_h=9.5, cloud_cover=0.60, rain=0.35, smoke_density=0.05,
        vegetation_fraction=0.35, base_satellites=28,
        gps_denial_zones=[(200.0, 200.0, 420.0, 420.0)],
        victim_prior_weights={"structure": 1.7, "road": 1.2},
    ),
    # ---- everything at once: the robustness sweep ------------------------- #
    "stress": ScenarioSpec(
        name="compound_stress", disaster="compound", origin=_KOTA,
        extent_north_m=900.0, extent_east_m=900.0, grid_res_m=2.0,
        n_victims=14, n_distractors=14, n_buildings=90,
        flood_fraction=0.30, fire_sites=4, landslide_sites=2,
        downed_powerlines=5, wind_speed_ms=11.0, wind_direction_deg=230.0,
        time_of_day_h=3.0, cloud_cover=0.85, rain=0.70, smoke_density=0.60,
        vegetation_fraction=0.25, base_satellites=19,
        gps_denial_zones=[(100.0, 100.0, 500.0, 500.0)],
        gps_canyon_zones=[(520.0, 120.0, 820.0, 420.0)],
        victim_prior_weights={"water_edge": 1.5, "structure": 1.6, "rooftop": 1.3},
    ),
}

#: Small fast worlds used by unit tests (a few seconds to build, not minutes).
SMOKE_SCENARIOS: Dict[str, ScenarioSpec] = {
    k: replace(v, extent_north_m=min(v.extent_north_m, 320.0),
               extent_east_m=min(v.extent_east_m, 320.0),
               grid_res_m=4.0,
               n_victims=min(v.n_victims, 5),
               n_distractors=min(v.n_distractors, 4),
               n_buildings=min(v.n_buildings, 14))
    for k, v in SCENARIOS.items()
}


def scenario_names(smoke: bool = False) -> Tuple[str, ...]:
    return tuple((SMOKE_SCENARIOS if smoke else SCENARIOS).keys())


def get_spec(name: str, smoke: bool = False, seed: int | None = None,
             **overrides: Any) -> ScenarioSpec:
    """Look up a preset, optionally shrinking it (tests) or overriding fields."""
    table = SMOKE_SCENARIOS if smoke else SCENARIOS
    if name not in table:
        raise KeyError(f"unknown scenario '{name}'; have {sorted(table)}")
    spec = table[name]
    if seed is not None:
        spec = replace(spec, seed=int(seed))
    if overrides:
        spec = replace(spec, **overrides)
    return spec


def build_reference_scenario(name: str = "flood", seed: int | None = None,
                             smoke: bool = False,
                             hfov_deg: float | None = None,
                             scale: float = 1.0,
                             **overrides: Any
                             ) -> Tuple[DisasterWorld, CameraSpec, CameraSpec]:
    """Construct ``(world, lwir_spec, rgb_spec)`` for a named preset.

    This is the single entry point used by ``scripts/*``, ``tests/*`` and the
    mission runner, so a scenario identifier plus a seed is enough to reproduce
    any published result exactly.
    """
    spec = get_spec(name, smoke=smoke, seed=seed, **overrides)
    world = DisasterWorld(spec)
    specs = make_specs(hfov_deg=hfov_deg, scale=scale, kind="both")
    return world, specs["lwir"], specs["rgb"]
