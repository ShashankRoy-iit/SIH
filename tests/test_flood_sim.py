"""Flood town + bridges: determinism, renderer honesty, graceful fallback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sar.sim.airsim_bridge import (airsim_available, airsim_settings, pseudo_thermal)
from sar.sim.flood_render import render_flood_frame
from sar.sim.flood_scene import build_flood_town
from sar.sim.gazebo_bridge import gazebo_available, gazebo_world_sdf


def test_town_is_deterministic():
    a = build_flood_town(seed=7)
    b = build_flood_town(seed=7)
    assert [(s.n, s.e) for s in a.survivors] == [(s.n, s.e) for s in b.survivors]
    assert len(a.survivors) == 12 and len(a.houses) > 10


def test_render_is_deterministic_and_radiometric():
    scene = build_flood_town(seed=7)
    f1 = render_flood_frame(scene, 0, 0, 45.0, seed=1)
    f2 = render_flood_frame(scene, 0, 0, 45.0, seed=1)
    assert np.array_equal(f1["lwir"], f2["lwir"])
    assert f1["lwir"].shape == (240, 320) and f1["rgb"].shape == (480, 640, 3)
    # Radiometric sanity: water cold, bodies warm.
    assert 10.0 < f1["lwir"].min() < 16.0
    assert f1["lwir"].max() > 25.0
    assert 0.05 < f1["gsd_m"] < 0.5


def test_render_truth_matches_painted_survivors():
    scene = build_flood_town(seed=7)
    s = scene.survivors[0]
    f = render_flood_frame(scene, s.n, s.e, 45.0, seed=2)
    assert any(t["sid"] == s.sid for t in f["truth"])


def test_airsim_settings_and_pseudo_thermal():
    from sar.sim.flood_scene import build_flood_town
    s = airsim_settings(build_flood_town(seed=7).to_dict())
    assert s["Vehicles"]["SAR_Drone"]["Sensors"]["lidar"]["SensorType"] == 6
    seg = np.full((48, 64), 10, np.int32)
    seg[20:28, 28:36] = 30  # survivor patch
    depth = np.full((48, 64), 45.0, np.float32)
    th = pseudo_thermal(seg, depth, seed=0)
    assert th[24, 32] > th[0, 0] + 5.0  # body much warmer than water


def test_bridges_degrade_cleanly_without_simulators():
    ok_air, why_air = airsim_available()
    ok_gz, why_gz = gazebo_available()
    # In CI neither is installed; the contract is a clean False + reason.
    assert isinstance(ok_air, bool) and isinstance(why_air, str)
    assert isinstance(ok_gz, bool) and isinstance(why_gz, str)
    if not ok_air:
        try:
            from sar.sim.airsim_bridge import AirsimFloodClient
            AirsimFloodClient()
            raise AssertionError("must raise without the airsim package")
        except RuntimeError as exc:
            assert "cosys-airsim" in str(exc)


def test_gazebo_sdf_contains_town_and_drone(tmp_path=None):
    sdf = gazebo_world_sdf(build_flood_town(seed=7))
    assert "<world name=\"flood_town\">" in sdf
    assert "sar_drone" in sdf and "thermal_camera" in sdf
    assert "victim_V00" in sdf and "house_00" in sdf
