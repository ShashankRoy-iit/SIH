"""LiDAR behaviour: honest AGL, water widening, and the stop bubble."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sar.hardware.lidar import (AglEstimate, AglFuser, ObstacleGuard, PolarScan,
                                RangeSample, Rangefinder1D, SimulatedLidar)


def test_rangefinder_envelope_gating():
    samples = [RangeSample(t=0, range_m=4.0, quality=200),
               RangeSample(t=1, range_m=9.9, quality=0),     # above max: invalid
               RangeSample(t=2, range_m=0.1, quality=200),     # below min: invalid
               RangeSample(t=3, range_m=4.0, quality=10)]      # weak signal: invalid
    it = iter(samples)
    rf = Rangefinder1D(read=lambda: next(it, None))
    assert rf.poll().valid is True
    assert rf.poll().valid is False
    assert rf.poll().valid is False
    assert rf.poll().valid is False


def test_agl_trusts_lidar_inside_envelope():
    rf = Rangefinder1D(read=lambda: RangeSample(t=0, range_m=5.0, quality=220))
    fuser = AglFuser(rf)
    fuser.update(baro_alt_amsl_m=80.0, terrain_amsl_m=55.0)  # 1st: variance unknown
    est = fuser.update(baro_alt_amsl_m=80.0, terrain_amsl_m=55.0)  # 2nd: trusted
    assert est.source == "lidar" and abs(est.agl_m - 5.0) < 0.01
    assert est.sigma_m < 0.3


def test_agl_falls_back_outside_envelope_with_wide_sigma():
    rf = Rangefinder1D(read=lambda: RangeSample(t=0, range_m=9.9, quality=0, valid=False))
    est = AglFuser(rf).update(baro_alt_amsl_m=100.0, terrain_amsl_m=55.0)
    assert est.source == "baro_terrain" and abs(est.agl_m - 45.0) < 0.01
    assert est.sigma_m > 1.0  # honest: never better than the sensors deserve


def test_water_widens_lidar_sigma():
    def both(wet):
        fuser = AglFuser(Rangefinder1D(
            read=lambda: RangeSample(t=0, range_m=5.0, quality=220)))
        fuser.update(60.0, 55.0, water_likely=wet)
        return fuser.update(60.0, 55.0, water_likely=wet)
    dry, wet = both(False), both(True)
    assert dry.source == wet.source == "lidar"
    assert wet.sigma_m > dry.sigma_m * 2


def test_obstacle_guard_clear_slow_stop():
    guard = ObstacleGuard()
    mk = lambda r: PolarScan(t=0, angles_deg=np.arange(0, 360, 5),
                             ranges_m=np.full(72, r), max_range_m=12.0)
    assert guard.decide(mk(11.0)).action == "clear"
    slow = guard.decide(mk(4.5))
    assert slow.action == "slow" and 0.2 <= slow.scale_velocity <= 1.0
    stop = guard.decide(mk(2.0))
    assert stop.action == "stop" and stop.scale_velocity == 0.0


def test_obstacle_packing_72_bins():
    scan = PolarScan(t=0, angles_deg=np.array([0.0, 90.0]),
                     ranges_m=np.array([5.0, np.inf]), max_range_m=12.0)
    bins = ObstacleGuard.to_mavlink_obstacle_distance(scan)
    assert len(bins) == 72 and bins[0] == 500 and bins[18] == 0  # 0 = unknown


def test_simulated_lidar_range_and_scan():
    sim = SimulatedLidar(agl_fn=lambda t: 45.0, obstacles=[(10.0, 0.0, 0.5)])
    assert sim.range_sample().valid is False  # above the honest envelope
    sim2 = SimulatedLidar(agl_fn=lambda t: 5.0)
    assert abs(sim2.range_sample().range_m - 5.0) < 0.5
    scan = sim.scan(pos_n=0.0, pos_e=0.0, heading_deg=0.0)
    assert scan.closest_m < 12.0  # the pole is seen
