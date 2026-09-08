"""Hardware layer: the safety supervisor and camera synchronisation.

The supervisor is the only component whose failure is unrecoverable, so these
tests are written as operational scenarios rather than as method calls: a
battery that is fine as a percentage but cannot reach home, a latch that must
not clear itself, a watchdog that fires when the autonomy loop stops feeding it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sar.hardware.camera import (CameraPair, FrameCapture, ground_sample_distance)
from sar.hardware.safety import (SafetyLimits, SafetyState, SafetySupervisor,
                                 _point_in_polygon)


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def fresh() -> SafetySupervisor:
    sup = SafetySupervisor(SafetyLimits(max_range_m=500.0, reserve_fraction=0.20,
                                        hover_power_w=400.0))
    sup.note_progress()
    return sup


def test_nominal_when_everything_is_fine():
    sup = fresh()
    d = sup.evaluate(position_ned=(50.0, 50.0, -40.0), battery_remaining=0.85,
                     agl_m=40.0, telemetry_silence_s=0.1)
    assert d.state is SafetyState.NOMINAL
    assert not d.must_return


def test_battery_percentage_can_be_fine_while_the_aircraft_cannot_get_home():
    """The rule that a percentage-based failsafe cannot express."""
    sup = fresh()
    d = sup.evaluate(position_ned=(450.0, 0.0, -60.0),   # 450 m out
                     battery_remaining=0.45,             # 'plenty' by percentage
                     battery_wh_remaining=8.0,           # ...but 8 Wh is not enough
                     agl_m=60.0)
    assert d.state is SafetyState.RTL_NOW
    assert any("needed to reach home" in r for r in d.reasons)


def test_energy_to_return_grows_with_distance():
    sup = fresh()
    near = sup.energy_to_return_wh((50.0, 0.0, -40.0), (0.0, 0.0, 0.0))
    far = sup.energy_to_return_wh((700.0, 0.0, -40.0), (0.0, 0.0, 0.0))
    assert far > near * 2.5


def test_range_limit_triggers_rtl():
    sup = fresh()
    d = sup.evaluate(position_ned=(600.0, 0.0, -50.0), agl_m=50.0)
    assert d.state is SafetyState.RTL_NOW
    assert "range" in " ".join(d.reasons)


def test_state_latches_and_only_an_operator_clears_it():
    sup = fresh()
    sup.evaluate(position_ned=(600.0, 0.0, -50.0), agl_m=50.0)
    assert sup.state is SafetyState.RTL_NOW
    # A single good sample must not resume the search.
    d = sup.evaluate(position_ned=(10.0, 0.0, -50.0), agl_m=50.0)
    assert d.state is SafetyState.RTL_NOW
    sup.clear("test")
    assert sup.state is SafetyState.NOMINAL


def test_watchdog_fires_when_the_autonomy_loop_stalls():
    sup = SafetySupervisor(SafetyLimits(max_loop_stall_s=0.05))
    sup.note_progress()
    time.sleep(0.08)
    d = sup.evaluate(position_ned=(0.0, 0.0, -20.0))
    assert d.state is SafetyState.RTL_NOW
    assert any("has not progressed" in r for r in d.reasons)


def test_rc_loss_is_a_caution_not_an_abort():
    """Autonomous search is precisely the mission that must survive RC loss."""
    sup = fresh()
    d = sup.evaluate(position_ned=(20.0, 20.0, -40.0), rc_loss_s=30.0, agl_m=40.0)
    assert d.state is SafetyState.CAUTION
    assert not d.must_stop_searching


def test_denial_is_bounded_by_policy():
    sup = SafetySupervisor(SafetyLimits(max_denial_s=1.0))
    sup.note_progress()
    t = time.monotonic()
    sup.evaluate(position_ned=(0.0, 0.0, -30.0), gps_denied=True, now=t)
    d = sup.evaluate(position_ned=(0.0, 0.0, -30.0), gps_denied=True, now=t + 2.0)
    assert d.state is SafetyState.RTL_NOW
    assert any("GNSS denied" in r for r in d.reasons)


def test_position_sigma_beyond_actionable_triggers_return():
    sup = SafetySupervisor(SafetyLimits(max_position_sigma_m=10.0))
    sup.note_progress()
    d = sup.evaluate(position_ned=(0.0, 0.0, -30.0), position_sigma_m=25.0)
    assert d.state is SafetyState.RETURN
    assert any("not be actionable" in r for r in d.reasons)


def test_geofence_polygon_is_enforced():
    poly = [(-100.0, -100.0), (-100.0, 100.0), (100.0, 100.0), (100.0, -100.0)]
    sup = SafetySupervisor(SafetyLimits(geofence_polygon=poly, max_range_m=10_000))
    sup.note_progress()
    assert sup.evaluate(position_ned=(0.0, 0.0, -30.0)).state is SafetyState.NOMINAL
    assert sup.evaluate(position_ned=(150.0, 0.0, -30.0)).state is SafetyState.RTL_NOW


def test_point_in_polygon_handles_the_boundary_case():
    poly = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    assert _point_in_polygon((5.0, 5.0), poly)
    assert not _point_in_polygon((15.0, 5.0), poly)


def test_every_decision_carries_its_reason_and_measurements():
    sup = fresh()
    d = sup.evaluate(position_ned=(600.0, 0.0, -50.0), agl_m=50.0,
                     battery_remaining=0.5)
    assert d.reasons and d.measurements
    assert "range_m" in d.measurements
    assert sup.summary()["triggered_rules"]


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
class FakeSource:
    def __init__(self, kind: str, t: float, value: float = 20.0) -> None:
        self.kind, self.name = kind, f"fake-{kind}"
        self.t, self.value = t, value
        self.frames = 0
        self.dropped = 0

    def open(self) -> None:
        pass

    def read(self):
        self.frames += 1
        return FrameCapture(image=np.full((60, 80), self.value, np.float32),
                            kind=self.kind, t=self.t, camera=self.name,
                            meta={"hfov_deg": 57.0})

    def close(self) -> None:
        pass


def pair_with_skew(skew_s: float, max_skew_s: float = 0.08) -> CameraPair:
    pair = CameraPair(lwir=FakeSource("lwir", 100.0),
                      rgb=FakeSource("rgb", 100.0 + skew_s),
                      max_skew_s=max_skew_s)
    # Drive the pump synchronously rather than starting threads: the test is
    # about the pairing rule, not about threading.
    pair._latest["lwir"] = pair.lwir.read()
    pair._latest["rgb"] = pair.rgb.read()
    return pair


def test_frames_within_the_skew_budget_are_paired():
    pair = pair_with_skew(0.02)
    frames = pair.latest_pair(agl_m=45.0)
    assert {f.kind for f in frames} == {"lwir", "rgb"}
    assert pair.stats.pairs == 1
    assert pair.stats.worst_skew_ms == pytest.approx(20.0, abs=1.0)


def test_frames_outside_the_budget_degrade_to_thermal_only():
    """Silently fusing a 300 ms-old RGB frame is worse than not fusing at all."""
    pair = pair_with_skew(0.30)
    frames = pair.latest_pair(agl_m=45.0)
    assert [f.kind for f in frames] == ["lwir"]
    assert pair.stats.rejected_skew == 1
    assert pair.stats.pairs == 0


def test_gsd_is_computed_from_geometry_not_guessed():
    pair = pair_with_skew(0.01)
    frames = pair.latest_pair(agl_m=50.0)
    expected = ground_sample_distance(50.0, 57.0, 80)
    assert frames[0].gsd_m == pytest.approx(expected, rel=1e-6)
    assert 0.5 < frames[0].gsd_m < 1.0     # 80 px across a 54 m swath


def test_off_nadir_tilt_increases_gsd():
    nadir = ground_sample_distance(50.0, 57.0, 320, tilt_deg=0.0)
    tilted = ground_sample_distance(50.0, 57.0, 320, tilt_deg=45.0)
    assert tilted > nadir * 1.3


def test_camera_health_is_reportable():
    pair = pair_with_skew(0.01)
    pair.latest_pair(agl_m=45.0)
    health = pair.health()
    assert health["sync"]["pairs"] == 1
    assert health["lwir"]["frames"] == 1
