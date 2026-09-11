"""Phone-as-brain behaviour: streams, timeouts, and failure responses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sar.hardware.mobile import (MobileCompanion, PhoneLinkState, PhoneDetection,
                                 SimulatedPhone, SimulatedPhoneConfig, VioSample)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_absent_phone_reports_absent():
    mc = MobileCompanion(clock=Clock())
    st = mc.poll_health()
    assert st.state == PhoneLinkState.ABSENT
    assert not mc.rgb_healthy and not mc.vio_healthy


def test_streaming_phone_reports_streaming():
    clk = Clock()
    mc = MobileCompanion(
        read_frame=lambda: {"image": np.zeros((8, 8, 3), np.uint8), "t": clk()},
        read_vio=lambda: VioSample(t=clk(), n=0, e=0, d=-45, vn=0, ve=0, vd=0),
        clock=clk)
    mc.frame()
    mc.vio()
    st = mc.poll_health(battery_pct=80, temp_c=40)
    assert st.state == PhoneLinkState.STREAMING
    assert mc.rgb_healthy and mc.vio_healthy


def test_stale_vio_degrades_and_failed_after_timeout():
    clk = Clock()
    box = {"v": VioSample(t=100, n=0, e=0, d=-45, vn=0, ve=0, vd=0)}

    def read_vio():
        v, box["v"] = box["v"], None
        return v

    mc = MobileCompanion(read_vio=read_vio, clock=clk)
    mc.vio()
    clk.advance(1.0)
    mc.vio()
    assert mc.poll_health().state == PhoneLinkState.DEGRADED
    assert not mc.vio_healthy
    clk.advance(3.0)
    mc.vio()
    assert mc.poll_health().state == PhoneLinkState.FAILED


def test_overheat_throttles():
    clk = Clock()
    mc = MobileCompanion(
        read_frame=lambda: {"image": np.zeros((4, 4, 3), np.uint8), "t": clk()},
        read_vio=lambda: VioSample(t=clk(), n=0, e=0, d=-45, vn=0, ve=0, vd=0),
        clock=clk)
    mc.frame()
    mc.vio()
    st = mc.poll_health(temp_c=52.0)
    assert st.throttled and st.state == PhoneLinkState.DEGRADED


def test_simulated_phone_vio_drifts_but_bounds():
    truth = lambda t: (t * 2.0, 0.0, -45.0, 2.0, 0.0, 0.0)  # 2 m/s north
    ph = SimulatedPhone(SimulatedPhoneConfig(drift_pct=1.5), truth_fn=truth, seed=3)
    sigmas, errs = [], []
    for i in range(90):  # 3 s at 30 Hz
        v = ph._gen_vio()
        n_true = ph._t * 2.0
        sigmas.append(v.pos_sigma_m)
        errs.append(abs(v.n - n_true))
    # Reported sigma stays conservative (bounds the error on average).
    assert float(np.mean(sigmas)) >= float(np.mean(errs))
    assert max(errs) < 3.0


def test_simulated_phone_frame_and_health():
    ph = SimulatedPhone(SimulatedPhoneConfig(throttle_after_s=10.0), seed=1)
    f = ph.frame()
    assert f["image"].shape == (720, 1280, 3)
    ph.vio()
    st = ph.poll_health()
    assert st.state in (PhoneLinkState.STREAMING, PhoneLinkState.PRESENT)


def test_osd_page_fits_and_rate_limits():
    from sar.hardware.fpv import OsdFormatter, OsdState
    clk = Clock()
    shown = []
    osd = OsdFormatter(write=shown.append, clock=clk)
    lines = osd.format(OsdState(armed=True, survivors=2, nearest_bearing_deg=120,
                                nearest_dist_m=85, nearest_tier="P1"))
    assert all(len(x) == 28 for x in lines)
    assert "SURV 02" in lines[1]
    assert osd.emit(OsdState()) is True
    assert osd.emit(OsdState()) is False  # rate-limited
    assert osd.dropped == 1
    clk.advance(1.0)
    assert osd.emit(OsdState()) is True
