"""Analog FPV + MSP DisplayPort OSD: AI overlay on the pilot's goggles.

The pilot flies by an analog camera + VTX + goggles (5.8 GHz, near-zero
latency, works through the same multipath that kills Wi-Fi).  The autonomy
talks back through the *same* goggles: ArduPilot's MSP DisplayPort driver
(``SERIALx_PROTOCOL=42``, ``OSD_TYPE=5``) renders text pages over the analog
video.  Text only — no bounding boxes — so the overlay protocol is designed
around that limit instead of fighting it:

* Line 1: mode + armed + battery + RSSI/LQ (always).
* Line 2: nearest survivor: bearing, distance, triage tier (when any).
* Line 3: nav health: GPS sats / VIO sigma / denial flag.
* Line 4: AI: detector mode, fps, last veto reason (one line, rotating).

This module formats those pages and rate-limits them (4 Hz max — MSP text
over a 115200-baud AUX serial cannot take more).  The actual byte writer is
injected, so bench tests assert on strings, not on serial ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class OsdState:
    mode: str = "GUIDED"
    armed: bool = False
    batt_v: float = 24.9
    batt_pct: int = 100
    rssi_dbm: int = -70
    lq_pct: int = 100
    sats: int = 0
    gnss_ok: bool = True
    pos_sigma_m: float = 1.0
    survivors: int = 0
    nearest_bearing_deg: Optional[float] = None
    nearest_dist_m: Optional[float] = None
    nearest_tier: str = ""
    detector: str = "auto"
    ai_fps: float = 0.0
    veto_note: str = ""
    alert: str = ""           # CRITICAL line, shown inverted/blinking when set


class OsdFormatter:
    """Format + rate-limit MSP DisplayPort text pages (28x12 typical)."""

    WIDTH = 28
    RATE_HZ = 4.0

    def __init__(self, write: Optional[Callable[[List[str]], None]] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        import time as _t
        self._write = write or (lambda lines: None)
        self._clock = clock or _t.monotonic
        self._last = -1e9
        self.last_lines: List[str] = []
        self.dropped = 0

    # -- page ----------------------------------------------------------- #
    def format(self, s: OsdState) -> List[str]:
        arm = "ARM" if s.armed else "DSARM"
        l1 = f"{s.mode:>8} {arm:<5} {s.batt_v:4.1f}V {s.lq_pct:3d}%"
        if s.survivors > 0 and s.nearest_bearing_deg is not None:
            tier = f" {s.nearest_tier}" if s.nearest_tier else ""
            l2 = (f"SURV {s.survivors:02d} {s.nearest_bearing_deg:03.0f}deg "
                  f"{s.nearest_dist_m or 0:4.0f}m{tier}")
        else:
            l2 = f"SURV {s.survivors:02d}  -- searching --"
        nav = f"GPS{s.sats:02d}" if s.gnss_ok else "DENIED"
        l3 = f"{nav:<7} sig {s.pos_sigma_m:4.1f}m {s.detector}"
        ai = f"AI {s.ai_fps:4.1f}fps"
        veto = (s.veto_note[: self.WIDTH - len(ai) - 1] if s.veto_note else "")
        l4 = f"{ai} {veto}".strip()
        lines = [self._fit(x) for x in (l1, l2, l3, l4)]
        if s.alert:
            lines.insert(0, self._fit(f"!! {s.alert.upper()} !!"))
        return lines

    def _fit(self, s: str) -> str:
        return s[: self.WIDTH].ljust(self.WIDTH)

    # -- rate-limited emit ---------------------------------------------- #
    def emit(self, s: OsdState, force: bool = False) -> bool:
        now = self._clock()
        if not force and (now - self._last) < 1.0 / self.RATE_HZ:
            self.dropped += 1
            return False
        self._last = now
        self.last_lines = self.format(s)
        self._write(self.last_lines)
        return True


__all__ = ["OsdState", "OsdFormatter"]
