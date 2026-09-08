"""Simulation time sources.

Two clocks are provided:

* :class:`SimClock` - deterministic, stepped by the simulator.  Every module
  in the stack reads time from a clock object rather than ``time.time()`` so
  that a full mission can be replayed bit-for-bit (essential for regression
  testing an autonomy stack).
* :class:`WallClock` - real wall time, used when flying a physical vehicle or
  a real ArduPilot SITL instance.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Anything that can report monotonic seconds."""

    def now(self) -> float: ...

    def millis(self) -> int: ...

    def micros(self) -> int: ...


class SimClock:
    """Virtual clock advanced explicitly by the simulator."""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def millis(self) -> int:
        return int(self._t * 1_000)

    def micros(self) -> int:
        return int(self._t * 1_000_000)

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("SimClock cannot run backwards")
        self._t += float(dt)
        return self._t

    def set(self, t: float) -> None:
        self._t = float(t)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SimClock(t={self._t:.3f})"


class WallClock:
    """Monotonic real-time clock."""

    __slots__ = ("_t0",)

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def millis(self) -> int:
        return int(self.now() * 1_000)

    def micros(self) -> int:
        return int(self.now() * 1_000_000)

    def advance(self, dt: float) -> float:  # pragma: no cover - API parity
        return self.now()

    def __repr__(self) -> str:  # pragma: no cover
        return f"WallClock(t={self.now():.3f})"
