"""Typed mission events and the append-only mission journal.

The journal is the backbone of SAHYOG's *offline resilience* claim: every
decision-relevant occurrence is written to an append-only JSONL file on the
companion computer's flash **at the moment it happens**, not when the radio
link is available.  When the data link returns, the comms layer drains the
journal in priority order (see :mod:`sar.comms.offline`).

Design notes
------------
* JSONL (one JSON object per line) rather than SQLite: it survives a power cut
  mid-write - at worst the last line is truncated, and the loader skips it.
* Every event carries ``t`` (mission seconds), ``seq`` (monotonic) and
  ``priority`` so the store-and-forward queue can drop low-value traffic under
  bandwidth pressure while never dropping a survivor fix.
"""

from __future__ import annotations

import enum
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


class EventType(str, enum.Enum):
    """Categories of journalled mission events."""

    # lifecycle
    MISSION_START = "mission_start"
    MISSION_END = "mission_end"
    ARMED = "armed"
    DISARMED = "disarmed"
    MODE_CHANGE = "mode_change"
    TAKEOFF = "takeoff"
    LAND = "land"
    RTL = "rtl"
    # navigation / state
    WAYPOINT_REACHED = "waypoint_reached"
    GPS_STATE = "gps_state"
    EK_F_SOURCE_CHANGE = "ekf_source_change"
    POSITION_DEGRADED = "position_degraded"
    OBSTACLE = "obstacle"
    AVOIDANCE = "avoidance"
    # perception
    DETECTION = "detection"
    TRACK_NEW = "track_new"
    TRACK_LOST = "track_lost"
    VICTIM_CONFIRMED = "victim_confirmed"
    HAZARD_FOUND = "hazard_found"
    FALSE_POSITIVE_SUPPRESSED = "false_positive_suppressed"
    # decisions
    TRIAGE_UPDATE = "triage_update"
    ALERT_RAISED = "alert_raised"
    PAYLOAD_DROP = "payload_drop"
    RELAY_DEPLOYED = "relay_deployed"
    REPLAN = "replan"
    # power / safety
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    FAILSAFE = "failsafe"
    # comms
    LINK_LOST = "link_lost"
    LINK_RESTORED = "link_restored"
    # reporting
    REPORT_GENERATED = "report_generated"


class Priority(enum.IntEnum):
    """Store-and-forward transmission priority (higher is sent first).

    The ordering encodes a deliberate operational policy: *a survivor fix is
    worth more than a thousand telemetry frames*.
    """

    DEBUG = 0
    TELEMETRY = 10
    STATUS = 20
    HAZARD = 60
    VICTIM = 90
    ALERT = 95
    SAFETY = 100


# Which event types matter most when the pipe is narrow.
_EVENT_PRIORITY: Dict[EventType, Priority] = {
    EventType.MISSION_START: Priority.STATUS,
    EventType.MISSION_END: Priority.STATUS,
    EventType.ARMED: Priority.STATUS,
    EventType.DISARMED: Priority.STATUS,
    EventType.MODE_CHANGE: Priority.TELEMETRY,
    EventType.TAKEOFF: Priority.STATUS,
    EventType.LAND: Priority.STATUS,
    EventType.RTL: Priority.STATUS,
    EventType.WAYPOINT_REACHED: Priority.TELEMETRY,
    EventType.GPS_STATE: Priority.STATUS,
    EventType.EK_F_SOURCE_CHANGE: Priority.STATUS,
    EventType.POSITION_DEGRADED: Priority.SAFETY,
    EventType.OBSTACLE: Priority.HAZARD,
    EventType.AVOIDANCE: Priority.HAZARD,
    EventType.DETECTION: Priority.TELEMETRY,
    EventType.TRACK_NEW: Priority.HAZARD,
    EventType.TRACK_LOST: Priority.TELEMETRY,
    EventType.VICTIM_CONFIRMED: Priority.VICTIM,
    EventType.HAZARD_FOUND: Priority.HAZARD,
    EventType.FALSE_POSITIVE_SUPPRESSED: Priority.DEBUG,
    EventType.TRIAGE_UPDATE: Priority.ALERT,
    EventType.ALERT_RAISED: Priority.ALERT,
    EventType.PAYLOAD_DROP: Priority.ALERT,
    EventType.RELAY_DEPLOYED: Priority.ALERT,
    EventType.REPLAN: Priority.TELEMETRY,
    EventType.BATTERY_LOW: Priority.SAFETY,
    EventType.BATTERY_CRITICAL: Priority.SAFETY,
    EventType.FAILSAFE: Priority.SAFETY,
    EventType.LINK_LOST: Priority.SAFETY,
    EventType.LINK_RESTORED: Priority.STATUS,
    EventType.REPORT_GENERATED: Priority.STATUS,
}


@dataclass
class Event:
    """One journalled occurrence."""

    t: float
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    priority: Priority = Priority.TELEMETRY
    source: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["priority"] = int(self.priority)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Event":
        return Event(
            t=float(d["t"]),
            type=EventType(d["type"]),
            data=dict(d.get("data", {})),
            seq=int(d.get("seq", 0)),
            priority=Priority(int(d.get("priority", Priority.TELEMETRY))),
            source=str(d.get("source", "system")),
        )


def default_priority(event_type: EventType) -> Priority:
    return _EVENT_PRIORITY.get(event_type, Priority.TELEMETRY)


class MissionJournal:
    """Append-only, thread-safe, crash-tolerant event log.

    Parameters
    ----------
    path : str, optional
        JSONL file to append to.  ``None`` keeps everything in memory only
        (used by unit tests and by the pure in-process simulator).
    persist : bool
        When true, every write is flushed to disk immediately.  This costs
        throughput but guarantees that a mid-mission power loss keeps the
        survivor fixes - which is the whole point of the journal.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        persist: bool = True,
        clock: Optional[Any] = None,
    ) -> None:
        self.path = path
        self.persist = persist
        self.clock = clock
        self._events: List[Event] = []
        self._seq = 0
        self._lock = threading.RLock()
        self._listeners: List[Callable[[Event], None]] = []
        self._fh = None
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            # Resume sequence numbering if the file already exists.
            if os.path.exists(path):
                for ev in MissionJournal.load(path):
                    self._seq = max(self._seq, ev.seq + 1)
            self._fh = open(path, "a", encoding="utf-8")

    # -- subscription ------------------------------------------------------ #
    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    # -- writing ----------------------------------------------------------- #
    def record(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        t: Optional[float] = None,
        priority: Optional[Priority] = None,
        source: str = "system",
    ) -> Event:
        if t is None:
            t = self.clock.now() if self.clock is not None else 0.0
        with self._lock:
            ev = Event(
                t=float(t),
                type=event_type,
                data=dict(data or {}),
                seq=self._seq,
                priority=priority if priority is not None else default_priority(event_type),
                source=source,
            )
            self._seq += 1
            self._events.append(ev)
            if self._fh is not None:
                self._fh.write(json.dumps(ev.to_dict(), default=_json_default) + "\n")
                if self.persist:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
            listeners = list(self._listeners)
        for cb in listeners:
            cb(ev)
        return ev

    # -- reading ----------------------------------------------------------- #
    @property
    def events(self) -> List[Event]:
        with self._lock:
            return list(self._events)

    def since(self, seq: int) -> List[Event]:
        """All events with sequence number > ``seq`` (used by store-and-forward)."""
        with self._lock:
            return [e for e in self._events if e.seq > seq]

    def of_type(self, *types: EventType) -> List[Event]:
        wanted = set(types)
        with self._lock:
            return [e for e in self._events if e.type in wanted]

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    # -- persistence ------------------------------------------------------- #
    @staticmethod
    def load(path: str) -> List[Event]:
        """Load a journal, skipping any truncated trailing line."""
        out: List[Event] = []
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Event.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Truncated final write after a power cut - tolerate it.
                    continue
        out.sort(key=lambda e: (e.t, e.seq))
        return out

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "MissionJournal":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars/arrays used inside event data."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)
