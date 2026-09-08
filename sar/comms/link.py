"""Offline-first data link: priority-tiered store-and-forward.

The operational constraint this module exists for
-------------------------------------------------
A search aircraft over a flood or an earthquake is, most of the time, out of
reach of whatever network the command centre is on.  Cellular is down - that is
what the disaster did.  Satellite is expensive, slow to acquire and often
occluded by the very terrain that caused the problem.  What is left is a
line-of-sight radio: the ELRS control link at a few hundred baud of telemetry,
maybe a 900 MHz LoRa data link at a few kilobit, maybe a Wi-Fi bridge when the
aircraft happens to be near the incident command post.

Every one of those is asymmetric, bursty, and interrupted without warning.  So
the aircraft cannot be designed around "send the detection when it happens".  It
has to be designed around "the detection is recorded locally the instant it
happens, and reaches the ground when the link allows, in an order that reflects
what matters".

Two properties follow, and they are the ones tested here.

**Nothing is lost because the link was down.**  The queue is bounded by bytes,
not by time, and eviction is by priority tier and age rather than FIFO - so a
link outage costs latency for bulk data and never costs a survivor.

**A survivor is one alert, not two hundred.**  A person in frame for 30 seconds
at 10 Hz is 300 detections of the same person.  Transmitting each one would
saturate a LoRa link with redundancy while a second, unsent survivor waited
behind it in the queue.  So the link layer coalesces by track identity: one
``DISCOVER`` alert on confirmation, then ``UPDATE`` messages only when something
a responder would act on has changed - position beyond the reported error
ellipse, priority tier, viability, or group size.  This is the single largest
lever on how many survivors a low-bandwidth sortie can actually report.

The tiers
---------
``LIFE_SAFETY``   A confirmed survivor, or a hazard that is about to become one.
                  Never evicted by a lower tier.  Sent first, always.
``TACTICAL``      Track updates, hazard polygons, landing zones.  Things the
                  sortie itself needs acknowledged.
``SITUATIONAL``   Coverage map, belief state, nav quality.  Useful for
                  reconstructing what the aircraft did; not needed live.
``BULK``          Imagery thumbnails, full logs.  Sent only when there is spare
                  capacity, and the first thing dropped when there is not.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("sar.comms")

__all__ = [
    "PriorityTier", "CommsProfile", "TRANSPORTS", "Message", "MessageKind",
    "StoreAndForwardQueue", "LinkSimulator", "Coalescer", "TelemetryUplink",
    "LinkReport",
]


# --------------------------------------------------------------------------- #
# Priorities and message kinds
# --------------------------------------------------------------------------- #
class PriorityTier(enum.IntEnum):
    """Ordered so that a plain ``<`` comparison is an eviction decision.

    Lower value means more urgent.  ``IntEnum`` rather than ``Enum`` because the
    queue sorts by it and because a tier that cannot be compared is a tier that
    needs a lookup table at every call site.
    """

    LIFE_SAFETY = 0
    TACTICAL = 1
    SITUATIONAL = 2
    BULK = 3

    @property
    def label(self) -> str:
        return self.name.lower()


class MessageKind(str, enum.Enum):
    """What a message *is*, which decides its tier and its coalescing key."""

    SURVIVOR_DISCOVER = "survivor_discover"
    SURVIVOR_UPDATE = "survivor_update"
    #: The rich detail behind an alert - viability breakdown, landing zones,
    #: nearby hazards.  Deliberately a *separate, lower-tier* message: the alert
    #: has to fit the narrowest radio the airframe carries, and none of this
    #: belongs in it.  A responder needs "person, here, now, this urgent" to
    #: act; the rest is for planning and can wait for a wider link.
    SURVIVOR_DETAIL = "survivor_detail"
    HAZARD_ALERT = "hazard_alert"
    HAZARD_UPDATE = "hazard_update"
    TRACK_STATE = "track_state"
    LANDING_ZONE = "landing_zone"
    NAV_STATUS = "nav_status"
    COVERAGE = "coverage"
    VEHICLE_TELEMETRY = "vehicle_telemetry"
    IMAGE_THUMBNAIL = "image_thumbnail"
    EVENT_LOG = "event_log"

    @property
    def tier(self) -> PriorityTier:
        return _KIND_TIER[self]

    @property
    def wire_code(self) -> str:
        """Two-character discriminator used on the wire.

        ``"survivor_discover"`` costs 22 bytes; ``"sd"`` costs 2.  On a 222-byte
        LoRa packet that 20 bytes is 9% of the message, and it is the difference
        between an alert that fits with its hazard field intact and one that has
        to drop it.  Readable names stay everywhere else - in the queue, in logs,
        in the dashboard - and only the serialised form is abbreviated.
        """
        return _WIRE_CODE[self]

    @classmethod
    def from_wire_code(cls, code: str) -> "MessageKind":
        try:
            return _KIND_BY_CODE[code]
        except KeyError:
            raise ValueError(
                f"unknown wire code {code!r}; known: {sorted(_KIND_BY_CODE)}")

    @property
    def coalesces(self) -> bool:
        """Whether repeated instances replace rather than accumulate.

        A survivor alert does: the ground wants the current best answer about
        *that person*, not a history of every frame they appeared in.  An event
        log does not - each entry is a distinct fact and dropping one loses
        information.
        """
        return self in _COALESCING_KINDS


_KIND_TIER = {
    MessageKind.SURVIVOR_DISCOVER: PriorityTier.LIFE_SAFETY,
    MessageKind.SURVIVOR_UPDATE: PriorityTier.LIFE_SAFETY,
    MessageKind.SURVIVOR_DETAIL: PriorityTier.TACTICAL,
    MessageKind.HAZARD_ALERT: PriorityTier.LIFE_SAFETY,
    MessageKind.LANDING_ZONE: PriorityTier.TACTICAL,
    MessageKind.HAZARD_UPDATE: PriorityTier.TACTICAL,
    MessageKind.TRACK_STATE: PriorityTier.TACTICAL,
    MessageKind.NAV_STATUS: PriorityTier.SITUATIONAL,
    MessageKind.VEHICLE_TELEMETRY: PriorityTier.SITUATIONAL,
    MessageKind.COVERAGE: PriorityTier.SITUATIONAL,
    MessageKind.IMAGE_THUMBNAIL: PriorityTier.BULK,
    MessageKind.EVENT_LOG: PriorityTier.BULK,
}

_WIRE_CODE = {
    MessageKind.SURVIVOR_DISCOVER: "sd",
    MessageKind.SURVIVOR_UPDATE: "su",
    MessageKind.SURVIVOR_DETAIL: "sx",
    MessageKind.HAZARD_ALERT: "ha",
    MessageKind.HAZARD_UPDATE: "hu",
    MessageKind.TRACK_STATE: "ts",
    MessageKind.LANDING_ZONE: "lz",
    MessageKind.NAV_STATUS: "nv",
    MessageKind.COVERAGE: "cv",
    MessageKind.VEHICLE_TELEMETRY: "vt",
    MessageKind.IMAGE_THUMBNAIL: "it",
    MessageKind.EVENT_LOG: "el",
}
_KIND_BY_CODE = {v: k for k, v in _WIRE_CODE.items()}
assert len(_KIND_BY_CODE) == len(_WIRE_CODE), "wire codes must be unique"
assert len(_WIRE_CODE) == len(MessageKind), "every kind needs a wire code"

_COALESCING_KINDS = frozenset({
    MessageKind.SURVIVOR_UPDATE, MessageKind.HAZARD_UPDATE,
    MessageKind.TRACK_STATE, MessageKind.NAV_STATUS,
    MessageKind.VEHICLE_TELEMETRY, MessageKind.COVERAGE,
})


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CommsProfile:
    """One radio link, characterised by what it actually costs to use.

    ``bitrate`` is the *sustained application-layer* rate after framing and
    duty-cycle limits, not the PHY rate a datasheet quotes.  For LoRa those
    differ by an order of magnitude and quoting the PHY rate is how a design
    ends up assuming it can stream video over it.
    """

    name: str
    bitrate_bps: float
    latency_s: float
    loss_pct: float
    max_packet_bytes: int
    #: Whether the link can carry anything at all when the aircraft is beyond
    #: line of sight.  A satellite link cannot be modelled as "slow" - it is
    #: absent, and absence has to be a distinct state or the queue never
    #: exercises its store-and-forward path.
    requires_los: bool = True
    half_duplex: bool = True
    #: Whether this link can carry a survivor alert at all.  Distinct from
    #: bitrate: the ELRS control link is up whenever the aircraft is in range and
    #: is the most reliable thing on the airframe, but its 64-byte CRSF payload
    #: cannot hold even a stripped alert.  Calling it a data link anyway is how a
    #: design ends up assuming survivors are reported over the RC channel.
    #: LoRa is the *minimum* alert-capable link, and that is a hardware decision
    #: with a cost - it is the reason the airframe carries a second radio.
    carries_alerts: bool = True
    notes: str = ""

    def bytes_per_second(self) -> float:
        return self.bitrate_bps / 8.0

    def seconds_for(self, nbytes: int) -> float:
        """Air time for one packet, including the per-packet overhead."""
        overhead = 24                            # header + CRC + preamble
        return (nbytes + overhead) / max(self.bytes_per_second(), 1e-6)


#: The links this airframe actually has, worst to best.
#:
#: The ordering matters more than the numbers: a design that works on
#: ``ELRS_TELEMETRY`` works on everything above it, and a design that only works
#: on ``WIFI_BRIDGE`` fails in precisely the scenario the project exists for.
TRANSPORTS: Dict[str, CommsProfile] = {
    # ELRS carries MAVLink telemetry on the control link.  It is always up when
    # the aircraft is in range, and it is almost nowhere near enough for
    # anything but short structured alerts.
    "elrs_telemetry": CommsProfile(
        name="elrs_telemetry", bitrate_bps=2400.0, latency_s=0.12, loss_pct=0.5,
        max_packet_bytes=64, requires_los=True, carries_alerts=False,
        notes="RadioMaster Pocket ELRS, MAVLink on the control link.  ~300 B/s "
              "sustained, 64 B CRSF payload.  Control, arming, mode, and short "
              "status only - it is NOT a data link.  A survivor alert does not "
              "fit one packet and fragmenting it over a control channel is the "
              "wrong trade: the aircraft carries LoRa for that."),
    # A 900 MHz LoRa data link is the honest answer for over-the-horizon
    # reporting: kilometres of range at a rate that suits structured alerts and
    # nothing else.
    "lora_900": CommsProfile(
        name="lora_900", bitrate_bps=2900.0, latency_s=0.85, loss_pct=2.0,
        max_packet_bytes=222, requires_los=False,
        notes="SF9/125 kHz, ~10% duty cycle -> ~290 B/s application layer.  "
              "Non-line-of-sight capable, which is the whole reason to carry it."),
    # Wi-Fi to an incident command post.  High rate, short range, and it exists
    # only when someone has driven a vehicle with a mast to the scene.
    "wifi_bridge": CommsProfile(
        name="wifi_bridge", bitrate_bps=8_000_000.0, latency_s=0.02,
        loss_pct=0.2, max_packet_bytes=1500, requires_los=True,
        notes="1 Mb/s sustained assumed, not the PHY rate.  Imagery and full "
              "logs go here, and only here."),
    # What the aircraft has when nothing else is in range.  Modelled as zero
    # rather than as a very slow link, because "no link" and "slow link" need
    # different behaviour: the first must buffer indefinitely, the second must
    # shed load.
    "out_of_range": CommsProfile(
        name="out_of_range", bitrate_bps=0.0, latency_s=float("inf"),
        loss_pct=100.0, max_packet_bytes=0, requires_los=True,
        carries_alerts=False,
        notes="Not a transport.  The state the aircraft is in for most of a "
              "real sortie, and the reason the queue exists."),
}


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    """One queued item.

    ``body`` is kept as a dict and serialised on demand rather than stored as
    bytes, because the coalescer has to compare payloads to decide whether an
    update is worth sending, and comparing two JSON blobs that differ only in
    key order is a false negative every time.
    """

    kind: MessageKind
    body: Dict[str, Any]
    #: Identity for coalescing - a track id, a hazard cell, "nav".  Two messages
    #: with the same key and a coalescing kind replace each other.
    key: str = ""
    tier: PriorityTier = PriorityTier.SITUATIONAL
    created_at: float = field(default_factory=time.time)
    #: After this the message is stale and dropped unsent.  A survivor alert
    #: from an hour ago is still worth delivering; a coverage map from an hour
    #: ago describes a sortie that has already landed.
    ttl_s: Optional[float] = None
    sent_at: Optional[float] = None
    attempts: int = 0
    #: Monotonic within a key, so the ground can discard an out-of-order update.
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            self.key = self.kind.value

    @property
    def payload(self) -> Dict[str, Any]:
        """The wire form: self-describing, versioned, and compact.

        ``tier`` is not on the wire because it is a pure function of ``kind`` -
        sending it would spend bytes restating something the receiver already
        knows.  On a 222-byte link that is not a rounding error, it is a field
        that decides whether the message fits at all.
        """
        return {
            "k": self.kind.wire_code, "i": self.key, "r": self.revision,
            "t": int(self.created_at), "d": self.body,
        }

    def size_bytes(self) -> int:
        """Serialised length, which is what the queue and the air-time model use.

        Computed rather than cached because coalescing mutates ``body`` in place,
        and a cached size that has gone stale is how a queue overruns its byte
        budget without ever appearing to.
        """
        try:
            return len(json.dumps(self.payload, separators=(",", ":"),
                                  default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return 4096

    #: Fields a responder would act on, in both the compact wire spelling and
    #: the verbose one used by in-process bodies.  Listed together because a
    #: fingerprint that only knows one spelling matches nothing in the other: it
    #: hashes an empty dict, returns the same value every time, and suppresses
    #: either everything or nothing depending on which path built the message.
    _ACTIONABLE = ("la", "lat", "lo", "lon", "sg", "sigma_m", "pr", "priority",
                   "gs", "group_size", "vi", "viability", "nb", "needs", "tc",
                   "time_critical_s", "label", "severity", "hazard_class",
                   "state", "n_survivors")

    #: Fields that move every frame without meaning anything changed.  Excluded
    #: from the fallback fingerprint so a message is not permanently "new".
    _VOLATILE = frozenset({"t", "tm", "rev", "_rev", "_rx_t", "_kind", "_tier",
                           "timestamp", "queue_bytes"})

    def fingerprint(self) -> str:
        """Content hash over the fields a responder would act on.

        Used to suppress an update that has not changed anything meaningful.
        Deliberately excludes timestamps, confidence and observation count, all
        of which move every frame and would defeat the suppression entirely.
        """
        interesting = {k: v for k, v in self.body.items() if k in self._ACTIONABLE}
        if not interesting:
            # No field this kind considers actionable, so fall back to the whole
            # body minus the volatile parts.  Without this the fingerprint is a
            # hash of an empty dict - the same value forever - and the first
            # transmission of a kind permanently suppresses every later one.  A
            # vehicle telemetry message whose position and battery never "change"
            # is not a stable vehicle, it is a fingerprint that cannot see it.
            interesting = {k: v for k, v in self.body.items()
                           if k not in self._VOLATILE}
        # Position is quantised to the reported error: an update that moves the
        # marker by less than its own uncertainty ellipse is not new information,
        # and sending it costs air time that a second survivor needs.
        sig = float(self.body.get("sg", self.body.get("sigma_m", 5.0)) or 5.0)
        q = max(sig, 1.0)
        for (lat_k, lon_k) in (("la", "lo"), ("lat", "lon")):
            if lat_k in interesting and lon_k in interesting:
                interesting[lat_k] = round(float(interesting[lat_k]) * 111320.0 / q)
                interesting[lon_k] = round(float(interesting[lon_k]) * 111320.0 / q)
        blob = json.dumps(interesting, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "Message":
        """Rebuild a message from its wire form.

        Present so the ground side is not left hand-rolling the inverse of
        :attr:`payload`, and so a round-trip is testable: an encoder with no
        decoder next to it drifts, and the drift shows up as a dashboard that
        silently stops understanding one message kind.
        """
        kind = MessageKind.from_wire_code(payload["k"])
        return cls(kind=kind, body=dict(payload.get("d") or {}),
                   key=str(payload.get("i", "")), tier=kind.tier,
                   created_at=float(payload.get("t", time.time())),
                   revision=int(payload.get("r", 0)))

    def expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_s is None:
            return False
        return (now or time.time()) - self.created_at > self.ttl_s


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #
@dataclass
class QueueStats:
    enqueued: int = 0
    sent: int = 0
    dropped_expired: int = 0
    dropped_evicted: int = 0
    coalesced: int = 0
    suppressed_duplicate: int = 0
    failed: int = 0
    bytes_sent: int = 0
    bytes_dropped: int = 0
    peak_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["delivery_ratio"] = round(self.sent / max(1, self.enqueued), 4)
        return d


class StoreAndForwardQueue:
    """Byte-bounded priority queue with tier-aware eviction.

    The eviction rule is the part that matters.  When the queue is full
    something has to go, and the wrong choice is silent: a FIFO queue drops the
    *newest* message, which during a long outage is the survivor just found,
    while retaining an hour-old coverage map that nobody needs.  This one drops
    the least important thing that is oldest within its tier, and refuses to
    evict ``LIFE_SAFETY`` at all unless there is nothing else left - and if it
    does have to, it says so in the log, because that is a sortie that has found
    more survivors than it can report and the operator needs to know.
    """

    def __init__(self, max_bytes: int = 192 * 1024,
                 max_items: int = 2000) -> None:
        self.max_bytes = int(max_bytes)
        self.max_items = int(max_items)
        self.stats = QueueStats()
        self._items: List[Message] = []
        self._lock = threading.RLock()
        self._by_key: Dict[Tuple[str, str], int] = {}
        self._current_bytes = 0
        self._revision: Dict[str, int] = {}
        #: Fingerprint of the last message actually handed to the link, per key.
        #: Without this, suppression only works against a message still *queued*,
        #: so a healthy link that drains instantly coalesces nothing: a survivor
        #: in frame for 100 seconds produces ~100 transmissions at ~400 bytes
        #: each, which on a 2.9 kbit LoRa link is 110 seconds of air time for one
        #: person - more than the entire sortie budget, and it crowds out every
        #: other survivor found in the same pass.
        self._sent_fp: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def bytes_queued(self) -> int:
        with self._lock:
            return self._current_bytes

    def _order(self) -> None:
        """Sort so that ``pop(0)`` yields the next thing to send.

        Tier first, then age within a tier.  Age ascending means the *oldest*
        survivor alert goes first, which is right: it has been waiting longest
        and the responder's clock on it started when it was found, not when it
        was transmitted.
        """
        self._items.sort(key=lambda m: (int(m.tier), m.created_at))

    def enqueue(self, msg: Message) -> bool:
        """Add a message, coalescing if an equivalent one is already queued.

        Returns True if the message was added or replaced something, False if it
        was suppressed as a duplicate of what is already queued.
        """
        now = time.time()
        with self._lock:
            self._purge_expired(now)

            # Coalesce: replace an unsent message of the same kind and key.  A
            # survivor found once and updated twenty times before the link comes
            # up should occupy one slot, not twenty.
            ck = (msg.kind.value, msg.key)
            if msg.kind.coalesces:
                fp = msg.fingerprint()
                if self._sent_fp.get(ck) == fp:
                    # The ground already has exactly this.  Not a duplicate in
                    # the queue - a duplicate in the world.
                    self.stats.suppressed_duplicate += 1
                    return False
            if msg.kind.coalesces and ck in self._by_key:
                idx = self._by_key[ck]
                old = self._items[idx]
                if old.fingerprint() == msg.fingerprint():
                    self.stats.suppressed_duplicate += 1
                    return False
                msg.revision = old.revision + 1
                msg.created_at = min(old.created_at, msg.created_at)
                self._current_bytes -= old.size_bytes()
                self._items[idx] = msg
                self.stats.coalesced += 1
            else:
                self._revision[msg.key] = self._revision.get(msg.key, 0) + 1
                msg.revision = self._revision[msg.key]
                self._items.append(msg)
                self._by_key[ck] = len(self._items) - 1
                self._current_bytes += msg.size_bytes()
                self.stats.enqueued += 1

            self._order()
            self._by_key = {(m.kind.value, m.key): i
                            for i, m in enumerate(self._items)}
            self._evict_if_needed()
            self.stats.peak_bytes = max(self.stats.peak_bytes,
                                        self._current_bytes)
            return True

    def _evict_if_needed(self) -> None:
        while (self._current_bytes > self.max_bytes
               or len(self._items) > self.max_items) and self._items:
            victim = self._pick_victim()
            if victim is None:
                return
            self._remove(victim)
            self.stats.dropped_evicted += 1
            self.stats.bytes_dropped += victim.size_bytes()
            if victim.tier == PriorityTier.LIFE_SAFETY:
                log.error("EVICTED A LIFE_SAFETY MESSAGE (%s key=%s): the queue "
                          "is full of survivor alerts and cannot report them all. "
                          "Raise max_bytes or add a link.",
                          victim.kind.value, victim.key)

    def _pick_victim(self) -> Optional[Message]:
        """Least important, oldest within that tier.  LIFE_SAFETY is last resort."""
        for tier in (PriorityTier.BULK, PriorityTier.SITUATIONAL,
                     PriorityTier.TACTICAL, PriorityTier.LIFE_SAFETY):
            candidates = [m for m in self._items if m.tier == tier]
            if candidates:
                # Oldest first within the tier, except LIFE_SAFETY where the
                # newest is more actionable than one found an hour ago.
                return (min(candidates, key=lambda m: m.created_at)
                        if tier != PriorityTier.LIFE_SAFETY
                        else max(candidates, key=lambda m: m.created_at)
                        if len(candidates) > 1 else candidates[0])
        return None

    def _remove(self, msg: Message) -> None:
        try:
            self._items.remove(msg)
        except ValueError:
            return
        self._current_bytes -= msg.size_bytes()
        self._by_key = {(m.kind.value, m.key): i
                        for i, m in enumerate(self._items)}

    def _purge_expired(self, now: float) -> None:
        stale = [m for m in self._items if m.expired(now)]
        for m in stale:
            self._remove(m)
            self.stats.dropped_expired += 1

    # ------------------------------------------------------------------ #
    def next_batch(self, budget_bytes: int, mtu_bytes: Optional[int] = None,
                   now: Optional[float] = None) -> List[Message]:
        """Take as much as fits in ``budget_bytes``, highest priority first.

        The caller supplies the budget because only the link knows how much air
        time it has.  Taking a batch rather than one message at a time is what
        lets a bursty link - LoRa with a duty-cycle restriction, say - fill a
        whole window in one go instead of paying latency per message.

        Two rules keep this from deadlocking, and both were found the hard way.

        A message is never skipped for being larger than the *whole* budget.  On
        a 2.4 kbit link a 0.25 s window is 75 bytes and a survivor alert is 395,
        so a strict budget means the highest-priority message in the system can
        never be transmitted and the queue wedges permanently at full capacity.
        The first message always goes; the credit accounting in the caller then
        runs negative and throttles the next window, which is the correct way to
        express "this cost more air time than we had saved up".

        A message larger than the link's MTU is never taken at all.  That one is
        not transient: retrying a 1500-byte thumbnail over a 64-byte radio fails
        identically forever, and requeueing it puts it back at the head where it
        blocks everything behind it.  The caller must split it or drop it.
        """
        now = now if now is not None else time.time()
        with self._lock:
            self._purge_expired(now)
            self._order()
            out: List[Message] = []
            used = 0
            for m in list(self._items):
                sz = m.size_bytes()
                if mtu_bytes is not None and sz > mtu_bytes:
                    continue
                if out and used + sz > budget_bytes:
                    # Keep scanning rather than stopping: a large BULK item must
                    # not block a small LIFE_SAFETY one queued behind it.
                    continue
                out.append(m)
                used += sz
                if len(out) >= 32:
                    break
            for m in out:
                self._remove(m)
                m.attempts += 1
            return out

    def mark_sent(self, msg: Message) -> None:
        """Record that the ground now has this content for ``msg.key``.

        Called on delivery rather than on dequeue, so a message lost in
        transmission still counts as unsent and is retried - remembering it as
        sent on the way out would suppress the retry and lose the update.
        """
        if msg.kind.coalesces:
            with self._lock:
                self._sent_fp[(msg.kind.value, msg.key)] = msg.fingerprint()

    def requeue(self, msg: Message) -> None:
        """Put a message back after a failed transmission."""
        with self._lock:
            if msg not in self._items:
                self._items.append(msg)
                self._current_bytes += msg.size_bytes()
                self.stats.failed += 1
                self._order()
                self._by_key = {(m.kind.value, m.key): i
                                for i, m in enumerate(self._items)}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            by_tier: Dict[str, int] = {}
            bytes_tier: Dict[str, int] = {}
            for m in self._items:
                k = m.tier.label
                by_tier[k] = by_tier.get(k, 0) + 1
                bytes_tier[k] = bytes_tier.get(k, 0) + m.size_bytes()
            return {
                "items": len(self._items), "bytes": self._current_bytes,
                "max_bytes": self.max_bytes,
                "by_tier": by_tier, "bytes_by_tier": bytes_tier,
                "oldest_age_s": (round(time.time() - self._items[0].created_at, 1)
                                 if self._items else 0.0),
                "next": (self._items[0].kind.value if self._items else None),
                "stats": self.stats.to_dict(),
            }


# --------------------------------------------------------------------------- #
# The link
# --------------------------------------------------------------------------- #
class LinkSimulator:
    """Models what a radio does to a batch of bytes, including not working.

    Deliberately separate from the queue: the queue's job is deciding *what* to
    send and in what order, and the link's job is deciding whether it arrives.
    Conflating them makes it impossible to test the buffering behaviour, because
    every failure looks like a queue bug.

    Range is modelled explicitly rather than as a loss percentage, because the
    two have different consequences.  A 2% loss rate means retries work; being
    9 km beyond a 5 km radio means nothing works and the only correct behaviour
    is to store.  A sortie that never leaves range never exercises the code path
    the whole design exists for.
    """

    def __init__(self, profile: Optional[CommsProfile] = None,
                 seed: int = 5, range_m: Optional[float] = None) -> None:
        self.profile = profile if profile is not None else TRANSPORTS["lora_900"]
        # stdlib RNG rather than numpy: this module has no arrays in it, and
        # pulling numpy in for one uniform draw would make the comms layer
        # unimportable in a bare environment for no benefit.
        self.rng = random.Random(seed)
        self.range_m = range_m
        self.in_range = True
        self.sent_packets = 0
        self.sent_bytes = 0
        self.failed_packets = 0
        self.air_time_s = 0.0
        self.outage_s = 0.0
        self._outage_since: Optional[float] = None

    # ------------------------------------------------------------------ #
    def set_range(self, distance_m: float) -> bool:
        """Update range state.  Returns the new ``in_range``."""
        was = self.in_range
        if self.range_m is None:
            self.in_range = True
        else:
            self.in_range = distance_m <= self.range_m
        now = time.time()
        if self.in_range and not was:
            if self._outage_since is not None:
                gap = now - self._outage_since
                self.outage_s += gap
                self._outage_since = None
                log.info("link %s reacquired at %.0f m after %.1f s of outage",
                         self.profile.name, distance_m, gap)
        elif not self.in_range and was:
            self._outage_since = now
            log.info("link %s lost at %.0f m (range %.0f m)",
                     self.profile.name, distance_m, self.range_m or 0.0)
        elif not self.in_range and self._outage_since is None:
            self._outage_since = now
        return self.in_range

    def budget_bytes(self, window_s: float) -> int:
        """How much can go out in ``window_s`` seconds of air time."""
        if not self.in_range or self.profile.bitrate_bps <= 0:
            return 0
        return max(0, int(self.profile.bytes_per_second() * window_s))

    def transmit(self, msgs: Sequence[Message]) -> Tuple[List[Message], List[Message]]:
        """Send a batch.  Returns ``(delivered, failed)``.

        Failures are per-packet and independent, which is optimistic for a bursty
        channel and pessimistic for a faded one; both are wrong in a way that
        averages out over a sortie, and neither is worth the complexity of a
        Gilbert-Elliott model for testing queue behaviour.
        """
        delivered: List[Message] = []
        failed: List[Message] = []
        if not self.in_range or self.profile.bitrate_bps <= 0:
            return [], list(msgs)
        p_loss = min(0.95, max(0.0, self.profile.loss_pct / 100.0))
        for m in msgs:
            sz = m.size_bytes()
            if sz > self.profile.max_packet_bytes:
                # Too big for this link at all.  This is not a transient failure
                # and retrying it forever would starve the queue, so it is
                # reported back to the caller to be split or downgraded.
                log.debug("message %s (%d B) exceeds %s MTU %d B",
                          m.kind.value, sz, self.profile.name,
                          self.profile.max_packet_bytes)
                failed.append(m)
                continue
            self.air_time_s += self.profile.seconds_for(sz)
            if self._random() < p_loss:
                self.failed_packets += 1
                failed.append(m)
            else:
                self.sent_packets += 1
                self.sent_bytes += sz
                m.sent_at = time.time() + self.profile.latency_s
                delivered.append(m)
        return delivered, failed

    def _random(self) -> float:
        return self.rng.random()

    def report(self) -> Dict[str, Any]:
        if self._outage_since is not None:
            self.outage_s += time.time() - self._outage_since
            self._outage_since = time.time()
        total = self.sent_packets + self.failed_packets
        return {
            "transport": self.profile.name,
            "bitrate_bps": self.profile.bitrate_bps,
            "in_range": self.in_range,
            "range_m": self.range_m,
            "sent_packets": self.sent_packets,
            "sent_bytes": self.sent_bytes,
            "failed_packets": self.failed_packets,
            "packet_success_pct": round(100.0 * self.sent_packets / max(1, total), 2),
            "air_time_s": round(self.air_time_s, 2),
            "outage_s": round(self.outage_s, 2),
        }


# --------------------------------------------------------------------------- #
# The uplink
# --------------------------------------------------------------------------- #
@dataclass
class LinkReport:
    """What the ground station has received so far, for a dashboard."""

    survivors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hazards: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    nav: Dict[str, Any] = field(default_factory=dict)
    vehicle: Dict[str, Any] = field(default_factory=dict)
    received: int = 0
    last_rx_t: float = 0.0

    def apply(self, msg: Message) -> None:
        """Fold one received message into the ground picture."""
        self.received += 1
        self.last_rx_t = time.time()
        body = dict(msg.body)
        # The key travels with the body.  It is the survivor's identity across
        # revisions, and without it the dashboard has to invent one, which means
        # a marker changes label every time an update arrives and the operator
        # cannot tell one person from another.
        body["_key"] = msg.key
        body["_kind"] = msg.kind.value
        body["_rev"] = msg.revision
        body["_tier"] = msg.tier.label
        body["_rx_t"] = self.last_rx_t
        if msg.kind in (MessageKind.SURVIVOR_DISCOVER,
                        MessageKind.SURVIVOR_UPDATE):
            prev = self.survivors.get(msg.key)
            # Reject an out-of-order update: the queue can reorder across a
            # requeue, and a stale position overwriting a fresh one is worse
            # than no update at all.
            if prev and int(prev.get("_rev", 0)) > msg.revision:
                return
            if prev:
                body["_first_rx_t"] = prev.get("_first_rx_t", body["_rx_t"])
                body["_n_updates"] = int(prev.get("_n_updates", 1)) + 1
            else:
                body["_first_rx_t"] = body["_rx_t"]
                body["_n_updates"] = 1
            self.survivors[msg.key] = body
        elif msg.kind in (MessageKind.HAZARD_ALERT, MessageKind.HAZARD_UPDATE):
            self.hazards[msg.key] = body
        elif msg.kind == MessageKind.COVERAGE:
            self.coverage = body
        elif msg.kind == MessageKind.NAV_STATUS:
            self.nav = body
        elif msg.kind == MessageKind.VEHICLE_TELEMETRY:
            self.vehicle = body

    def to_dict(self) -> Dict[str, Any]:
        return {
            "received": self.received,
            "n_survivors": len(self.survivors),
            "n_hazards": len(self.hazards),
            "survivors": list(self.survivors.values()),
            "hazards": list(self.hazards.values()),
            "coverage": self.coverage, "nav": self.nav, "vehicle": self.vehicle,
        }


class TelemetryUplink:
    """Pumps a queue over a link, on a thread, at a rate the link can sustain.

    Owns the timing so the mission loop does not have to: the sortie calls
    :meth:`publish_survivor` when it finds someone and never thinks about
    bandwidth again.  Everything about *when* that reaches the ground is this
    class's problem.
    """

    def __init__(self, queue: Optional[StoreAndForwardQueue] = None,
                 link: Optional[LinkSimulator] = None,
                 window_s: float = 1.0,
                 on_deliver: Optional[Callable[[Message], None]] = None) -> None:
        # `queue or StoreAndForwardQueue()` looks equivalent and is not: the
        # queue defines __len__, so an *empty* queue is falsy and the caller's
        # queue is silently replaced by a private one.  Every publish then goes
        # into an object nobody reads, and the summary reports an empty system
        # that is in fact working perfectly.  Explicit None checks throughout.
        self.queue = queue if queue is not None else StoreAndForwardQueue()
        self.link = link if link is not None else LinkSimulator()
        self.window_s = float(window_s)
        self.on_deliver = on_deliver
        self.report = LinkReport()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._wake = threading.Event()
        # Air-time credit, accumulated between pumps rather than reset each one.
        # A per-window budget that does not carry over underuses a link that was
        # idle, and one that never allows overshoot cannot send a message bigger
        # than a window.  Credit does both: it accrues while nothing is queued
        # and goes negative when a survivor alert costs more than was saved.
        self._credit_bytes = 0.0
        self.max_credit_s = 2.0
        self._last_pump = time.time()

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="comms-uplink")
        self._thread.start()
        log.info("uplink started on %s (%.0f bps)", self.link.profile.name,
                 self.link.profile.bitrate_bps)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump_once()
            except Exception as exc:                  # pragma: no cover
                log.exception("uplink error: %r", exc)
            self._wake.wait(timeout=self.window_s)
            self._wake.clear()

    def pump_once(self, now: Optional[float] = None) -> int:
        """Move as much as the link allows right now.  Returns the count sent."""
        now = now if now is not None else time.time()
        dt = max(0.0, now - self._last_pump)
        self._last_pump = now
        if self.link.in_range and self.link.profile.bitrate_bps > 0:
            self._credit_bytes += self.link.profile.bytes_per_second() * dt
            cap = self.link.profile.bytes_per_second() * self.max_credit_s
            self._credit_bytes = min(self._credit_bytes, cap)
        if self._credit_bytes < 1.0:
            return 0
        batch = self.queue.next_batch(int(self._credit_bytes),
                                      mtu_bytes=self.link.profile.max_packet_bytes)
        if not batch:
            return 0
        self._credit_bytes -= sum(m.size_bytes() for m in batch)
        delivered, failed = self.link.transmit(batch)
        for m in failed:
            self.queue.requeue(m)
        for m in delivered:
            self.queue.mark_sent(m)
            with self._lock:
                self.report.apply(m)
            if self.on_deliver is not None:
                try:
                    self.on_deliver(m)
                except Exception as exc:              # pragma: no cover
                    log.warning("on_deliver callback failed: %r", exc)
        return len(delivered)

    # ------------------------------------------------------------------ #
    # Publishing helpers - what the mission loop actually calls
    # ------------------------------------------------------------------ #
    #: The alert must fit the narrowest *alert-capable* radio the airframe
    #: carries, or the offline-first claim is false in exactly the conditions it
    #: was designed for.  Derived from the transport table rather than
    #: hardcoded, so adding a narrower radio makes the budget tighten and the
    #: test that guards it start failing - which is the point.
    ALERT_BUDGET_BYTES = min(
        (t.max_packet_bytes for t in TRANSPORTS.values()
         if t.carries_alerts and t.bitrate_bps > 0 and t.max_packet_bytes > 0),
        default=222) - 22                            # envelope headroom

    #: Fields dropped, in this order, if the alert still does not fit.  Ordered
    #: by what a responder can least afford to lose last: position and urgency
    #: are never dropped, because without them the alert is not actionable.
    _ALERT_DROP_ORDER = ("hz", "lz", "vi", "nb", "cf", "no", "sg")

    def publish_survivor(self, assessment: Any, *,
                         first_report: bool = True,
                         key: Optional[str] = None) -> bool:
        """Queue a survivor from a :class:`~sar.perception.SurvivorAssessment`.

        ``first_report`` selects DISCOVER over UPDATE.  The distinction is worth
        keeping even though both are LIFE_SAFETY: the ground station's clock on a
        survivor starts at DISCOVER, and an UPDATE for someone never announced
        would be silently dropped as an out-of-order revision.

        Two messages go out, not one.  The alert carries the minimum a responder
        needs to act and is sized to fit the narrowest link; the detail carries
        everything else at a lower tier, so a saturated LoRa link still reports
        survivors while the richer picture waits for a wider one.
        """
        d = assessment.to_dict() if hasattr(assessment, "to_dict") \
            else dict(assessment)
        # "s12" rather than "survivor:12": the key travels in every packet and
        # the prefix is pure overhead once the kind code already says what it is.
        #
        # A caller may override it, and the mission runner does: it keys by
        # survivor *location* rather than by track id.  That is what makes report
        # merging real.  Merging only the publish decision while keeping a
        # per-track key would still land one ground-station entry per fragment,
        # and the command centre would show eight markers for one person with a
        # log that says the system merged them.
        key = key if key is not None else f"s{d.get('track_id', 0)}"
        kind = (MessageKind.SURVIVOR_DISCOVER if first_report
                else MessageKind.SURVIVOR_UPDATE)

        body = self._compact_alert(d)
        probe = Message(kind=kind, body=body, key=key, tier=kind.tier)
        for field_name in self._ALERT_DROP_ORDER:
            if probe.size_bytes() <= self.ALERT_BUDGET_BYTES:
                break
            body.pop(field_name, None)
            probe = Message(kind=kind, body=body, key=key, tier=kind.tier)
        if probe.size_bytes() > self.ALERT_BUDGET_BYTES:
            # Should be unreachable - position, priority and group size alone are
            # well under budget - but a silent oversize alert is the one failure
            # here that would strand a survivor, so it is loud.
            log.error("survivor alert for %s is %d B, over the %d B budget even "
                      "after dropping optional fields; it will not fit a narrow "
                      "link", key, probe.size_bytes(), self.ALERT_BUDGET_BYTES)

        # A survivor alert has no TTL: if the link comes back three hours later,
        # the ground still needs to know.  Stale position is better than no
        # position, and the revision number says which one it is.
        ok = self.queue.enqueue(Message(kind=kind, body=body, key=key,
                                        tier=kind.tier, ttl_s=None))

        detail = self._survivor_detail(d)
        if detail:
            self.queue.enqueue(Message(
                kind=MessageKind.SURVIVOR_DETAIL, body=detail,
                key=f"{key}:detail", tier=PriorityTier.TACTICAL, ttl_s=1800.0))
        return ok

    @staticmethod
    def _compact_alert(d: Dict[str, Any]) -> Dict[str, Any]:
        """Minimum actionable content about one survivor.

        Short keys and rounded values throughout.  Five decimal degrees is about
        1.1 m, which is an order of magnitude finer than the reported position
        sigma, so rounding loses nothing a responder could use - and it is the
        difference between fitting a LoRa packet and not.
        """
        tc = d.get("time_critical_s")
        return {
            "la": round(float(d.get("lat", 0.0)), 5),
            "lo": round(float(d.get("lon", 0.0)), 5),
            "sg": round(float(d.get("sigma_m", 0.0)), 1),
            "pr": str(d.get("priority", ""))[:1].upper(),
            "gs": int(d.get("group_size", 1) or 1),
            "tc": int(tc) if tc else None,
            "xm": 1 if d.get("cross_modal") else 0,
            "nb": str(d.get("needs", ""))[:1].upper(),
            "cf": round(float(d.get("confidence", 0.0)), 2),
            "no": int(d.get("n_obs", 0) or 0),
            "tm": round(float(d.get("last_seen_t") or d.get("t_mission") or 0.0), 1),
            "vi": (d.get("viability") or {}).get("band")
                  or (d.get("viability") or {}).get("score"),
            "lz": [z.get("north_m") for z in (d.get("landing_zones") or [])[:1]],
            "hz": [h.get("hazard_class") for h in (d.get("hazards_nearby") or [])[:1]],
        }

    @staticmethod
    def _survivor_detail(d: Dict[str, Any]) -> Dict[str, Any]:
        """Everything about a survivor that is useful but not immediately actionable."""
        return {
            "track_id": d.get("track_id"),
            "north_m": d.get("north_m"), "east_m": d.get("east_m"),
            "r95_m": d.get("r95_m"), "label": d.get("label"),
            "needs": d.get("needs"), "immobility": d.get("immobility"),
            "speed_ms": d.get("speed_ms"), "best_gsd_m": d.get("best_gsd_m"),
            "viability": d.get("viability"),
            "landing_zones": (d.get("landing_zones") or [])[:3],
            "hazards_nearby": (d.get("hazards_nearby") or [])[:4],
            "needs_confirmation_pass": d.get("needs_confirmation_pass"),
            "first_seen_t": d.get("first_seen_t"),
        }

    def publish_hazard(self, hazard: Any, key: Optional[str] = None,
                       alert: bool = True) -> bool:
        d = hazard.to_dict() if hasattr(hazard, "to_dict") else dict(hazard)
        k = key or f"hazard:{d.get('hazard_class', 'x')}:" \
                   f"{round(float(d.get('north_m', 0)) / 10)}:" \
                   f"{round(float(d.get('east_m', 0)) / 10)}"
        kind = MessageKind.HAZARD_ALERT if alert else MessageKind.HAZARD_UPDATE
        return self.queue.enqueue(Message(kind=kind, body=d, key=k,
                                          tier=kind.tier, ttl_s=1800.0))

    def publish_nav(self, nav: Dict[str, Any]) -> bool:
        return self.queue.enqueue(Message(
            kind=MessageKind.NAV_STATUS, body=dict(nav), key="nav",
            tier=MessageKind.NAV_STATUS.tier, ttl_s=30.0))

    def publish_vehicle(self, vehicle: Dict[str, Any]) -> bool:
        return self.queue.enqueue(Message(
            kind=MessageKind.VEHICLE_TELEMETRY, body=dict(vehicle), key="vehicle",
            tier=MessageKind.VEHICLE_TELEMETRY.tier, ttl_s=10.0))

    def publish_coverage(self, coverage: Dict[str, Any]) -> bool:
        return self.queue.enqueue(Message(
            kind=MessageKind.COVERAGE, body=dict(coverage), key="coverage",
            tier=MessageKind.COVERAGE.tier, ttl_s=120.0))

    def publish_event(self, text: str, **extra: Any) -> bool:
        body = {"text": text}
        body.update(extra)
        return self.queue.enqueue(Message(
            kind=MessageKind.EVENT_LOG, body=body,
            key=f"event:{time.time_ns()}", tier=PriorityTier.BULK,
            ttl_s=3600.0))

    def set_range(self, distance_m: float) -> bool:
        return self.link.set_range(distance_m)

    def set_transport(self, name: str) -> None:
        """Switch radio.  The queue does not care; only the budget changes."""
        if name not in TRANSPORTS:
            raise KeyError(f"unknown transport {name!r}; "
                           f"known: {sorted(TRANSPORTS)}")
        self.link.profile = TRANSPORTS[name]
        log.info("uplink transport -> %s (%.0f bps, MTU %d B)", name,
                 self.link.profile.bitrate_bps, self.link.profile.max_packet_bytes)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            ground = self.report.to_dict()
        return {
            "queue": self.queue.snapshot(),
            "link": self.link.report(),
            "credit_bytes": round(self._credit_bytes, 1),
            "ground": {k: v for k, v in ground.items()
                       if k in ("received", "n_survivors", "n_hazards")},
        }
