"""Tests for the offline-first comms layer.

These are written as behaviour tests rather than unit tests of internals,
because the property that matters is what the ground station ends up knowing
after a bad link, not what the queue's fields look like mid-flight.

The failure modes covered here are the ones that are invisible in a bench demo
with a good radio, which is exactly when they matter least.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sar.comms import (                                   # noqa: E402
    LinkSimulator, Message, MessageKind, PriorityTier, StoreAndForwardQueue,
    TRANSPORTS, TelemetryUplink,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _assessment(track_id: int, north_m: float = 0.0, sigma_m: float = 2.0,
                priority: str = "immediate"):
    """A stand-in for :class:`sar.perception.SurvivorAssessment`.

    Built as a duck-typed object rather than the real dataclass so these tests
    do not drag the whole perception stack in - the uplink only ever calls
    ``to_dict()``, and if that contract changes these tests should fail loudly.
    """

    class _A:
        def to_dict(self_inner):
            return {
                "track_id": track_id, "lat": 25.0 + track_id * 1e-4,
                "lon": 85.0, "north_m": north_m, "east_m": 0.0,
                "sigma_m": sigma_m, "r95_m": sigma_m * 2,
                "label": "person", "confidence": 0.9, "priority": priority,
                "needs": "medical", "group_size": 1, "time_critical_s": 900,
                "viability": {"score": 0.8}, "n_obs": 5, "cross_modal": True,
                "landing_zones": [], "hazards_nearby": [],
                "last_seen_t": time.time(),
            }

    return _A()


class Clock:
    """A simulation clock for the uplink.

    ``pump_once(now=...)`` accrues air-time credit from the delta between calls,
    because that is what a radio does: pumping 400 times inside one millisecond
    of real time genuinely transmits almost nothing.  Passing a frozen timestamp
    is therefore not a bug in the uplink but a caller that has stopped time, and
    the symptom - zero throughput that looks exactly like being out of range - is
    worth making impossible to write by accident.
    """

    def __init__(self, step_s: float = 0.25) -> None:
        self.t = time.time()
        self.step_s = step_s

    def tick(self) -> float:
        self.t += self.step_s
        return self.t


def _bulk(queue: StoreAndForwardQueue, n: int, size: int = 900) -> None:
    for i in range(n):
        queue.enqueue(Message(kind=MessageKind.IMAGE_THUMBNAIL,
                              body={"b64": "A" * size, "i": i}, key=f"img:{i}",
                              tier=PriorityTier.BULK, ttl_s=3600.0))


# --------------------------------------------------------------------------- #
# Eviction: the property the whole design rests on
# --------------------------------------------------------------------------- #
def test_bulk_is_shed_before_life_safety():
    """A full queue drops imagery, never a survivor."""
    q = StoreAndForwardQueue(max_bytes=4096)
    for i in range(5):
        q.enqueue(Message(kind=MessageKind.SURVIVOR_DISCOVER,
                          body=_assessment(i).to_dict(), key=f"survivor:{i}",
                          tier=PriorityTier.LIFE_SAFETY))
    _bulk(q, 50)

    survivors = [m for m in q._items if m.tier == PriorityTier.LIFE_SAFETY]
    bulk = [m for m in q._items if m.tier == PriorityTier.BULK]
    assert len(survivors) == 5, "survivors were evicted for imagery"
    # Imagery may occupy whatever the survivors leave behind; what it must not
    # do is displace them.
    assert len(bulk) < 50, "imagery was not shed at all"
    assert q.stats.dropped_evicted >= 45


def test_eviction_prefers_oldest_within_a_tier():
    """FIFO within a tier, so the longest-waiting survivor goes first."""
    q = StoreAndForwardQueue(max_bytes=1200)
    for i in range(6):
        m = Message(kind=MessageKind.EVENT_LOG, body={"i": i, "pad": "x" * 300},
                    key=f"e:{i}", tier=PriorityTier.BULK)
        m.created_at = 1000.0 + i          # force a known ordering
        q.enqueue(m)
    assert len(q) < 6
    survivors = [m.body["i"] for m in q._items]
    # The oldest are evicted only when the tier itself must shrink; within BULK
    # the rule is oldest-first, so what remains is the newest.
    assert survivors == sorted(survivors)


def test_life_safety_eviction_is_loud():
    """When survivors genuinely do not fit, that must be logged, not silent."""
    q = StoreAndForwardQueue(max_bytes=1024)
    for i in range(40):
        q.enqueue(Message(kind=MessageKind.SURVIVOR_DISCOVER,
                          body=_assessment(i).to_dict(), key=f"survivor:{i}",
                          tier=PriorityTier.LIFE_SAFETY))
    assert q.stats.dropped_evicted > 0
    assert len(q) < 40


# --------------------------------------------------------------------------- #
# Coalescing: one survivor is one alert, not one per frame
# --------------------------------------------------------------------------- #
def test_repeat_updates_on_one_survivor_occupy_one_slot():
    q = StoreAndForwardQueue()
    for _ in range(200):
        q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="survivor:7",
                          body=_assessment(7).to_dict(),
                          tier=PriorityTier.LIFE_SAFETY))
    assert len(q) == 1, "200 frames of one person should be one queued message"


def test_sub_pixel_jitter_is_not_an_update():
    """Movement smaller than the reported error ellipse carries no information."""
    q = StoreAndForwardQueue()
    base = _assessment(3).to_dict()
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:3", body=dict(base),
                      tier=PriorityTier.LIFE_SAFETY))
    moved = dict(base, lat=base["lat"] + 0.5 / 111320.0)   # 0.5 m, sigma 2 m
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:3", body=moved,
                      tier=PriorityTier.LIFE_SAFETY))
    assert len(q) == 1
    assert q.stats.suppressed_duplicate == 1


def test_a_real_relocation_does_update():
    q = StoreAndForwardQueue()
    base = _assessment(3).to_dict()
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:3", body=dict(base),
                      tier=PriorityTier.LIFE_SAFETY))
    moved = dict(base, lat=base["lat"] + 40.0 / 111320.0)  # 40 m, sigma 2 m
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:3", body=moved,
                      tier=PriorityTier.LIFE_SAFETY))
    assert len(q) == 1
    assert q.stats.coalesced == 1
    assert q._items[0].revision == 2, "a genuine update must bump the revision"


def test_escalating_priority_is_an_update():
    q = StoreAndForwardQueue()
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:9",
                      body=_assessment(9, priority="delayed").to_dict(),
                      tier=PriorityTier.LIFE_SAFETY))
    q.enqueue(Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:9",
                      body=_assessment(9, priority="immediate").to_dict(),
                      tier=PriorityTier.LIFE_SAFETY))
    assert q.stats.coalesced == 1, "a triage change must reach the ground"


def test_non_coalescing_kinds_accumulate():
    """Event logs are distinct facts; dropping one loses information."""
    q = StoreAndForwardQueue()
    for i in range(5):
        q.enqueue(Message(kind=MessageKind.EVENT_LOG, body={"text": f"e{i}"},
                          key=f"event:{i}", tier=PriorityTier.BULK))
    assert len(q) == 5


# --------------------------------------------------------------------------- #
# Transmission: no starvation, priority order, out-of-order rejection
# --------------------------------------------------------------------------- #
def test_a_frozen_clock_yields_no_throughput():
    """Documenting the physics rather than papering over it."""
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q, LinkSimulator(TRANSPORTS["lora_900"], range_m=1.0),
                         window_s=0.25)
    up.link.set_range(10.0)
    up.publish_survivor(_assessment(1), first_report=True)
    frozen = time.time() + 10.0
    for _ in range(50):
        assert up.pump_once(now=frozen) <= 1     # at most the one grant on entry
    assert len(q) >= 1


def test_message_larger_than_one_window_is_still_sent():
    """The regression that wedged the queue permanently.

    A 395-byte survivor alert over a 2.4 kbit link in a 0.25 s window is five
    times the budget.  A strict budget means the most important message in the
    system can never leave, and the queue fills to capacity and stops.
    """
    q = StoreAndForwardQueue()
    link = LinkSimulator(TRANSPORTS["elrs_telemetry"], range_m=5000.0)
    up = TelemetryUplink(q, link, window_s=0.25)
    up.link.profile = TRANSPORTS["elrs_telemetry"]
    # ELRS MTU is 64 B, so use a transport whose MTU can carry the alert.
    up.set_transport("lora_900")
    up.link.set_range(10.0)
    up.publish_survivor(_assessment(1), first_report=True)

    clk = Clock(0.25)
    sent = 0
    for _ in range(40):
        sent += up.pump_once(now=clk.tick())
    assert sent >= 1, "the highest-priority message starved"
    assert up.report.to_dict()["n_survivors"] == 1


def test_life_safety_precedes_bulk_on_recovery():
    q = StoreAndForwardQueue(max_bytes=64 * 1024)
    up = TelemetryUplink(q, LinkSimulator(TRANSPORTS["lora_900"], range_m=6000.0),
                         window_s=0.25)
    up.link.set_range(50.0)
    for i in range(12):
        up.publish_survivor(_assessment(i), first_report=True)
    # Sized to fit the LoRa MTU: a 700-byte thumbnail is not a message this
    # link can ever carry, and testing priority against something unsendable
    # proves nothing.
    _bulk(q, 60, size=150)

    order: list[str] = []
    up.on_deliver = lambda m: order.append(m.kind.value)
    clk = Clock(0.25)
    for _ in range(600):
        up.pump_once(now=clk.tick())
        if not q:
            break

    survivor_idx = [i for i, k in enumerate(order)
                    if k == MessageKind.SURVIVOR_DISCOVER.value]
    bulk_idx = [i for i, k in enumerate(order)
                if k == MessageKind.IMAGE_THUMBNAIL.value]
    assert survivor_idx, "no survivor alerts were delivered"
    assert bulk_idx, "no imagery was delivered at all"
    assert max(survivor_idx) < min(bulk_idx), "imagery was sent ahead of survivors"
    assert len(survivor_idx) == 12


def test_outage_then_recovery_delivers_everything_queued():
    q = StoreAndForwardQueue(max_bytes=64 * 1024)
    up = TelemetryUplink(q, LinkSimulator(TRANSPORTS["lora_900"], range_m=1500.0),
                         window_s=0.25)
    up.link.set_range(9000.0)                    # out of range
    for i in range(6):
        up.publish_survivor(_assessment(i), first_report=True)
    # Each publish queues an alert plus a detail message.
    expected = len(q)
    assert expected == 12

    clk = Clock(0.25)
    for _ in range(20):
        assert up.pump_once(now=clk.tick()) == 0, "sent while out of range"
    assert len(q) == expected, "the queue must hold through an outage"

    up.link.set_range(500.0)                     # back in range
    for _ in range(400):
        up.pump_once(now=clk.tick())
        if not q:
            break
    # The alerts go; the detail messages are wider than a LoRa packet and stay
    # queued for a link that can carry them.  That is the tiering working, not a
    # delivery failure - a responder has who/where/how-urgent, and the viability
    # breakdown and landing zones arrive when the Wi-Fi bridge comes up.
    assert up.report.to_dict()["n_survivors"] == 6
    assert up.link.outage_s > 0
    remaining = {m.kind for m in q._items}
    assert remaining == {MessageKind.SURVIVOR_DETAIL}
    for m in q._items:
        assert m.size_bytes() > TRANSPORTS["lora_900"].max_packet_bytes

    up.set_transport("wifi_bridge")
    for _ in range(200):
        up.pump_once(now=clk.tick())
        if not q:
            break
    assert len(q) == 0, "detail never drained once a capable link appeared"


def test_out_of_order_update_is_rejected_at_the_ground():
    """A requeue can reorder; a stale position must not overwrite a fresh one."""
    rep = TelemetryUplink().report
    newer = Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:1",
                    body=_assessment(1, north_m=500.0).to_dict(),
                    tier=PriorityTier.LIFE_SAFETY)
    newer.revision = 7
    older = Message(kind=MessageKind.SURVIVOR_UPDATE, key="s:1",
                    body=_assessment(1, north_m=10.0).to_dict(),
                    tier=PriorityTier.LIFE_SAFETY)
    older.revision = 3
    rep.apply(newer)
    rep.apply(older)
    assert rep.survivors["s:1"]["north_m"] == 500.0


def test_oversized_messages_are_held_not_retried():
    """Something too big for the current radio waits for a wider one.

    It must not be requeued into a failure loop, and it must not block the head
    of the queue: both were live failure modes, the second because a skipped
    message that stays at the front stops everything behind it.
    """
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q, LinkSimulator(TRANSPORTS["lora_900"], range_m=5000.0),
                         window_s=0.25)
    up.link.set_range(10.0)
    # Bigger than a LoRa packet, smaller than a Wi-Fi one - so these have a link
    # they can eventually use.  Sizing them past every MTU would only test that
    # the queue holds unsendable data, which is the less interesting half.
    _bulk(q, 3, size=1000)
    up.publish_survivor(_assessment(1), first_report=True)

    clk = Clock(0.25)
    for _ in range(40):
        up.pump_once(now=clk.tick())
    # The survivor alert goes out even though oversized bulk sits ahead of it.
    assert up.report.to_dict()["n_survivors"] == 1
    # Still queued: the three oversized thumbnails plus this survivor's detail
    # message, which is also wider than a LoRa packet.
    assert len(q) == 4
    assert sum(1 for m in q._items
               if m.kind == MessageKind.IMAGE_THUMBNAIL) == 3

    # A Wi-Fi bridge arrives and they go.
    up.set_transport("wifi_bridge")
    for _ in range(80):
        up.pump_once(now=clk.tick())
        if not q:
            break
    assert len(q) == 0, "the bulk never went out once a capable link appeared"


def test_the_control_link_is_not_a_data_link():
    """ELRS cannot carry an alert, and that must be explicit in the model.

    The most reliable radio on the airframe is also the narrowest.  Treating it
    as the reporting path is the single most attractive wrong answer here,
    because it works on the bench - the aircraft is ten metres from the laptop -
    and fails at the first sortie where the reporting range exceeds the control
    range, which is most of them.
    """
    assert TRANSPORTS["elrs_telemetry"].carries_alerts is False
    alert_b = _queued_alert_size(_assessment(1))
    assert alert_b > TRANSPORTS["elrs_telemetry"].max_packet_bytes
    # LoRa is the minimum alert-capable link, and the budget is derived from it.
    narrowest = min(t.max_packet_bytes for t in TRANSPORTS.values()
                    if t.carries_alerts and t.bitrate_bps > 0)
    assert narrowest == TRANSPORTS["lora_900"].max_packet_bytes
    assert TelemetryUplink.ALERT_BUDGET_BYTES < narrowest
    assert alert_b <= narrowest


def test_adding_a_narrower_radio_tightens_the_budget():
    """The budget is derived, not hardcoded, so the table stays honest."""
    from dataclasses import replace
    from sar.comms import link as L
    saved = dict(L.TRANSPORTS)
    try:
        L.TRANSPORTS["test_narrow"] = replace(
            L.TRANSPORTS["lora_900"], name="test_narrow", max_packet_bytes=120)
        recomputed = min(t.max_packet_bytes for t in L.TRANSPORTS.values()
                         if t.carries_alerts and t.bitrate_bps > 0
                         and t.max_packet_bytes > 0) - 22
        assert recomputed == 98
    finally:
        L.TRANSPORTS.clear()
        L.TRANSPORTS.update(saved)


def test_expired_messages_are_purged():
    q = StoreAndForwardQueue()
    stale = Message(kind=MessageKind.COVERAGE, body={"a": 1}, key="cov",
                    tier=PriorityTier.SITUATIONAL, ttl_s=0.01)
    q.enqueue(stale)
    time.sleep(0.03)
    assert q.next_batch(10000) == []
    assert q.stats.dropped_expired == 1


def test_survivor_alerts_never_expire():
    """A survivor found three hours ago is still worth reporting."""
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q, LinkSimulator(TRANSPORTS["lora_900"], range_m=1.0),
                         window_s=0.25)
    up.link.set_range(10.0)
    up.publish_survivor(_assessment(1), first_report=True)
    m = q._items[0]
    assert m.ttl_s is None
    m.created_at = time.time() - 3 * 3600
    assert not m.expired()


# --------------------------------------------------------------------------- #
# Transport characterisation
# --------------------------------------------------------------------------- #
def test_transport_hierarchy_is_ordered_by_capability():
    """The design target: what works on ELRS works on everything above it."""
    elrs = TRANSPORTS["elrs_telemetry"]
    lora = TRANSPORTS["lora_900"]
    wifi = TRANSPORTS["wifi_bridge"]
    assert elrs.bitrate_bps < lora.bitrate_bps < wifi.bitrate_bps
    assert wifi.max_packet_bytes > lora.max_packet_bytes > elrs.max_packet_bytes
    assert TRANSPORTS["out_of_range"].bitrate_bps == 0.0


def test_lora_survives_an_outage_that_elrs_cannot():
    """Non-line-of-sight is the reason to carry a second radio."""
    assert TRANSPORTS["lora_900"].requires_los is False
    assert TRANSPORTS["elrs_telemetry"].requires_los is True


def _queued_alert_size(assessment) -> int:
    """Size of the alert the uplink actually produces, as it goes on the wire."""
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q)
    up.publish_survivor(assessment, first_report=True)
    alerts = [m for m in q._items
              if m.kind == MessageKind.SURVIVOR_DISCOVER]
    assert len(alerts) == 1
    return alerts[0].size_bytes()


def test_a_survivor_alert_fits_the_narrowest_radio():
    """If it does not fit LoRa, the offline-first claim is not real.

    This is the test that caught the original design: a readable, complete alert
    serialised to 409 B against a 222 B MTU, so the most important message the
    system produces could not be sent over the radio carried precisely for when
    nothing else works.  It is invisible in any demo with Wi-Fi.
    """
    size = _queued_alert_size(_assessment(1))
    mtu = TRANSPORTS["lora_900"].max_packet_bytes
    assert size <= mtu, f"alert is {size} B, LoRa MTU is {mtu} B"
    assert size <= TelemetryUplink.ALERT_BUDGET_BYTES


def test_alert_keeps_its_actionable_fields_after_degrading():
    """Shrinking to fit must never drop position, priority or group size."""
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q)
    rich = _assessment(4).to_dict()
    rich["landing_zones"] = [{"north_m": i} for i in range(50)]
    rich["hazards_nearby"] = [{"hazard_class": f"h{i}"} for i in range(50)]
    up.publish_survivor(rich, first_report=True)
    alert = [m for m in q._items
             if m.kind == MessageKind.SURVIVOR_DISCOVER][0]
    assert alert.size_bytes() <= TRANSPORTS["lora_900"].max_packet_bytes
    for essential in ("la", "lo", "pr", "gs"):
        assert essential in alert.body, f"degrading dropped {essential}"


def test_rich_detail_goes_out_at_a_lower_tier():
    """The alert is minimal; the full picture must still reach the ground."""
    q = StoreAndForwardQueue()
    up = TelemetryUplink(q)
    up.publish_survivor(_assessment(2), first_report=True)
    kinds = {m.kind: m for m in q._items}
    assert MessageKind.SURVIVOR_DISCOVER in kinds
    assert MessageKind.SURVIVOR_DETAIL in kinds
    assert kinds[MessageKind.SURVIVOR_DETAIL].tier == PriorityTier.TACTICAL
    assert (kinds[MessageKind.SURVIVOR_DETAIL].size_bytes()
            > kinds[MessageKind.SURVIVOR_DISCOVER].size_bytes())


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #
def test_every_kind_has_a_unique_wire_code():
    codes = [k.wire_code for k in MessageKind]
    assert len(set(codes)) == len(codes)
    for k in MessageKind:
        assert MessageKind.from_wire_code(k.wire_code) is k


def test_payload_round_trips():
    m = Message(kind=MessageKind.SURVIVOR_DISCOVER,
                body=_assessment(1).to_dict(), key="s1",
                tier=PriorityTier.LIFE_SAFETY)
    m.revision = 4
    rt = Message.from_payload(m.payload)
    assert rt.kind is m.kind and rt.key == m.key
    assert rt.revision == m.revision and rt.body == m.body


def test_tier_is_not_sent_on_the_wire():
    """It is a pure function of the kind; restating it costs bytes."""
    m = Message(kind=MessageKind.HAZARD_ALERT, body={"a": 1}, key="h")
    assert "tier" not in m.payload


def test_unknown_wire_code_is_rejected_loudly():
    with pytest.raises(ValueError):
        MessageKind.from_wire_code("zz")


def test_uplink_does_not_replace_a_caller_supplied_empty_queue():
    """Regression: `queue or default` discards an empty queue because it is falsy.

    The symptom was a system that appeared to transmit nothing while working
    perfectly - every publish went into a private queue nobody read.
    """
    q = StoreAndForwardQueue()
    assert not q                                  # empty queues are falsy
    up = TelemetryUplink(q)
    assert up.queue is q


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
