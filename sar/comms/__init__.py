"""Offline-first communications: priority-tiered store-and-forward.

See :mod:`sar.comms.link` for the design rationale.  The short version: the
aircraft records every detection locally the instant it happens and transmits
when the link allows, in an order that reflects what a responder needs first.
Nothing is lost because the radio was out of range, and one survivor is one
alert rather than one per frame.
"""

from sar.comms.link import (
    CommsProfile, LinkReport, LinkSimulator, Message, MessageKind,
    PriorityTier, QueueStats, StoreAndForwardQueue, TRANSPORTS, TelemetryUplink,
)

__all__ = [
    "CommsProfile", "LinkReport", "LinkSimulator", "Message", "MessageKind",
    "PriorityTier", "QueueStats", "StoreAndForwardQueue", "TRANSPORTS",
    "TelemetryUplink",
]
