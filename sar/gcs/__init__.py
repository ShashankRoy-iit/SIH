"""Ground control station: a zero-internet command-centre dashboard.

See :mod:`sar.gcs.dashboard`.  It renders only what the data link has actually
delivered, so a queue of unreported survivors on the aircraft shows up as an empty
survivor list with a non-zero queue depth - which is the operationally true state,
and the one a controller needs to be able to see.
"""

from sar.gcs.dashboard import Dashboard

__all__ = ["Dashboard"]
