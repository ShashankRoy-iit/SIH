#!/usr/bin/env python3
"""Serve an offline, read-only replay of a recorded SAHYOG sortie.

Example::

    python scripts/serve_replay.py --artifact artifacts/mission_flood.json --port 8090

Then open http://0.0.0.0:8090 in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- repo-root bootstrap (see scripts/_bootstrap.py) ------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap import bootstrap  # noqa: E402

bootstrap()
# ---------------------------------------------------------------------------

from sar.gcs.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
