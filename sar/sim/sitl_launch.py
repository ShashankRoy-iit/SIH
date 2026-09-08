"""ArduPilot SITL lifecycle management.

Launches the real ``arducopter`` software-in-the-loop binary as a subprocess,
waits for it to become reachable, hands back a :class:`~sar.mavlink.connection.
MavConnection`, and tears it down again.  This is the "library-based simulation"
the project is built around: the same MAVLink client code, the same parameter
profile and the same mission logic run against this process, against
:class:`~sar.sim.sitl.MiniSITL`, and against the physical TBS Lucid H743.

Two SITL behaviours are worth knowing before reading the code, because both look
like a hang if you have not met them before.

**SERIAL0 blocks until a TCP client attaches.**  A standalone ``arducopter``
exposes its USB/console serial port as a TCP server (5760 by default, plus 10
per ``--instance``) and prints ``Waiting for connection ....``, and does not
proceed to ``ArduPilot Ready`` until something connects.  ``sim_vehicle.py``
normally does this by starting MAVProxy; here the flight client connects to that
port directly, which is both simpler and exactly what plugging a laptop into the
board's USB port does.

**Parameter overrides for serial ports turn them into more TCP servers.**  Any
``SERIALn_PROTOCOL`` written through ``--defaults`` makes SITL bind another
blocking TCP port.  That is why ``configs/ardupilot_sitl.parm`` deliberately
omits the UART routing that ``configs/ardupilot_hardware.parm`` carries.

Getting the binary
------------------
``find_ardupilot_binary()`` searches ``$SAR_ARDUPILOT_BIN``, then a checkout
built with ``./waf configure --board sitl && ./waf copter``, then ``$PATH``.
Building SITL needs a C++ toolchain and cmake and takes several minutes; if it
is not present, everything in this package still runs against
:class:`~sar.sim.sitl.MiniSITL`, which needs no toolchain at all.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("sar.sim.sitl_launch")

__all__ = ["ArduPilotSitl", "SitlNotAvailable", "find_ardupilot_binary",
           "REPO_ROOT", "SITL_PARAMS", "HW_PARAMS"]

REPO_ROOT = Path(__file__).resolve().parents[2]
SITL_PARAMS = REPO_ROOT / "configs" / "ardupilot_sitl.parm"
HW_PARAMS = REPO_ROOT / "configs" / "ardupilot_hardware.parm"

#: Reference scenario origin: Kota, on the Chambal.  The SITL home and the
#: perception world share this so a detection's local metres and the vehicle's
#: reported latitude are the same coordinate frame with no offset to reconcile.
DEFAULT_HOME = (25.185, 75.8357, 250.0, 0.0)


class SitlNotAvailable(RuntimeError):
    """No ``arducopter`` binary could be found or it failed to start."""

    def __init__(self, message: str, hint: str = "") -> None:
        self.hint = hint
        super().__init__(message + (f"\n  {hint}" if hint else ""))


def find_ardupilot_binary(vehicle: str = "copter") -> Optional[Path]:
    """Locate a built ArduPilot SITL binary, or ``None``.

    Search order: ``$SAR_ARDUPILOT_BIN``, then ``$SAR_ARDUPILOT_ROOT/build/sitl/
    bin``, then a few conventional checkout locations, then ``$PATH``.
    """
    exe = f"ardu{vehicle}"
    candidates: List[Path] = []
    env_bin = os.environ.get("SAR_ARDUPILOT_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    roots = [os.environ.get("SAR_ARDUPILOT_ROOT"),
             os.path.expanduser("~/.thirdparty/ardupilot"),
             os.path.expanduser("~/ardupilot"),
             "/opt/ardupilot"]
    for root in roots:
        if root:
            candidates.append(Path(root) / "build" / "sitl" / "bin" / exe)
            candidates.append(Path(root) / "build" / "sitl" / "bin" / f"{exe}.exe")
    for c in candidates:
        try:
            if c.is_file() and os.access(c, os.X_OK):
                return c
        except OSError:
            continue
    found = shutil.which(exe)
    return Path(found) if found else None


def _tcp_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class SitlLog:
    """Captured SITL stdout/stderr, which is where pre-arm reasons appear."""

    lines: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.lines.extend(text.splitlines())
        if len(self.lines) > 4000:
            del self.lines[:2000]

    def grep(self, needle: str) -> List[str]:
        return [l for l in self.lines if needle.lower() in l.lower()]

    def tail(self, n: int = 25) -> str:
        return "\n".join(self.lines[-n:])


class ArduPilotSitl:
    """An ``arducopter`` SITL process plus a MAVLink connection to it.

    Parameters
    ----------
    home : tuple
        ``(lat, lon, alt_m, heading_deg)``.  Defaults to the reference scenario
        origin so the flight and perception worlds coincide.
    model : str
        SITL airframe model.  ``"+"`` is a quad in plus orientation, ``"quad"``
        in X.  ``--list-models`` prints the embedded set.
    speedup : float
        Simulation rate multiplier.  Greater than 1 runs faster than real time,
        which is how a 20-minute survey is tested in 2 minutes; the MAVLink
        client sees the same messages, just sooner.  Set to 1.0 when measuring
        anything time-dependent, including control-loop behaviour and the
        geo-tagger's latency term.
    instance : int
        SITL instance number.  Adds ``10 * instance`` to every port, so several
        vehicles can fly at once - which is how the multi-aircraft search
        decomposition is tested.
    params : Path or str or None
        Defaults file.  ``None`` uses ``configs/ardupilot_sitl.parm``.
    extra_args : sequence of str
        Anything else to pass through to the binary.
    scratch : Path or None
        Working directory for the eeprom, logs and terrain cache.  A temporary
        directory is used and removed on exit unless this is given, in which
        case it is kept - useful for inspecting ``.tlog`` files afterwards.
    wipe : bool
        Start from a blank eeprom so ``params`` is applied cleanly.  Leaving it
        false keeps the previous run's parameter writes, which is what you want
        when iterating on one parameter at a time.
    """

    #: SITL exposes SERIAL0 as a TCP server here (plus 10 per instance) and will
    #: not finish booting until a client connects.
    SERIAL0_PORT_BASE = 5760

    def __init__(self,
                 home: Tuple[float, float, float, float] = DEFAULT_HOME,
                 model: str = "+",
                 speedup: float = 1.0,
                 instance: int = 0,
                 params: Optional[os.PathLike] = None,
                 extra_args: Sequence[str] = (),
                 scratch: Optional[os.PathLike] = None,
                 wipe: bool = True,
                 vehicle: str = "copter",
                 console_passthrough: bool = False) -> None:
        self.home = tuple(home)
        self.model = model
        self.speedup = float(speedup)
        self.instance = int(instance)
        self.vehicle = vehicle
        self.console_passthrough = console_passthrough
        self.params = Path(params) if params else SITL_PARAMS
        self.extra_args = list(extra_args)
        self.wipe = bool(wipe)
        self.binary = find_ardupilot_binary(vehicle)
        self.log = SitlLog()
        self.proc: Optional[subprocess.Popen] = None
        self.conn = None
        self.serial0_port = self.SERIAL0_PORT_BASE + 10 * self.instance
        self.telemetry_port = 14550 + 10 * self.instance
        self._reader_thread = None
        self._stop = False
        self._keep_scratch = scratch is not None
        self._scratch = (Path(scratch) if scratch
                         else Path(tempfile.mkdtemp(prefix="sahyog-sitl-")))
        if self.binary is None:
            raise SitlNotAvailable(
                "no ArduPilot SITL binary found",
                hint="set SAR_ARDUPILOT_BIN=/path/to/build/sitl/bin/arducopter, "
                     "or build one with './waf configure --board sitl && ./waf "
                     "copter' inside an ArduPilot checkout.  Everything in this "
                     "package also runs against sar.sim.sitl.MiniSITL, which "
                     "needs no toolchain.")
        if not self.params.is_file():
            raise SitlNotAvailable(f"parameter file not found: {self.params}")

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<ArduPilotSitl {self.vehicle} {state} port={self.serial0_port}>"

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def scratch_dir(self) -> Path:
        return self._scratch

    # ------------------------------------------------------------------ #
    def command(self) -> List[str]:
        """The exact argv used, so a run can be reproduced by hand."""
        lat, lon, alt, hdg = self.home
        cmd = [str(self.binary),
               "--model", self.model,
               "--home", f"{lat},{lon},{alt},{hdg}",
               "--speedup", f"{self.speedup:g}",
               "--instance", str(self.instance),
               "--defaults", str(self.params)]
        if self.wipe:
            cmd.append("--wipe")
        cmd.extend(self.extra_args)
        return cmd

    def start(self, ready_timeout_s: float = 60.0) -> "ArduPilotSitl":
        """Launch and wait until MAVLink is reachable."""
        self._scratch.mkdir(parents=True, exist_ok=True)
        cmd = self.command()
        log.info("starting SITL: %s", " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(self._scratch),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None)
        except OSError as exc:
            raise SitlNotAvailable(f"could not exec {self.binary}: {exc}") from exc
        atexit.register(self.stop)
        self._start_reader()

        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SitlNotAvailable(
                    f"SITL exited immediately with code {self.proc.returncode}",
                    hint=self.log.tail(30))
            if _tcp_port_open("127.0.0.1", self.serial0_port):
                break
            time.sleep(0.2)
        else:
            self.stop()
            raise SitlNotAvailable(
                f"SITL did not open TCP {self.serial0_port} within "
                f"{ready_timeout_s:.0f} s", hint=self.log.tail(30))
        log.info("SITL listening on tcp 127.0.0.1:%d (pid %d)",
                 self.serial0_port, self.proc.pid)
        return self

    def _start_reader(self) -> None:
        """Drain stdout on a thread.

        SITL's output is unbounded and a full pipe blocks the process, so it has
        to be read continuously rather than after the flight.  It is also where
        pre-arm diagnostics and EKF complaints appear, so it is worth keeping.
        """
        import threading

        def _pump() -> None:
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                if self._stop:
                    break
                self.log.add(line)
                if self.console_passthrough:
                    sys.stdout.write(line)
                    sys.stdout.flush()

        self._stop = False
        self._reader_thread = threading.Thread(target=_pump, name="sitl-log",
                                               daemon=True)
        self._reader_thread.start()

    # ------------------------------------------------------------------ #
    def connect(self, timeout: float = 30.0, **kwargs):
        """Open a :class:`MavConnection` to the running SITL.

        Connecting to the SERIAL0 TCP port is what unblocks SITL's boot, so this
        is not merely convenient - it is part of the startup sequence.
        """
        from sar.mavlink.connection import MavConnection, LinkTimeout
        target = f"tcp:127.0.0.1:{self.serial0_port}"
        end = time.time() + timeout
        last: Optional[Exception] = None
        while time.time() < end:
            try:
                conn = MavConnection(target, **kwargs)
                conn.wait_heartbeat(timeout=min(10.0, max(1.0, end - time.time())))
                self.conn = conn
                return conn
            except Exception as exc:
                last = exc
                time.sleep(0.5)
        raise SitlNotAvailable(
            f"could not establish MAVLink on {target}",
            hint=f"last error: {last!r}\nSITL log tail:\n{self.log.tail(20)}")

    def boot(self, ready_timeout_s: float = 60.0, **conn_kwargs):
        """Start, connect, and wait for ``ArduPilot Ready``, in that order.

        The order matters and is the reason this method exists.  SITL does not
        print ``ArduPilot Ready`` until a client attaches to its SERIAL0 TCP
        port, so waiting for readiness *before* connecting waits for something
        that cannot happen yet and always times out.  Use this instead of
        calling :meth:`start`, :meth:`connect` and :meth:`wait_ready` separately.
        """
        self.start(ready_timeout_s=ready_timeout_s)
        conn = self.connect(timeout=ready_timeout_s, **conn_kwargs)
        ready = self.wait_ready(timeout=ready_timeout_s)
        if not ready:
            log.warning("SITL did not print 'ArduPilot Ready'; log tail:\n%s",
                        self.log.tail(15))
        return conn

    #: Either backend may announce readiness, and they say so differently.
    #: Matching only the ArduPilot string makes MiniSITL look like a hung boot.
    READY_PATTERN = re.compile(r"(ArduPilot|MiniSITL)\s+Ready")

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Wait for the ready banner over MAVLink, falling back to the log.

        The banner is emitted as a MAVLink ``STATUSTEXT`` rather than on stdout,
        so an already-open connection is checked first and the process log is
        only a fallback.  Waiting on stdout alone always times out, which reads
        exactly like a failed boot.
        """
        end = time.time() + timeout
        while time.time() < end:
            if self.conn is not None:
                self.conn.pump(0.2)
                if any(self.READY_PATTERN.search(s)
                       for _, s in self.conn._statustext):
                    return True
            if self.log.grep("Ready"):
                return True
            if not self.running:
                return False
            time.sleep(0.25)
        return False

    # ------------------------------------------------------------------ #
    def stop(self, timeout: float = 10.0) -> int:
        """Terminate the process group and optionally clean up the scratch dir."""
        rc = 0
        self._stop = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:                                      # pragma: no cover
                    self.proc.terminate()
            except (OSError, ProcessLookupError):
                pass
            try:
                rc = self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:                                  # pragma: no cover
                        self.proc.kill()
                except (OSError, ProcessLookupError):
                    pass
                rc = self.proc.wait(timeout=timeout)
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:                              # pragma: no cover
                pass
            self.conn = None
        self.proc = None
        if not self._keep_scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)
        return int(rc or 0)

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "ArduPilotSitl":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    def prearm_blockers(self) -> List[str]:
        """Pre-arm failure reasons harvested from the SITL log."""
        out: List[str] = []
        for l in self.log.lines:
            ls = l.strip()
            if ls.startswith(("PreArm:", "Arm:", "EKF", "Check ")):
                out.append(ls)
        seen, uniq = set(), []
        for o in out:
            if o not in seen:
                seen.add(o)
                uniq.append(o)
        return uniq

    def info(self) -> Dict[str, object]:
        return {"binary": str(self.binary), "pid": self.proc.pid if self.proc else None,
                "running": self.running, "serial0_port": self.serial0_port,
                "telemetry_port": self.telemetry_port, "home": self.home,
                "model": self.model, "speedup": self.speedup,
                "instance": self.instance, "params": str(self.params),
                "scratch": str(self._scratch),
                "ready": bool(self.log.grep("ArduPilot Ready")),
                "prearm_blockers": self.prearm_blockers()}
