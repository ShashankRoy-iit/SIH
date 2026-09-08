#!/usr/bin/env bash
# Install SAHYOG SAR onto a companion computer (Qualcomm RB3 Gen 2, Jetson,
# Raspberry Pi, or any Debian/Ubuntu-derived aarch64 board).
#
#   sudo ./deploy/install_companion.sh [--no-ai] [--prefix /opt/sar]
#
# What it does, and why each step is here:
#   1. system packages          - python venv, v4l utils, serial tools
#   2. a dedicated `sar` user   - the flight loop must not run as root; it needs
#                                 exactly two groups, dialout and video
#   3. /opt/sar + virtualenv    - isolated from the distro python, so an OS
#                                 update cannot change what flies
#   4. project install          - editable-equivalent copy with [hardware] and
#                                 optionally [ai] extras
#   5. /etc/sar/onboard.yaml    - config lives outside the code tree so an
#                                 update never silently changes tuning
#   6. udev rules               - stable /dev/sar-* names (see the rules file)
#   7. /var/log/sar             - artifacts and sortie reports
#   8. systemd unit INSTALLED   - but deliberately NOT enabled.  Enabling it is
#                                 a decision made after the field-test checklist
#                                 passes, not by an installer.
#
# It is safe to re-run: existing configuration is never overwritten.

set -euo pipefail

PREFIX="/opt/sar"
WITH_AI=1
SAR_USER="sar"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-ai)  WITH_AI=0; shift ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "error: run as root (sudo $0)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[0;32mok\033[0m  %s\n' "$*"; }

# --------------------------------------------------------------------------- #
say "1/8  system packages"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        v4l-utils libgl1 libglib2.0-0 udev
    ok "apt packages installed"
else
    echo "    ! not a Debian-family system; install python3-venv, v4l-utils by hand"
fi

# --------------------------------------------------------------------------- #
say "2/8  service account"
if ! id -u "$SAR_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir /var/lib/sar \
            --shell /usr/sbin/nologin "$SAR_USER"
    ok "created user '$SAR_USER'"
else
    ok "user '$SAR_USER' exists"
fi
# dialout: the autopilot UART.  video: the cameras.  Nothing else.
usermod -aG dialout,video "$SAR_USER"
ok "groups: dialout, video"

# --------------------------------------------------------------------------- #
say "3/8  install tree at $PREFIX"
mkdir -p "$PREFIX" "$PREFIX/models"
# Copy rather than symlink the repo: the aircraft must keep flying the exact
# code it was tested with, even if someone pulls on the source checkout.
tar -C "$REPO_ROOT" \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='artifacts' --exclude='datasets' \
    -cf - . | tar -C "$PREFIX" -xf -
ok "code copied to $PREFIX"

if [[ ! -d "$PREFIX/venv" ]]; then
    python3 -m venv "$PREFIX/venv"
    ok "virtualenv created"
fi
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip wheel

# --------------------------------------------------------------------------- #
say "4/8  python dependencies"
"$PREFIX/venv/bin/pip" install --quiet "$PREFIX[hardware]"
ok "core + hardware extras"
if [[ $WITH_AI -eq 1 ]]; then
    # onnxruntime is the flight inference path.  On a Qualcomm board with the
    # QNN SDK present, install onnxruntime-qnn instead and the registry will
    # prefer the .bin models automatically.
    if "$PREFIX/venv/bin/pip" install --quiet "$PREFIX[ai]"; then
        ok "ai extras (onnxruntime)"
    else
        echo "    ! onnxruntime unavailable for this platform;"
        echo "      the audited heuristic detector will be used (supported)"
    fi
else
    ok "skipping ai extras (--no-ai): heuristic detector only"
fi

# --------------------------------------------------------------------------- #
say "5/8  configuration"
mkdir -p /etc/sar
if [[ -f /etc/sar/onboard.yaml ]]; then
    ok "/etc/sar/onboard.yaml exists - left untouched"
    cp "$REPO_ROOT/configs/onboard.yaml" /etc/sar/onboard.yaml.new
    echo "      reference copy written to /etc/sar/onboard.yaml.new"
else
    cp "$REPO_ROOT/configs/onboard.yaml" /etc/sar/onboard.yaml
    ok "installed /etc/sar/onboard.yaml  (EDIT THIS - hover_power_w must be measured)"
fi
chown -R "$SAR_USER":"$SAR_USER" /etc/sar

# --------------------------------------------------------------------------- #
say "6/8  udev rules"
install -m 0644 "$REPO_ROOT/deploy/99-sar-devices.rules" \
        /etc/udev/rules.d/99-sar-devices.rules
udevadm control --reload-rules || true
udevadm trigger || true
ok "stable device names: /dev/sar-lwir /dev/sar-rgb /dev/sar-autopilot"

# --------------------------------------------------------------------------- #
say "7/8  log and artifact directory"
mkdir -p /var/log/sar
chown -R "$SAR_USER":"$SAR_USER" /var/log/sar "$PREFIX/models"
ok "/var/log/sar"

# --------------------------------------------------------------------------- #
say "8/8  systemd unit"
install -m 0644 "$REPO_ROOT/deploy/sar-onboard.service" \
        /etc/systemd/system/sar-onboard.service
systemctl daemon-reload
ok "sar-onboard.service installed (NOT enabled)"

# --------------------------------------------------------------------------- #
cat <<EOF

Installed.  Nothing is running yet - that is deliberate.

Next, in this order:

  1. Edit the configuration for THIS airframe:
       sudo nano /etc/sar/onboard.yaml
     In particular safety.hover_power_w must be a measured number, and the
     camera devices must match \`v4l2-ctl --list-devices\`.

  2. Check the environment:
       $PREFIX/venv/bin/python $PREFIX/scripts/doctor.py

  3. Hardware-free dry run (no autopilot, no cameras):
       $PREFIX/venv/bin/python $PREFIX/scripts/run_onboard.py --dry-run --duration 60

  4. Bench, real autopilot, PROPS OFF:
       $PREFIX/venv/bin/python $PREFIX/scripts/run_onboard.py \\
           --mode hitl --target /dev/sar-autopilot:921600

  5. On the aircraft, the preflight gate only:
       $PREFIX/venv/bin/python $PREFIX/scripts/run_onboard.py \\
           --mode flight --preflight-only ; echo \$?

  6. Only after docs/FIELD_TEST_CHECKLIST.md passes end to end:
       sudo systemctl enable --now sar-onboard
       journalctl -u sar-onboard -f

EOF
