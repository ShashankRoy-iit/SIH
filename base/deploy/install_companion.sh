#!/bin/bash
# Install companion computer for drone rescue AI stack.
# Target: Qualcomm RB3 Gen 2 / VOXL 2 or laptop.

set -e

echo "=== Drone Rescue AI — Companion Install ==="
echo "Installing dependencies..."

# Python environment
python3 -m venv /opt/drone-rescue-venv || true
source /opt/drone-rescue-venv/bin/activate || true

pip install --break-system-packages -r /home/user/drone-rescue-system/requirements-base.txt || pip install -r /home/user/drone-rescue-system/requirements-base.txt || true

echo "Installing systemd service..."
cp /home/user/drone-rescue-system/deploy/sar-onboard.service /etc/systemd/system/
systemctl daemon-reload || true
systemctl enable sar-onboard.service || true

echo "=== Install Complete ==="
echo "To run manually:"
echo "  cd /home/user/drone-rescue-system/base"
echo "  python3 scripts/run_drone_ai.py --live --dashboard"
