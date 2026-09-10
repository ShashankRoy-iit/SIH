#!/bin/bash
# Deploy companion computer for operational plan (no simulation).
set -e
echo "Installing operational plan..."
cp deploy/sar-plan.service /etc/systemd/system/ || true
systemctl daemon-reload || true
systemctl enable sar-plan.service || true
echo "Done. No simulation components installed."
