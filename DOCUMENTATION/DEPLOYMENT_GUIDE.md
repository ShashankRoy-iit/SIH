# Deployment Guide — From Box to Flown Sortie

## Hardware Mapping

| Component | Model / Spec | Connection |
|---|---|---|
| Flight Controller | TBS Lucid H743 Wing (`TBS_LUCID_H7_WING`) | MAVLink over Telem1/2 |
| Companion Computer | Qualcomm RB3 Gen 2 / VOXL 2 | USB / UART / Ethernet |
| Thermal Camera | LWIR module (simulated in `emulator/`) | MIPI-CSI / USB |
| RGB Camera | Analog + VTX / Companion camera | Analog / USB |
| GPS Module | 30+ satellite fix | UART (Serial2) |
| Control Link | ELRS + RadioMaster Pocket | MAVLink (Serial6) |
| Data Link | LoRa 900 | UART (Serial7) |

## Flash Instructions

1. Download ArduPilot `.apj` or `.bin` for `TBS_LUCID_H7_WING`.
2. Load parameter file: `configs/ardupilot_base_params.parm` or `plan/configs/ardupilot_plan_params.parm`.
3. Verify `GPS_TYPE=2`, `SERIALx_PROTOCOL` settings.

## Field Checklist

See `docs/FIELD_TEST_CHECKLIST.md` (derived from original SIH repo).

## Companion Install

```bash
sudo ./base/deploy/install_companion.sh
# or for operational plan:
sudo ./plan/deploy/install_plan.sh
```
