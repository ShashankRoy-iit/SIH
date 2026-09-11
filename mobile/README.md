# `mobile/` — the phone side of the aircraft

The flight stack in `sar/` sees the phone through `sar/hardware/mobile.py`.
This folder is what runs **on** the phone.

| Path | What |
|---|---|
| `phone_onboard.py` | Termux bridge: capture + TFLite AI + VIO → MAVLink over USB-OTG. Runs today, no Android build tools. |
| `android/` | Native APK build spec (Camera2 + NNAPI + usb-serial). Same MAVLink dialect as the bridge. |

Quick bench check (laptop, no phone):

```bash
python3 mobile/phone_onboard.py --mode bench --duration 20
cat artifacts/phone_bench.json
```

Full wiring + app setup: [`docs/MOBILE_COMPANION.md`](../docs/MOBILE_COMPANION.md).
