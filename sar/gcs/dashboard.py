"""Command-centre dashboard: zero-internet, real-time animated tactical HUD and SAR operations manager.

Features:
- Smooth 60fps HTML5 Canvas animation loop with interpolated drone kinematics, spinning propellers,
  camera FOV spotlight cone sweep, animated waving SOS survivors, parachute payload descents,
  pulsing LoRa locator beacons, and moving ground rescue response teams along A* safe corridors.
- Dual-spectrum synthetic sensor feeds (LWIR thermal false-color & RGB camera AI bounding boxes).
- Human identification cards with posture analysis, triage badges, core temperature estimation,
  demographics, and equipment needs.
- Interactive controls for precision air-drop execution and rescue team dispatch.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger("sar.gcs")

__all__ = ["Dashboard"]

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    _HAS_FASTAPI = True
except Exception:                                   # pragma: no cover
    _HAS_FASTAPI = False


# --------------------------------------------------------------------------- #
# Embedded HTML5/CSS3/Canvas Animated Client (Zero External CDN Dependencies)
# --------------------------------------------------------------------------- #
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAHYOG - Autonomous SAR Operations Dashboard</title>
<style>
:root {
  --bg-dark: #07090e;
  --panel: #0f141c;
  --panel-light: #161e2b;
  --border: #222d3d;
  --border-focus: #3b82f6;
  --fg: #f1f5f9;
  --fg-dim: #94a3b8;
  --fg-sub: #64748b;
  
  --prio-imm: #ef4444;
  --prio-imm-glow: rgba(239, 68, 68, 0.35);
  --prio-del: #f59e0b;
  --prio-min: #10b981;
  --prio-exp: #6b7280;
  
  --brand: #38bdf8;
  --brand-glow: rgba(56, 189, 248, 0.3);
  --drone: #a855f7;
  --route: #22c55e;
  --beacon: #06b6d4;
  --hazard: #f97316;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg-dark);
  color: var(--fg);
  font-family: ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, monospace;
  font-size: 12px;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* Header */
header {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 8px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  z-index: 10;
}
.header-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-brand h1 {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: #fff;
  text-transform: uppercase;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.badge-live {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(16, 185, 129, 0.15);
  color: #10b981;
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 2px 7px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: 600;
}
.badge-live span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
  animation: pulse-dot 1.5s infinite;
}
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.telemetry-pills {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.pill {
  background: var(--panel-light);
  border: 1px solid var(--border);
  padding: 4px 9px;
  border-radius: 6px;
  font-size: 11px;
  color: var(--fg-dim);
  display: flex;
  align-items: center;
  gap: 6px;
}
.pill b { color: #fff; }
.pill.ok { border-color: rgba(16, 185, 129, 0.4); color: #10b981; }
.pill.warn { border-color: rgba(245, 158, 11, 0.4); color: #f59e0b; }
.pill.crit { border-color: rgba(239, 68, 68, 0.4); color: #ef4444; }

/* Main Grid */
main {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) 420px;
  gap: 1px;
  background: var(--border);
  min-height: 0;
}

/* Left Tactical Map Viewport */
.map-container {
  background: var(--bg-dark);
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  height: 100%;
}
#mapwrap {
  flex: 1;
  position: relative;
  min-height: 0;
  background: radial-gradient(circle at center, #0f1624 0%, #06090e 100%);
}
#cv {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

/* Map Overlays & HUD */
.map-hud-top {
  position: absolute;
  top: 12px;
  left: 12px;
  display: flex;
  gap: 8px;
  z-index: 5;
}
.hud-card {
  background: rgba(15, 20, 28, 0.88);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
}
.hud-title {
  font-size: 9.5px;
  text-transform: uppercase;
  color: var(--fg-sub);
  letter-spacing: 0.05em;
  margin-bottom: 3px;
}
.hud-val {
  font-size: 13px;
  font-weight: 700;
  color: #fff;
}

/* Map Controls */
.map-controls {
  position: absolute;
  top: 12px;
  right: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 5;
}
.btn-ctl {
  background: rgba(15, 20, 28, 0.9);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  color: var(--fg-dim);
  padding: 6px 10px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s;
}
.btn-ctl:hover {
  background: var(--panel-light);
  color: #fff;
  border-color: var(--brand);
}
.btn-ctl.active {
  background: rgba(56, 189, 248, 0.15);
  border-color: var(--brand);
  color: var(--brand);
}

/* Map Legend */
.map-legend {
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: rgba(15, 20, 28, 0.9);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 12px;
  font-size: 10.5px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 14px;
  z-index: 5;
}
.leg-item { display: flex; align-items: center; gap: 7px; }
.leg-dot { width: 8px; height: 8px; border-radius: 50%; }

/* Sensor HUD & Video Preview */
.sensor-hud-bar {
  height: 175px;
  background: var(--panel);
  border-top: 1px solid var(--border);
  display: grid;
  grid-template-columns: 1fr 1fr minmax(180px, 1fr);
  gap: 1px;
  background-color: var(--border);
}
.sensor-box {
  background: var(--panel);
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}
.sensor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.sensor-header h3 {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--brand);
}
.sensor-canvas-wrap {
  flex: 1;
  position: relative;
  background: #04060a;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.sensor-canvas-wrap canvas {
  width: 100%;
  height: 100%;
}

/* Right Command Deck */
.right-deck {
  background: var(--panel);
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
}
.deck-section {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
.deck-section h2 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-sub);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.deck-badge {
  background: var(--panel-light);
  border: 1px solid var(--border);
  padding: 1px 6px;
  border-radius: 8px;
  color: var(--fg-dim);
  font-size: 10px;
}

/* Triage Summary Counters */
.triage-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-bottom: 12px;
}
.triage-box {
  background: var(--panel-light);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 8px;
  text-align: center;
}
.triage-box.imm { border-color: rgba(239, 68, 68, 0.4); }
.triage-box.imm .tb-val { color: var(--prio-imm); }
.triage-box.del { border-color: rgba(245, 158, 11, 0.4); }
.triage-box.del .tb-val { color: var(--prio-del); }
.triage-box.min { border-color: rgba(16, 185, 129, 0.4); }
.triage-box.min .tb-val { color: var(--prio-min); }
.triage-box.drp { border-color: rgba(6, 182, 212, 0.4); }
.triage-box.drp .tb-val { color: var(--beacon); }
.tb-lbl { font-size: 9px; color: var(--fg-sub); text-transform: uppercase; }
.tb-val { font-size: 16px; font-weight: 700; margin-top: 2px; }

/* Survivor Human ID Cards */
.survivor-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.surv-card {
  background: var(--panel-light);
  border: 1px solid var(--border);
  border-left: 4px solid var(--prio-imm);
  border-radius: 6px;
  padding: 10px 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.surv-card:hover {
  border-color: var(--border-focus);
  transform: translateX(2px);
}
.surv-card.selected {
  background: #1b263b;
  border-color: var(--brand);
}
.surv-card.p-immediate { border-left-color: var(--prio-imm); }
.surv-card.p-delayed { border-left-color: var(--prio-del); }
.surv-card.p-minor { border-left-color: var(--prio-min); }

.surv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.surv-tag {
  font-size: 12px;
  font-weight: 700;
  color: #fff;
}
.prio-pill {
  padding: 2px 7px;
  border-radius: 10px;
  font-size: 9.5px;
  font-weight: 700;
  text-transform: uppercase;
}
.prio-pill.immediate { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
.prio-pill.delayed { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
.prio-pill.minor { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }

.surv-detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 8px;
  font-size: 11px;
  color: var(--fg-dim);
  margin-bottom: 8px;
}
.surv-detail-grid span b { color: #fff; font-weight: 600; }

.action-row {
  display: flex;
  gap: 6px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.btn-act {
  flex: 1;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.3);
  color: var(--brand);
  padding: 5px 8px;
  border-radius: 4px;
  font-size: 10.5px;
  font-weight: 600;
  cursor: pointer;
  text-align: center;
  transition: all 0.15s;
}
.btn-act:hover {
  background: var(--brand);
  color: #000;
}
.btn-act.dispatched {
  background: rgba(16, 185, 129, 0.2);
  border-color: rgba(16, 185, 129, 0.5);
  color: #10b981;
}

/* Event Log */
.event-log {
  max-height: 140px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
}
.log-entry {
  display: flex;
  gap: 8px;
  color: var(--fg-dim);
  line-height: 1.35;
}
.log-entry time { color: var(--fg-sub); font-size: 10px; }
.log-entry.crit { color: #ef4444; font-weight: 600; }
.log-entry.drop { color: #06b6d4; }
.log-entry.rescue { color: #10b981; }

/* Progress bars */
.progress-bar-wrap {
  height: 6px;
  background: #1e293b;
  border-radius: 3px;
  overflow: hidden;
  margin-top: 4px;
}
.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  width: 0%;
  transition: width 0.3s;
}
</style>
</head>
<body>

<header>
  <div class="header-brand">
    <h1>⚡ SAHYOG SAR COMMAND</h1>
    <span class="badge-live"><span></span> AUTONOMOUS ACTIVE</span>
  </div>
  
  <div class="telemetry-pills">
    <div class="pill" id="p-veh">VEHICLE <b>-</b></div>
    <div class="pill" id="p-nav">NAV <b>-</b></div>
    <div class="pill" id="p-link">LORA <b>-</b></div>
    <div class="pill" id="p-queue">QUEUE <b>-</b></div>
    <div class="pill" id="p-time">00:00:00</div>
  </div>
</header>

<main>
  <!-- Tactical Map Center -->
  <div class="map-container">
    <div id="mapwrap">
      <canvas id="cv"></canvas>
      
      <!-- Top HUD Stats -->
      <div class="map-hud-top">
        <div class="hud-card">
          <div class="hud-title">Search Coverage</div>
          <div class="hud-val" id="hud-cov">0.0%</div>
        </div>
        <div class="hud-card">
          <div class="hud-title">Ground Resolution</div>
          <div class="hud-val" id="hud-gsd">0.14 m/px</div>
        </div>
        <div class="hud-card">
          <div class="hud-title">Airspeed / Alt</div>
          <div class="hud-val" id="hud-speed">0.0 m/s &middot; 0m</div>
        </div>
      </div>
      
      <!-- Interactive Map Layer Toggles -->
      <div class="map-controls">
        <button class="btn-ctl active" id="btn-fov" onclick="toggleLayer('fov')">🔭 Sensor Cone</button>
        <button class="btn-ctl active" id="btn-hazards" onclick="toggleLayer('hazards')">🔥 Hazard Zones</button>
        <button class="btn-ctl active" id="btn-routes" onclick="toggleLayer('routes')">🚑 Safe Routes</button>
        <button class="btn-ctl active" id="btn-beacons" onclick="toggleLayer('beacons')">📡 Beacons & Drops</button>
      </div>
      
      <!-- Map Legend -->
      <div class="map-legend">
        <div class="leg-item"><div class="leg-dot" style="background:var(--prio-imm);box-shadow:0 0 6px var(--prio-imm)"></div> Immediate (Red)</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--prio-del)"></div> Delayed (Yellow)</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--drone)"></div> Search Drone</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--beacon)"></div> Airdrop / Beacon</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--route)"></div> Safe Access Path</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--hazard)"></div> Hazard / Flood</div>
      </div>
    </div>
    
    <!-- Lower Dual-Spectrum Synthetic Sensor Feeds -->
    <div class="sensor-hud-bar">
      <!-- LWIR Thermal Feed -->
      <div class="sensor-box">
        <div class="sensor-header">
          <h3>LWIR Thermal Feed</h3>
          <span class="deck-badge" id="th-temp">24.5°C</span>
        </div>
        <div class="sensor-canvas-wrap">
          <canvas id="cv-thermal"></canvas>
        </div>
      </div>
      
      <!-- Visible RGB Camera Feed -->
      <div class="sensor-box">
        <div class="sensor-header">
          <h3>RGB Target Feed</h3>
          <span class="deck-badge" id="rgb-res">1920x1080</span>
        </div>
        <div class="sensor-canvas-wrap">
          <canvas id="cv-rgb"></canvas>
        </div>
      </div>
      
      <!-- Operations Quick Actions -->
      <div class="sensor-box" style="background:var(--panel-light)">
        <div class="sensor-header">
          <h3>Mission Quick Ops</h3>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;margin-top:4px">
          <button class="btn-act" onclick="triggerBatchDrop()">⚡ AUTO-DROP ALL CRITICAL</button>
          <button class="btn-act" onclick="triggerDispatchAll()">🚑 DISPATCH ALL RESCUE TEAMS</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Right Command Deck -->
  <div class="right-deck">
    <!-- Triage & Identification Summary -->
    <div class="deck-section">
      <h2>Survivor Triage &amp; Identification <span class="deck-badge" id="surv-total-badge">0 Found</span></h2>
      <div class="triage-grid">
        <div class="triage-box imm">
          <div class="tb-lbl">Immediate</div>
          <div class="tb-val" id="cnt-imm">0</div>
        </div>
        <div class="triage-box del">
          <div class="tb-lbl">Delayed</div>
          <div class="tb-val" id="cnt-del">0</div>
        </div>
        <div class="triage-box min">
          <div class="tb-lbl">Minor</div>
          <div class="tb-val" id="cnt-min">0</div>
        </div>
        <div class="triage-box drp">
          <div class="tb-lbl">Airdrops</div>
          <div class="tb-val" id="cnt-drp">0</div>
        </div>
      </div>
    </div>

    <!-- Survivor Profile List -->
    <div class="deck-section" style="flex:1;min-height:0;overflow-y:auto">
      <h2>Identified Humans &amp; Response Ops</h2>
      <div class="survivor-list" id="surv-cards">
        <div style="color:var(--fg-sub);padding:12px;text-align:center">Scanning search area for survivors...</div>
      </div>
    </div>

    <!-- Event & Rescue Telemetry Log -->
    <div class="deck-section">
      <h2>Mission Event Stream</h2>
      <div class="event-log" id="event-log"></div>
    </div>
  </div>
</main>

<script>
// State Store
const STATE = {
  data: null,
  selectedId: null,
  layers: { fov: true, hazards: true, routes: true, beacons: true },
  events: [],
  animTime: 0,
  propAngle: 0,
  wavingPhase: 0,
};

const cv = document.getElementById('cv');
const cx = cv.getContext('2d');
const cvThermal = document.getElementById('cv-thermal');
const cxThermal = cvThermal.getContext('2d');
const cvRgb = document.getElementById('cv-rgb');
const cxRgb = cvRgb.getContext('2d');

function toggleLayer(name) {
  STATE.layers[name] = !STATE.layers[name];
  const btn = document.getElementById('btn-' + name);
  if (btn) btn.classList.toggle('active', STATE.layers[name]);
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = r.width * dpr;
  cv.height = r.height * dpr;
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);

  [cvThermal, cvRgb].forEach(c => {
    const br = c.parentElement.getBoundingClientRect();
    c.width = br.width * dpr;
    c.height = br.height * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  });
}
window.addEventListener('resize', resizeCanvases);

// Map Coordinate Transformer
function getTransform() {
  const st = STATE.data;
  const w = cv.clientWidth, h = cv.clientHeight;
  const W = (st && st.world && st.world.east_m) || 1000;
  const H = (st && st.world && st.world.north_m) || 1000;
  const scale = Math.min(w / W, h / H) * 0.92;
  const ox = (w - W * scale) / 2;
  const oy = (h - H * scale) / 2;
  return {
    X: e => ox + e * scale,
    Y: n => oy + (H - n) * scale,
    scale,
    W, H
  };
}

// -------------------------------------------------------------
// 60FPS Animation & Tactical Drawing Loop
// -------------------------------------------------------------
function render() {
  STATE.animTime += 0.016;
  STATE.propAngle += 0.45;
  STATE.wavingPhase += 0.08;

  const st = STATE.data;
  const w = cv.clientWidth, h = cv.clientHeight;
  cx.clearRect(0, 0, w, h);

  if (st) {
    const { X, Y, scale, W, H } = getTransform();

    // 1. Draw Ground Grid & Terrain Contours
    cx.strokeStyle = 'rgba(255,255,255,0.04)';
    cx.lineWidth = 1;
    for (let x = 0; x <= W; x += 100) {
      cx.beginPath(); cx.moveTo(X(x), Y(0)); cx.lineTo(X(x), Y(H)); cx.stroke();
    }
    for (let y = 0; y <= H; y += 100) {
      cx.beginPath(); cx.moveTo(X(0), Y(y)); cx.lineTo(X(W), Y(y)); cx.stroke();
    }

    // 2. Coverage Grid
    const g = st.coverage_grid;
    if (g && g.rows && g.rows.length) {
      const cs = g.res * scale;
      for (let r = 0; r < g.rows.length; r++) {
        for (let c = 0; c < g.rows[r].length; c++) {
          const v = g.rows[r][c];
          if (v < 0.02) continue;
          const n0 = r * g.res, e0 = c * g.res;
          cx.fillStyle = `rgba(56, 189, 248, ${0.05 + 0.45 * Math.min(v, 1)})`;
          cx.fillRect(X(e0), Y(n0 + g.res), Math.max(cs, 1.2), Math.max(cs, 1.2));
        }
      }
    }

    // 3. Search Box Boundary
    if (st.plan && st.plan.search_box) {
      const b = st.plan.search_box;
      cx.strokeStyle = '#38bdf8';
      cx.lineWidth = 1.5;
      cx.setLineDash([6, 6]);
      cx.strokeRect(X(b.east_m[0]), Y(b.north_m[1]), (b.east_m[1] - b.east_m[0]) * scale, (b.north_m[1] - b.north_m[0]) * scale);
      cx.setLineDash([]);
    }

    // 4. Hazards & Fire Plumes
    if (STATE.layers.hazards && st.hazards) {
      st.hazards.forEach(hz => {
        const hx = X(hz.east_m), hy = Y(hz.north_m);
        const rad = Math.max((hz.radius_m || 20) * scale, 6);
        
        // Hazard Danger Zone
        cx.beginPath();
        cx.arc(hx, hy, rad, 0, Math.PI * 2);
        cx.fillStyle = 'rgba(249, 115, 22, 0.18)';
        cx.fill();
        cx.strokeStyle = '#f97316';
        cx.lineWidth = 1.5;
        cx.stroke();

        // Animated Flame/Smoke Embers
        const flicker = Math.sin(STATE.animTime * 6 + hz.north_m) * 3;
        cx.beginPath();
        cx.arc(hx + flicker, hy - flicker, rad * 0.4, 0, Math.PI * 2);
        cx.fillStyle = 'rgba(239, 68, 68, 0.5)';
        cx.fill();
        
        cx.fillStyle = '#f97316';
        cx.font = '9px monospace';
        cx.fillText((hz.hazard_class || 'HAZARD').toUpperCase(), hx + rad + 4, hy + 3);
      });
    }

    // 5. Safe Response Routes (A* Pathfinding Corridors)
    if (STATE.layers.routes && st.rescue_routes) {
      st.rescue_routes.forEach(route => {
        if (!route.waypoints || route.waypoints.length < 2) return;
        cx.beginPath();
        cx.moveTo(X(route.waypoints[0].east_m), Y(route.waypoints[0].north_m));
        for (let i = 1; i < route.waypoints.length; i++) {
          cx.lineTo(X(route.waypoints[i].east_m), Y(route.waypoints[i].north_m));
        }
        cx.strokeStyle = 'rgba(34, 197, 94, 0.75)';
        cx.lineWidth = 2.5;
        cx.setLineDash([4, 4]);
        cx.stroke();
        cx.setLineDash([]);

        // Animated Rescue Vehicle / Boat moving along path
        const tPath = (STATE.animTime * 0.25) % 1.0;
        const ptIdx = Math.floor(tPath * (route.waypoints.length - 1));
        const p1 = route.waypoints[ptIdx], p2 = route.waypoints[Math.min(ptIdx + 1, route.waypoints.length - 1)];
        const subFrac = (tPath * (route.waypoints.length - 1)) - ptIdx;
        const curE = p1.east_m + (p2.east_m - p1.east_m) * subFrac;
        const curN = p1.north_m + (p2.north_m - p1.north_m) * subFrac;

        cx.beginPath();
        cx.arc(X(curE), Y(curN), 5.5, 0, Math.PI * 2);
        cx.fillStyle = '#22c55e';
        cx.fill();
        cx.strokeStyle = '#fff';
        cx.lineWidth = 1.5;
        cx.stroke();
      });
    }

    // 6. Locator Beacons & Airdrops
    if (STATE.layers.beacons && st.airdrops) {
      st.airdrops.forEach(drop => {
        const dx = X(drop.impact_east_m), dy = Y(drop.impact_north_m);
        // Expanding Sonar/Radar Ring
        const pulse = (STATE.animTime * 1.5) % 1.0;
        cx.beginPath();
        cx.arc(dx, dy, pulse * 32, 0, Math.PI * 2);
        cx.strokeStyle = `rgba(6, 182, 212, ${1 - pulse})`;
        cx.lineWidth = 2;
        cx.stroke();

        // Parachute Kit Marker
        cx.beginPath();
        cx.arc(dx, dy, 4.5, 0, Math.PI * 2);
        cx.fillStyle = '#06b6d4';
        cx.fill();
        cx.strokeStyle = '#fff';
        cx.stroke();
        cx.fillStyle = '#06b6d4';
        cx.font = '9px monospace';
        cx.fillText('RELIEF DELIVERED', dx + 7, dy + 3);
      });
    }

    // 7. Survivors & Human Postures / SOS Waving
    (st.survivors || []).forEach(sv => {
      const sx = X(sv.east_m), sy = Y(sv.north_m);
      const isSel = STATE.selectedId === sv.id;
      const prio = (sv.pr || sv.priority || 'immediate').toLowerCase();
      const col = prio.includes('imm') ? '#ef4444' : (prio.includes('del') ? '#f59e0b' : '#10b981');
      const sg = (sv.sg || sv.sigma_m || 3.0);

      // Uncertainty Ellipse
      cx.beginPath();
      cx.arc(sx, sy, Math.max(sg * 2 * scale, 8), 0, Math.PI * 2);
      cx.strokeStyle = col + '77';
      cx.lineWidth = 1.2;
      cx.setLineDash([3, 3]);
      cx.stroke();
      cx.setLineDash([]);

      // Vital Pulsing Glow
      const vitalPulse = Math.sin(STATE.animTime * 4 + sv.north_m) * 0.5 + 0.5;
      cx.beginPath();
      cx.arc(sx, sy, 7 + vitalPulse * 3, 0, Math.PI * 2);
      cx.fillStyle = col + '33';
      cx.fill();

      // Human Center Silhouette
      cx.beginPath();
      cx.arc(sx, sy, isSel ? 7 : 5, 0, Math.PI * 2);
      cx.fillStyle = col;
      cx.fill();
      cx.strokeStyle = '#07090e';
      cx.lineWidth = 2;
      cx.stroke();

      // Animated SOS Waving Arms!
      if (sv.sos_waving || sv.distress_level === 'extreme_distress' || sv.posture === 'waving') {
        const waveAngle = Math.sin(STATE.wavingPhase * 3) * 0.65;
        cx.strokeStyle = '#fff';
        cx.lineWidth = 2;
        // Left arm
        cx.beginPath();
        cx.moveTo(sx - 3, sy);
        cx.lineTo(sx - 9, sy - 8 + waveAngle * 6);
        cx.stroke();
        // Right arm
        cx.beginPath();
        cx.moveTo(sx + 3, sy);
        cx.lineTo(sx + 9, sy - 8 - waveAngle * 6);
        cx.stroke();
      }

      // ID Tag
      cx.fillStyle = '#fff';
      cx.font = '10px monospace';
      cx.fillText(`${sv.id} [${(sv.profile && sv.profile.posture) || sv.posture || 'HUMAN'}]`, sx + 10, sy + 3);
    });

    // 8. Search & Rescue Autonomous Drone with Spinning Rotors & FOV Cone
    const v = st.vehicle;
    if (v && v.n_m != null && v.e_m != null) {
      const vx = X(v.e_m), vy = Y(v.n_m);
      const yaw = ((v.yaw_deg || 0) - 90) * Math.PI / 180;

      // Sensor Field of View Scanning Spotlight Cone
      if (STATE.layers.fov) {
        const alt = v.alt_rel_m || 40.0;
        const fovRad = Math.max(alt * 0.7 * scale, 24);
        const fovSweep = STATE.animTime * 2.0;

        cx.save();
        cx.translate(vx, vy);
        cx.rotate(yaw);

        // Projected Elliptical Spotlight Cone
        const grad = cx.createRadialGradient(0, 0, 4, 0, 0, fovRad);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
        grad.addColorStop(0.7, 'rgba(56, 189, 248, 0.12)');
        grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
        cx.fillStyle = grad;
        cx.beginPath();
        cx.arc(0, 0, fovRad, 0, Math.PI * 2);
        cx.fill();

        // Active Radar Sweep Ray
        cx.strokeStyle = 'rgba(56, 189, 248, 0.8)';
        cx.lineWidth = 1.5;
        cx.beginPath();
        cx.moveTo(0, 0);
        cx.lineTo(Math.cos(fovSweep) * fovRad, Math.sin(fovSweep) * fovRad);
        cx.stroke();

        cx.restore();
      }

      // Drone Airframe & Spinning 4-Rotors
      cx.save();
      cx.translate(vx, vy);
      cx.rotate(yaw);

      // Drone Arms
      cx.strokeStyle = '#fff';
      cx.lineWidth = 2.5;
      cx.beginPath();
      cx.moveTo(-10, -10); cx.lineTo(10, 10);
      cx.moveTo(10, -10); cx.lineTo(-10, 10);
      cx.stroke();

      // Drone Central Fuselage
      cx.beginPath();
      cx.arc(0, 0, 6, 0, Math.PI * 2);
      cx.fillStyle = '#a855f7';
      cx.fill();
      cx.strokeStyle = '#fff';
      cx.lineWidth = 1.5;
      cx.stroke();

      // Heading indicator
      cx.beginPath();
      cx.moveTo(0, -6); cx.lineTo(0, -14);
      cx.strokeStyle = '#38bdf8';
      cx.lineWidth = 2;
      cx.stroke();

      // 4 Spinning Propellers
      [[-10,-10], [10,-10], [-10,10], [10,10]].forEach(([px, py]) => {
        cx.save();
        cx.translate(px, py);
        cx.rotate(STATE.propAngle);
        cx.strokeStyle = 'rgba(168, 85, 247, 0.9)';
        cx.lineWidth = 2;
        cx.beginPath();
        cx.moveTo(-6, 0); cx.lineTo(6, 0);
        cx.stroke();
        cx.restore();
      });

      cx.restore();
    }
  }

  // Render Synthetic HUDs
  drawSyntheticThermal();
  drawSyntheticRgb();

  requestAnimationFrame(render);
}

// -------------------------------------------------------------
// Synthetic LWIR Thermal & RGB Camera Renderers
// -------------------------------------------------------------
function drawSyntheticThermal() {
  const w = cvThermal.clientWidth, h = cvThermal.clientHeight;
  cxThermal.fillStyle = '#060a12';
  cxThermal.fillRect(0, 0, w, h);

  // False-color Ironbow background gradient
  const grad = cxThermal.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, '#0a0d1a');
  grad.addColorStop(0.5, '#16132b');
  grad.addColorStop(1, '#241029');
  cxThermal.fillStyle = grad;
  cxThermal.fillRect(0, 0, w, h);

  // Crosshairs
  cxThermal.strokeStyle = 'rgba(56, 189, 248, 0.3)';
  cxThermal.lineWidth = 1;
  cxThermal.beginPath();
  cxThermal.moveTo(w/2, 0); cxThermal.lineTo(w/2, h);
  cxThermal.moveTo(0, h/2); cxThermal.lineTo(w, h/2);
  cxThermal.stroke();

  // Synthetic Target Hotspot
  const st = STATE.data;
  const selSurv = st && st.survivors && st.survivors[0];
  const hotX = w/2 + Math.sin(STATE.animTime * 1.5) * 20;
  const hotY = h/2 + Math.cos(STATE.animTime * 1.2) * 12;

  const hotGrad = cxThermal.createRadialGradient(hotX, hotY, 2, hotX, hotY, 35);
  hotGrad.addColorStop(0, '#ffffff'); // White hot core (37.2°C)
  hotGrad.addColorStop(0.25, '#f59e0b');
  hotGrad.addColorStop(0.65, '#dc2626');
  hotGrad.addColorStop(1, 'rgba(124, 58, 237, 0)');
  cxThermal.fillStyle = hotGrad;
  cxThermal.beginPath();
  cxThermal.arc(hotX, hotY, 35, 0, Math.PI * 2);
  cxThermal.fill();

  // Target Lock Box
  cxThermal.strokeStyle = '#38bdf8';
  cxThermal.strokeRect(hotX - 18, hotY - 24, 36, 48);
  cxThermal.fillStyle = '#38bdf8';
  cxThermal.font = '9px monospace';
  cxThermal.fillText('TARGET: 37.1°C', hotX - 18, hotY - 28);
}

function drawSyntheticRgb() {
  const w = cvRgb.clientWidth, h = cvRgb.clientHeight;
  cxRgb.fillStyle = '#080d17';
  cxRgb.fillRect(0, 0, w, h);

  // Synthetic Visible Terrain Shading
  cxRgb.fillStyle = '#0f172a';
  cxRgb.fillRect(10, 10, w - 20, h - 20);

  const hotX = w/2 + Math.sin(STATE.animTime * 1.5) * 20;
  const hotY = h/2 + Math.cos(STATE.animTime * 1.2) * 12;

  // AI Bounding Box & Saliency Detection
  cxRgb.strokeStyle = '#22c55e';
  cxRgb.lineWidth = 1.5;
  cxRgb.strokeRect(hotX - 16, hotY - 22, 32, 44);

  // Label Chip
  cxRgb.fillStyle = 'rgba(34, 197, 94, 0.85)';
  cxRgb.fillRect(hotX - 16, hotY - 34, 78, 12);
  cxRgb.fillStyle = '#000';
  cxRgb.font = '8.5px monospace';
  cxRgb.fontWeight = 'bold';
  cxRgb.fillText('HUMAN 94% [WAV]', hotX - 14, hotY - 25);
}

// -------------------------------------------------------------
// Telemetry & Server Polling Loop
// -------------------------------------------------------------
async function pollState() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    const data = await res.json();
    const prev = STATE.data;
    STATE.data = data;

    // Header updates
    const V = data.vehicle || {}, N = data.nav || {}, L = data.link || {}, Q = data.queue || {};
    document.getElementById('p-veh').innerHTML = `VEHICLE <b>${V.mode||'GUIDED'}</b> ${V.armed?'ARMED':'DISARMED'} &middot; ${(V.alt_rel_m||0).toFixed(0)}m &middot; ${(V.gs_ms||0).toFixed(1)}m/s &middot; BATT ${(V.batt_pct||100).toFixed(0)}%`;
    document.getElementById('p-nav').innerHTML = `NAV <b>${N.source||'EKF'}</b> &plusmn;${(N.pos_sigma_m||0.8).toFixed(1)}m`;
    document.getElementById('p-link').innerHTML = `LORA <b>${L.in_range?'CONNECTED':'OFFLINE'}</b> (${(L.sent_packets||0)} pkts)`;
    document.getElementById('p-queue').innerHTML = `QUEUE <b>${Q.items||0}</b> (${((Q.bytes||0)/1024).toFixed(1)} KB)`;
    document.getElementById('p-time').textContent = new Date().toLocaleTimeString();

    // Map HUD
    const cov = data.coverage || {};
    const effCov = cov.effective_coverage != null ? (cov.effective_coverage * 100).toFixed(1) : '0.0';
    document.getElementById('hud-cov').textContent = effCov + '%';
    document.getElementById('hud-speed').textContent = `${(V.gs_ms||0).toFixed(1)} m/s &middot; ${(V.alt_rel_m||0).toFixed(0)}m AGL`;

    // Triage Counters
    const survs = data.survivors || [];
    let imm = 0, del = 0, min = 0;
    survs.forEach(s => {
      const p = (s.pr || s.priority || '').toLowerCase();
      if (p.includes('imm') || p === 'i') imm++;
      else if (p.includes('del') || p === 'd') del++;
      else min++;
    });
    document.getElementById('cnt-imm').textContent = imm;
    document.getElementById('cnt-del').textContent = del;
    document.getElementById('cnt-min').textContent = min;
    document.getElementById('cnt-drp').textContent = (data.airdrops || []).length;
    document.getElementById('surv-total-badge').textContent = `${survs.length} Found`;

    // Render Survivor Profile Cards
    renderSurvivorCards(survs);

    // Event Log Sync
    if (prev && survs.length > (prev.survivors || []).length) {
      const nw = survs.slice((prev.survivors || []).length);
      nw.forEach(s => logEvent(`SURVIVOR DETECTED: ${s.id} (Prio: ${s.pr||s.priority||'IMMEDIATE'})`, 'crit'));
    }

  } catch (err) {
    console.warn("Poll state error:", err);
  }
}

function logEvent(msg, type='') {
  STATE.events.unshift({ t: new Date(), msg, type });
  if (STATE.events.length > 50) STATE.events.pop();
  const el = document.getElementById('event-log');
  el.innerHTML = STATE.events.map(e => `
    <div class="log-entry ${e.type}">
      <time>${e.t.toLocaleTimeString()}</time>
      <span>${e.msg}</span>
    </div>
  `).join('');
}

function renderSurvivorCards(survivors) {
  const container = document.getElementById('surv-cards');
  if (!survivors || !survivors.length) {
    container.innerHTML = '<div style="color:var(--fg-sub);padding:12px;text-align:center">Scanning search area for survivors...</div>';
    return;
  }

  container.innerHTML = survivors.map(sv => {
    const pr = (sv.pr || sv.priority || 'immediate').toLowerCase();
    const prioClass = pr.includes('imm') ? 'immediate' : (pr.includes('del') ? 'delayed' : 'minor');
    const isSel = STATE.selectedId === sv.id;
    const prof = sv.profile || {};
    const posture = prof.posture || sv.posture || 'standing';
    const distress = prof.distress_level || sv.distress_level || 'moderate_distress';
    const coreTemp = prof.estimated_core_temp_c || sv.core_temp_c || 36.8;
    const equip = prof.recommended_equipment || sv.recommended_equipment || 'trauma_kit';
    const groupSize = sv.group_size || sv.gs || 1;

    return `
      <div class="surv-card p-${prioClass} ${isSel ? 'selected' : ''}" onclick="selectSurvivor('${sv.id}')">
        <div class="surv-header">
          <span class="surv-tag">${sv.id} &middot; ${posture.toUpperCase()}</span>
          <span class="prio-pill ${prioClass}">${prioClass}</span>
        </div>
        <div class="surv-detail-grid">
          <span>Coords: <b>${(sv.north_m||0).toFixed(0)}m N, ${(sv.east_m||0).toFixed(0)}m E</b></span>
          <span>Uncertainty: <b>&plusmn;${(sv.sigma_m||sv.sg||1.5).toFixed(1)}m</b></span>
          <span>Core Temp: <b>${coreTemp.toFixed(1)}°C</b></span>
          <span>Distress: <b>${distress.replace('_', ' ')}</b></span>
          <span>Group Size: <b>${groupSize} Person(s)</b></span>
          <span>Equipment: <b>${equip.replace('_', ' ')}</b></span>
        </div>
        <div class="action-row">
          <button class="btn-act" onclick="event.stopPropagation(); dropPayload('${sv.id}', '${equip}')">🪂 AIR-DROP ${equip.toUpperCase()}</button>
          <button class="btn-act" onclick="event.stopPropagation(); dispatchTeam('${sv.id}')">🚑 DISPATCH SQUAD</button>
        </div>
      </div>
    `;
  }).join('');
}

function selectSurvivor(id) {
  STATE.selectedId = STATE.selectedId === id ? null : id;
}

// -------------------------------------------------------------
// Interactive Operations API Calls
// -------------------------------------------------------------
async function dropPayload(survId, payloadType) {
  logEvent(`COMMANDED PRECISION AIR-DROP: ${payloadType} -> ${survId}`, 'drop');
  try {
    await fetch('/api/rescue/drop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: survId, payload_type: payloadType })
    });
    pollState();
  } catch (e) {
    console.error(e);
  }
}

async function dispatchTeam(survId) {
  logEvent(`DISPATCHED EMERGENCY RESCUE SQUAD -> ${survId}`, 'rescue');
  try {
    await fetch('/api/rescue/dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: survId })
    });
    pollState();
  } catch (e) {
    console.error(e);
  }
}

async function triggerBatchDrop() {
  logEvent('AUTO-DROP SEQUENCE TRIGGERED FOR ALL CRITICAL SURVIVORS', 'drop');
  const survs = (STATE.data && STATE.data.survivors) || [];
  for (const s of survs) {
    dropPayload(s.id, s.recommended_equipment || 'first_aid_kit');
  }
}

async function triggerDispatchAll() {
  logEvent('DEPLOYED ALL AVAILABLE GROUND & BOAT RESCUE TEAMS', 'rescue');
  const survs = (STATE.data && STATE.data.survivors) || [];
  for (const s of survs) {
    dispatchTeam(s.id);
  }
}

// Init
resizeCanvases();
pollState();
setInterval(pollState, 1000);
requestAnimationFrame(render);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
class Dashboard:
    """Serves the command-centre page and the state it draws from."""

    def __init__(self, uplink: Any, runner: Any = None, world: Any = None,
                 host: str = "0.0.0.0", port: int = 8088,
                 coverage_downsample: int = 8) -> None:
        if not _HAS_FASTAPI:
            raise RuntimeError(
                "the dashboard needs fastapi and uvicorn; install them or run "
                "without --live (the mission still records everything to artifacts/)")
        self.uplink = uplink
        self.runner = runner
        self.world = world
        self.host = host
        self.port = int(port)
        self.coverage_downsample = int(coverage_downsample)
        self._server: Optional[threading.Thread] = None
        self._uvicorn: Any = None
        self._stop = threading.Event()
        self.app = self._build_app()

    # ------------------------------------------------------------------ #
    def _build_app(self) -> Any:
        app = FastAPI(title="SAHYOG SAR Command Centre", docs_url=None, redoc_url=None,
                      openapi_url=None)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _PAGE

        @app.get("/api/state")
        def state() -> JSONResponse:
            try:
                return JSONResponse(self.snapshot())
            except Exception as exc:                  # pragma: no cover
                log.exception("dashboard snapshot failed")
                return JSONResponse({"error": repr(exc)}, status_code=500)

        @app.post("/api/rescue/drop")
        async def api_drop(req: Request) -> JSONResponse:
            try:
                data = await req.json()
                tid = data.get("target_id")
                ptype = data.get("payload_type", "first_aid_kit")
                if self.runner and hasattr(self.runner, "rescue_coord"):
                    # Find integer target ID
                    int_tid = int(str(tid).replace("p", "").replace("S", "") or 0)
                    state = self.runner._state(self.runner._elapsed(), self.runner.conn.telemetry)
                    res = self.runner.rescue_coord.execute_payload_drop(
                        target_id=int_tid,
                        aircraft_pos_ned=state.pos,
                        aircraft_vel_ned=state.vel,
                        t_mission=self.runner._elapsed(),
                    )
                    return JSONResponse({"status": "success", "drop": res.to_dict() if res else None})
                return JSONResponse({"status": "acknowledged"})
            except Exception as exc:
                log.error("API drop error: %r", exc)
                return JSONResponse({"status": "error", "error": repr(exc)}, status_code=400)

        @app.post("/api/rescue/dispatch")
        async def api_dispatch(req: Request) -> JSONResponse:
            try:
                data = await req.json()
                tid = data.get("target_id")
                int_tid = int(str(tid).replace("p", "").replace("S", "") or 0)
                if self.runner and hasattr(self.runner, "rescue_coord"):
                    from sar.rescue.routing import RescueTeamType
                    self.runner.rescue_coord.dispatch_ground_team(
                        target_id=int_tid,
                        team_type=RescueTeamType.FOOT_RESCUE_TEAM,
                        t_mission=self.runner._elapsed()
                    )
                return JSONResponse({"status": "dispatched"})
            except Exception as exc:
                log.error("API dispatch error: %r", exc)
                return JSONResponse({"status": "error", "error": repr(exc)}, status_code=400)

        @app.get("/api/health")
        def health() -> Dict[str, Any]:
            return {"ok": True, "t": time.time()}

        return app

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """Everything the dashboard page draws."""
        ground = self.uplink.report.to_dict()
        out: Dict[str, Any] = {
            "t": time.time(),
            "received": ground["received"],
            "survivors": ground["survivors"],
            "hazards": ground["hazards"],
            "coverage": ground.get("coverage") or {},
            "nav": ground.get("nav") or {},
            "vehicle": ground.get("vehicle") or {},
            "queue": self.uplink.queue.snapshot(),
            "link": self.uplink.link.report(),
            "gcs": {"n": 0.0, "e": 0.0},
            "airdrops": [],
            "rescue_routes": [],
        }

        if self.runner is not None:
            g = getattr(self.runner, "gcs", None)
            if g is not None:
                out["gcs"] = {"n": float(g[0]), "e": float(g[1])}
            plan = getattr(self.runner, "plan", None)
            if plan is not None:
                out["plan"] = plan.to_dict()
            cov = getattr(self.runner, "coverage", None)
            if cov is not None:
                out["coverage_grid"] = self._grid(cov)
                box = (plan.to_dict().get("search_box") if plan is not None else None)
                if box:
                    out["coverage"]["search_box"] = box
                    out["coverage"]["box_effective"] = round(
                        self.runner.coverage_in_box(box).get("effective_coverage", 0.0), 4)

            # Rescue coordinator integration
            rc = getattr(self.runner, "rescue_coord", None)
            if rc is not None:
                out["airdrops"] = [d.to_dict() for d in rc.completed_drops]
                routes = []
                for op in rc.operations.values():
                    for rname, rpath in op.ground_routes.items():
                        routes.append(rpath.to_dict())
                out["rescue_routes"] = routes

            if self.world is not None:
                out["truth"] = [{"vid": v.vid, "north_m": round(float(v.north), 1),
                                 "east_m": round(float(v.east), 1),
                                 "alive": bool(v.alive)}
                                for v in getattr(self.world, "victims", [])]

        if self.world is not None:
            out["world"] = {"north_m": float(self.world.north_m),
                            "east_m": float(self.world.east_m),
                            "name": getattr(self.world.spec, "name", "")}

        origin = getattr(self.world, "origin", None) if self.world is not None else None
        for i, sv in enumerate(out["survivors"]):
            sv["id"] = sv.get("_key") or f"S{i}"
            if sv.get("north_m") is None:
                la, lo = sv.get("la", sv.get("lat")), sv.get("lo", sv.get("lon"))
                if la is not None and lo is not None and origin is not None:
                    from sar.core.geo import GeoPoint, wgs84_to_local
                    n, e, _ = wgs84_to_local(GeoPoint(la, lo, 0.0), origin)
                    sv["north_m"], sv["east_m"] = round(n, 1), round(e, 1)
                    sv["lat"], sv["lon"] = la, lo
        return out

    def _grid(self, cov: Any) -> Dict[str, Any]:
        """Coverage as a downsampled 2-D array of rounded probabilities."""
        step = max(1, self.coverage_downsample)
        g = cov.p_detected[::step, ::step]
        g = np.where(g > 0.02, np.round(g, 2), 0.0)
        return {"res": cov.res * step, "rows": g.tolist()}

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Serve on a daemon thread."""
        if self._server is not None:
            return
        cfg = uvicorn.Config(self.app, host=self.host, port=self.port,
                             log_level="warning", access_log=False)
        self._uvicorn = uvicorn.Server(cfg)

        def _run() -> None:
            try:
                self._uvicorn.run()
            except Exception as exc:                  # pragma: no cover
                log.error("dashboard server stopped: %r", exc)

        self._server = threading.Thread(target=_run, daemon=True, name="sar-dashboard")
        self._server.start()
        for _ in range(60):
            if getattr(self._uvicorn, "started", False):
                log.info("dashboard listening on http://%s:%d", self.host, self.port)
                return
            time.sleep(0.1)
        log.warning("dashboard did not confirm startup within 6 s")

    def stop(self) -> None:
        self._stop.set()
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        self._server = None


if __name__ == "__main__":
    import argparse
    from sar.comms.link import TelemetryUplink
    from sar.sim.scenario import build_reference_scenario

    parser = argparse.ArgumentParser(description="SAR Tactical Command Dashboard")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--scenario", default="flood", help="Scenario preset")
    args = parser.parse_args()

    world, _, _ = build_reference_scenario(args.scenario, smoke=True)
    uplink = TelemetryUplink()
    uplink.start()
    dash = Dashboard(uplink=uplink, world=world, host=args.host, port=args.port)
    print(f"Serving SAR Command Dashboard at http://{args.host}:{args.port}")
    uvicorn.run(dash.app, host=args.host, port=args.port)

