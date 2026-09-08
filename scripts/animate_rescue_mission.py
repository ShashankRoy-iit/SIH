#!/usr/bin/env python3
"""Autonomous SAR Rescue Mission Animation Generator.

Generates:
1. High-fidelity standalone interactive HTML5 animated visualizer (`artifacts/rescue_mission_animation.html`)
   with full 60fps canvas rendering, drone kinematics, rotating rotors, sensor spotlight cone,
   waving SOS survivors, ballistic parachute drops, radio beacons, and safe evacuation corridors.
2. Animated GIF/MP4 export using Matplotlib (`artifacts/rescue_mission.gif`) for offline playback.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sar.core.geo import GeoPoint
from sar.perception.human_id import (
    DistressLevel,
    HumanIdentifier,
    HumanPosture,
    HumanProfile,
    RescueEquipmentNeed,
)
from sar.rescue.coordinator import RescueCoordinator
from sar.rescue.payload import BallisticDropCalculator, PAYLOAD_SPECS, RescuePayloadType
from sar.rescue.routing import GroundRescueRouter, RescueTeamType
from sar.sim.scenario import build_reference_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sar.animation")


# --------------------------------------------------------------------------- #
# Standalone Interactive HTML5 Animation Builder
# --------------------------------------------------------------------------- #
def generate_html_animation(
    scenario_name: str,
    mission_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate a zero-dependency, self-contained interactive animated HTML5 player."""
    json_data_str = json.dumps(mission_data)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAHYOG - Mission Animation Player ({scenario_name})</title>
<style>
:root {{
  --bg-dark: #06090e;
  --panel: #0d121a;
  --panel-light: #151c27;
  --border: #1f2937;
  --fg: #f8fafc;
  --fg-dim: #94a3b8;
  --brand: #38bdf8;
  --brand-glow: rgba(56, 189, 248, 0.4);
  --imm: #ef4444;
  --del: #f59e0b;
  --min: #10b981;
  --drone: #a855f7;
  --route: #22c55e;
  --drop: #06b6d4;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg-dark);
  color: var(--fg);
  font-family: ui-monospace, 'Source Code Pro', Menlo, Consolas, monospace;
  font-size: 12px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}
header {{
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 10px 18px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}}
header h1 {{
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.top-stats {{
  display: flex;
  gap: 10px;
  align-items: center;
}}
.stat-chip {{
  background: var(--panel-light);
  border: 1px solid var(--border);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  color: var(--fg-dim);
}}
.stat-chip b {{ color: #fff; }}

main {{
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) 380px;
  gap: 1px;
  background: var(--border);
  min-height: 0;
}}
.canvas-wrap {{
  position: relative;
  background: radial-gradient(circle at center, #0e1624 0%, #06090e 100%);
  display: flex;
  flex-direction: column;
  min-height: 0;
}}
#cv-mission {{
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}}
.playback-bar {{
  position: absolute;
  bottom: 14px;
  left: 14px;
  right: 14px;
  background: rgba(13, 18, 26, 0.92);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 10;
}}
.timeline-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.time-slider {{
  flex: 1;
  accent-color: var(--brand);
  cursor: pointer;
}}
.btn-group {{
  display: flex;
  gap: 6px;
  align-items: center;
}}
.btn {{
  background: var(--panel-light);
  border: 1px solid var(--border);
  color: #fff;
  padding: 5px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  font-weight: 600;
  transition: all 0.15s;
}}
.btn:hover {{
  background: var(--brand);
  color: #000;
}}
.btn.active {{
  background: var(--brand);
  color: #000;
}}

.side-deck {{
  background: var(--panel);
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
}}
.deck-sec {{
  padding: 14px;
  border-bottom: 1px solid var(--border);
}}
.deck-sec h2 {{
  font-size: 11px;
  text-transform: uppercase;
  color: var(--fg-dim);
  letter-spacing: 0.08em;
  margin-bottom: 10px;
}}
.card-list {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.surv-card {{
  background: var(--panel-light);
  border: 1px solid var(--border);
  border-left: 3px solid var(--imm);
  border-radius: 5px;
  padding: 9px 12px;
}}
.surv-card.imm {{ border-left-color: var(--imm); }}
.surv-card.del {{ border-left-color: var(--del); }}
.surv-card.min {{ border-left-color: var(--min); }}
.card-top {{
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}}
.card-title {{ font-weight: 700; color: #fff; }}
.card-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3px 6px;
  color: var(--fg-dim);
  font-size: 10.5px;
}}
.card-grid b {{ color: #fff; }}

.legend-box {{
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(13, 18, 26, 0.88);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 10.5px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  z-index: 5;
}}
.leg-row {{ display: flex; align-items: center; gap: 6px; }}
.leg-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
</style>
</head>
<body>

<header>
  <h1>🎬 SAHYOG - Autonomous Rescue Simulation Playback</h1>
  <div class="top-stats">
    <div class="stat-chip">Scenario: <b>{scenario_name}</b></div>
    <div class="stat-chip">Survivors Identified: <b id="stat-survs">0</b></div>
    <div class="stat-chip">Relief Drops: <b id="stat-drops">0</b></div>
    <div class="stat-chip">Mean Impact Error: <b id="stat-err">0.0m</b></div>
  </div>
</header>

<main>
  <div class="canvas-wrap">
    <canvas id="cv-mission"></canvas>
    
    <div class="legend-box">
      <div class="leg-row"><div class="leg-dot" style="background:var(--imm)"></div> Immediate Survivor (Red)</div>
      <div class="leg-row"><div class="leg-dot" style="background:var(--del)"></div> Delayed Survivor (Yellow)</div>
      <div class="leg-row"><div class="leg-dot" style="background:var(--drone)"></div> Search Drone (Spinning Rotors)</div>
      <div class="leg-row"><div class="leg-dot" style="background:var(--drop)"></div> Precision Airdrop &amp; Beacon</div>
      <div class="leg-row"><div class="leg-dot" style="background:var(--route)"></div> A* Safe Evacuation Route</div>
    </div>
    
    <!-- Playback Timeline Controller -->
    <div class="playback-bar">
      <div class="timeline-row">
        <button class="btn" id="btn-play" onclick="togglePlay()">⏸ PAUSE</button>
        <span id="txt-time" style="font-weight:700;color:var(--brand);min-width:65px">0.0 s</span>
        <input type="range" class="time-slider" id="slider-time" min="0" max="100" step="0.1" value="0" oninput="onSeek(this.value)">
        <span id="txt-max-time" style="color:var(--fg-dim)">0.0 s</span>
      </div>
      <div class="timeline-row" style="justify-content:space-between">
        <div class="btn-group">
          <span style="color:var(--fg-dim);margin-right:6px">SPEED:</span>
          <button class="btn" onclick="setSpeed(0.5)">0.5x</button>
          <button class="btn active" id="spd-1" onclick="setSpeed(1.0)">1.0x</button>
          <button class="btn" id="spd-2" onclick="setSpeed(2.0)">2.0x</button>
          <button class="btn" id="spd-5" onclick="setSpeed(5.0)">5.0x</button>
        </div>
        <div style="color:var(--fg-dim);font-size:11px">
          SPACE to Play/Pause &middot; Drag slider to scrub
        </div>
      </div>
    </div>
  </div>

  <div class="side-deck">
    <div class="deck-sec">
      <h2>Human Identification &amp; Relief Operations</h2>
      <div class="card-list" id="deck-survs"></div>
    </div>
    
    <div class="deck-sec" style="flex:1">
      <h2>Airdrop Trajectories &amp; Safe Routes</h2>
      <div class="card-list" id="deck-drops"></div>
    </div>
  </div>
</main>

<script>
const DATA = {json_data_str};
let isPlaying = true;
let curTime = 0;
let maxTime = DATA.duration_s || 60;
let playSpeed = 1.0;
let propAngle = 0;

const cv = document.getElementById('cv-mission');
const cx = cv.getContext('2d');

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = r.width * dpr;
  cv.height = r.height * dpr;
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
window.addEventListener('resize', resize);

function init() {{
  resize();
  document.getElementById('slider-time').max = maxTime;
  document.getElementById('txt-max-time').textContent = maxTime.toFixed(1) + ' s';
  document.getElementById('stat-survs').textContent = (DATA.survivors || []).length;
  document.getElementById('stat-drops').textContent = (DATA.drops || []).length;
  
  const errs = (DATA.drops || []).map(d => d.miss_distance_m || 0);
  const avgErr = errs.length ? (errs.reduce((a,b)=>a+b,0) / errs.length) : 0;
  document.getElementById('stat-err').textContent = avgErr.toFixed(1) + 'm';

  renderCards();
  requestAnimationFrame(animLoop);
}}

function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById('btn-play').textContent = isPlaying ? '⏸ PAUSE' : '▶ PLAY';
}}

function setSpeed(spd) {{
  playSpeed = spd;
  ['1','2','5'].forEach(s => {{
    const el = document.getElementById('spd-' + s);
    if (el) el.classList.toggle('active', s === String(Math.round(spd)));
  }});
}}

function onSeek(val) {{
  curTime = parseFloat(val);
}}

window.addEventListener('keydown', e => {{
  if (e.code === 'Space') {{
    e.preventDefault();
    togglePlay();
  }}
}});

function getTransform() {{
  const w = cv.clientWidth, h = cv.clientHeight;
  const W = DATA.world_width || 1000, H = DATA.world_height || 1000;
  const scale = Math.min(w / W, h / H) * 0.88;
  const ox = (w - W * scale) / 2;
  const oy = (h - H * scale) / 2;
  return {{
    X: e => ox + e * scale,
    Y: n => oy + (H - n) * scale,
    scale, W, H
  }};
}}

function animLoop() {{
  if (isPlaying) {{
    curTime += 0.016 * playSpeed;
    if (curTime > maxTime) curTime = 0;
    document.getElementById('slider-time').value = curTime;
    document.getElementById('txt-time').textContent = curTime.toFixed(1) + ' s';
  }}
  propAngle += 0.45;

  draw();
  requestAnimationFrame(animLoop);
}}

function draw() {{
  const w = cv.clientWidth, h = cv.clientHeight;
  cx.clearRect(0, 0, w, h);
  const {{ X, Y, scale, W, H }} = getTransform();

  // Grid
  cx.strokeStyle = 'rgba(255,255,255,0.04)';
  cx.lineWidth = 1;
  for (let x = 0; x <= W; x += 100) {{
    cx.beginPath(); cx.moveTo(X(x), Y(0)); cx.lineTo(X(x), Y(H)); cx.stroke();
  }}
  for (let y = 0; y <= H; y += 100) {{
    cx.beginPath(); cx.moveTo(X(0), Y(y)); cx.lineTo(X(W), Y(y)); cx.stroke();
  }}

  // Safe Evacuation Routes
  (DATA.routes || []).forEach(rt => {{
    if (!rt.waypoints || rt.waypoints.length < 2) return;
    cx.beginPath();
    cx.moveTo(X(rt.waypoints[0].east_m), Y(rt.waypoints[0].north_m));
    for (let i = 1; i < rt.waypoints.length; i++) {{
      cx.lineTo(X(rt.waypoints[i].east_m), Y(rt.waypoints[i].north_m));
    }}
    cx.strokeStyle = 'rgba(34, 197, 94, 0.7)';
    cx.lineWidth = 2.5;
    cx.setLineDash([4, 4]);
    cx.stroke();
    cx.setLineDash([]);
  }});

  // Hazards
  (DATA.hazards || []).forEach(hz => {{
    cx.beginPath();
    cx.arc(X(hz.east_m), Y(hz.north_m), (hz.radius_m || 20) * scale, 0, Math.PI * 2);
    cx.fillStyle = 'rgba(249, 115, 22, 0.16)';
    cx.fill();
    cx.strokeStyle = '#f97316';
    cx.lineWidth = 1.2;
    cx.stroke();
  }});

  // Survivors
  (DATA.survivors || []).forEach(sv => {{
    const sx = X(sv.east_m), sy = Y(sv.north_m);
    const prio = (sv.priority || 'immediate').toLowerCase();
    const col = prio.includes('imm') ? '#ef4444' : (prio.includes('del') ? '#f59e0b' : '#10b981');

    // Pulsing vital ring
    const pulse = Math.sin(curTime * 4 + sv.north_m) * 3 + 8;
    cx.beginPath();
    cx.arc(sx, sy, pulse, 0, Math.PI * 2);
    cx.fillStyle = col + '33';
    cx.fill();

    // Human marker
    cx.beginPath();
    cx.arc(sx, sy, 5, 0, Math.PI * 2);
    cx.fillStyle = col;
    cx.fill();
    cx.strokeStyle = '#fff';
    cx.lineWidth = 1.5;
    cx.stroke();

    // SOS waving arms
    if (sv.sos_waving || sv.posture === 'waving') {{
      const wave = Math.sin(curTime * 8) * 5;
      cx.strokeStyle = '#fff';
      cx.lineWidth = 1.8;
      cx.beginPath();
      cx.moveTo(sx - 3, sy); cx.lineTo(sx - 8, sy - 6 + wave);
      cx.moveTo(sx + 3, sy); cx.lineTo(sx + 8, sy - 6 - wave);
      cx.stroke();
    }}

    cx.fillStyle = '#fff';
    cx.font = '10px monospace';
    cx.fillText(`${{sv.id_tag||sv.label}} [${{sv.posture||'HUMAN'}}]`, sx + 8, sy + 3);
  }});

  // Airdrop Parachutes & Beacons
  (DATA.drops || []).forEach(dr => {{
    if (curTime < dr.release_time) return;
    const dt = curTime - dr.release_time;
    const dur = dr.flight_time_s || 3.5;
    const frac = Math.min(dt / dur, 1.0);

    const curN = dr.release_pos_ned[0] + (dr.impact_pos_ned[0] - dr.release_pos_ned[0]) * frac;
    const curE = dr.release_pos_ned[1] + (dr.impact_pos_ned[1] - dr.release_pos_ned[1]) * frac;
    const curAlt = dr.release_pos_ned[2] * (1 - frac);

    const dx = X(curE), dy = Y(curN);

    if (frac < 1.0) {{
      // Parachute in flight
      cx.beginPath();
      cx.arc(dx, dy - 10, 8, Math.PI, Math.PI * 2);
      cx.fillStyle = 'rgba(6, 182, 212, 0.85)';
      cx.fill();
      cx.strokeStyle = '#fff';
      cx.stroke();
      
      cx.beginPath();
      cx.moveTo(dx - 7, dy - 10); cx.lineTo(dx, dy);
      cx.moveTo(dx + 7, dy - 10); cx.lineTo(dx, dy);
      cx.strokeStyle = 'rgba(255,255,255,0.7)';
      cx.stroke();

      cx.beginPath();
      cx.arc(dx, dy, 4, 0, Math.PI * 2);
      cx.fillStyle = '#f59e0b';
      cx.fill();
    }} else {{
      // Impact Beacon pulsing
      const p = ((curTime - dr.release_time - dur) * 1.5) % 1.0;
      cx.beginPath();
      cx.arc(dx, dy, p * 25, 0, Math.PI * 2);
      cx.strokeStyle = `rgba(6, 182, 212, ${{1 - p}})`;
      cx.lineWidth = 1.8;
      cx.stroke();

      cx.beginPath();
      cx.arc(dx, dy, 4, 0, Math.PI * 2);
      cx.fillStyle = '#06b6d4';
      cx.fill();
    }}
  }});

  // Interpolated Flight Path Drone
  const traj = DATA.trajectory || [];
  if (traj.length > 0) {{
    const tFrac = Math.min(curTime / maxTime, 0.999);
    const idx = Math.floor(tFrac * (traj.length - 1));
    const p1 = traj[idx], p2 = traj[Math.min(idx + 1, traj.length - 1)];
    const sub = (tFrac * (traj.length - 1)) - idx;
    
    const curN = p1.n + (p2.n - p1.n) * sub;
    const curE = p1.e + (p2.e - p1.e) * sub;
    const yaw = ((p1.yaw || 0) - 90) * Math.PI / 180;
    const vx = X(curE), vy = Y(curN);

    // Flight breadcrumb trail
    cx.beginPath();
    cx.moveTo(X(traj[0].e), Y(traj[0].n));
    for (let i = 1; i <= idx; i++) {{
      cx.lineTo(X(traj[i].e), Y(traj[i].n));
    }}
    cx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
    cx.lineWidth = 2;
    cx.stroke();

    // Sensor Spotlight Cone
    cx.save();
    cx.translate(vx, vy);
    cx.rotate(yaw);
    const coneGrad = cx.createRadialGradient(0, 0, 2, 0, 0, 45);
    coneGrad.addColorStop(0, 'rgba(56, 189, 248, 0.5)');
    coneGrad.addColorStop(0.7, 'rgba(56, 189, 248, 0.15)');
    coneGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
    cx.fillStyle = coneGrad;
    cx.beginPath();
    cx.arc(0, 0, 45, 0, Math.PI * 2);
    cx.fill();

    // Drone Body
    cx.strokeStyle = '#fff';
    cx.lineWidth = 2;
    cx.beginPath();
    cx.moveTo(-8, -8); cx.lineTo(8, 8);
    cx.moveTo(8, -8); cx.lineTo(-8, 8);
    cx.stroke();

    cx.beginPath();
    cx.arc(0, 0, 5, 0, Math.PI * 2);
    cx.fillStyle = '#a855f7';
    cx.fill();
    cx.strokeStyle = '#fff';
    cx.stroke();

    // Propellers
    [[-8,-8], [8,-8], [-8,8], [8,8]].forEach(([px, py]) => {{
      cx.save();
      cx.translate(px, py);
      cx.rotate(propAngle);
      cx.strokeStyle = '#38bdf8';
      cx.lineWidth = 1.5;
      cx.beginPath(); cx.moveTo(-5, 0); cx.lineTo(5, 0); cx.stroke();
      cx.restore();
    }});

    cx.restore();
  }}
}}

function renderCards() {{
  const sc = document.getElementById('deck-survs');
  sc.innerHTML = (DATA.survivors || []).map(sv => {{
    const p = (sv.priority || 'immediate').toLowerCase();
    const pClass = p.includes('imm') ? 'imm' : (p.includes('del') ? 'del' : 'min');
    return `
      <div class="surv-card ${{pClass}}">
        <div class="card-top">
          <span class="card-title">${{sv.id_tag || sv.label}}</span>
          <span style="color:var(--brand);font-weight:700">${{sv.priority.toUpperCase()}}</span>
        </div>
        <div class="card-grid">
          <span>Posture: <b>${{sv.posture}}</b></span>
          <span>Core Temp: <b>${{sv.estimated_core_temp_c || 36.8}}°C</b></span>
          <span>Distress: <b>${{(sv.distress_level||'').replace('_',' ')}}</b></span>
          <span>Payload: <b>${{(sv.recommended_equipment||'').replace('_',' ')}}</b></span>
        </div>
      </div>
    `;
  }}).join('');

  const dc = document.getElementById('deck-drops');
  dc.innerHTML = (DATA.drops || []).map(dr => `
    <div class="surv-card del">
      <div class="card-top">
        <span class="card-title">${{dr.payload_type.toUpperCase()}}</span>
        <span style="color:${{dr.success ? '#10b981' : '#ef4444'}};font-weight:700">
          ${{dr.success ? 'DELIVERED (SUCCESS)' : 'MISSED'}}
        </span>
      </div>
      <div class="card-grid">
        <span>Impact Error: <b>${{dr.miss_distance_m.toFixed(1)}} m</b></span>
        <span>Drop Alt: <b>${{Math.abs(dr.release_pos_ned[2]).toFixed(0)}} m</b></span>
        <span>Descent Time: <b>${{dr.flight_time_s.toFixed(1)}} s</b></span>
        <span>Impact Speed: <b>${{dr.impact_velocity_ms.toFixed(1)}} m/s</b></span>
      </div>
    </div>
  `).join('');
}}

init();
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    log.info("Saved interactive HTML5 animation to %s", output_path)
    return output_path


# --------------------------------------------------------------------------- #
# Matplotlib GIF Animation Generator
# --------------------------------------------------------------------------- #
def generate_matplotlib_gif(
    scenario_name: str,
    mission_data: Dict[str, Any],
    output_path: Path,
    fps: int = 15,
) -> Optional[Path]:
    """Render and save an animated GIF using Matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("Matplotlib not available for GIF export.")
        return None

    log.info("Rendering Matplotlib mission animation GIF...")
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#06090e")
    ax.set_facecolor("#0a0f18")

    W = mission_data.get("world_width", 1000)
    H = mission_data.get("world_height", 1000)
    ax.set_xlim(-20, W + 20)
    ax.set_ylim(-20, H + 20)
    ax.set_aspect("equal")
    ax.set_title(f"SAHYOG - SAR Autonomous Mission Animation ({scenario_name})", color="#f8fafc", fontsize=12, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    # Static elements: Hazards
    for hz in mission_data.get("hazards", []):
        circle = plt.Circle((hz["east_m"], hz["north_m"]), hz.get("radius_m", 20),
                            color="#f97316", alpha=0.25, ec="#f97316", lw=1.5)
        ax.add_patch(circle)

    # Static elements: Safe routes
    for rt in mission_data.get("routes", []):
        wps = rt.get("waypoints", [])
        if len(wps) >= 2:
            es = [w["east_m"] for w in wps]
            ns = [w["north_m"] for w in wps]
            ax.plot(es, ns, "--", color="#22c55e", alpha=0.65, lw=2, label="Safe Evac Route")

    # Static elements: Survivors
    for sv in mission_data.get("survivors", []):
        col = "#ef4444" if "imm" in sv.get("priority", "immediate").lower() else "#f59e0b"
        ax.plot(sv["east_m"], sv["north_m"], "o", color=col, ms=7, mec="#ffffff", mew=1.2)
        ax.text(sv["east_m"] + 10, sv["north_m"] + 3, f"{sv['id_tag']} ({sv['posture']})",
                color="#f8fafc", fontsize=8, fontfamily="monospace")

    # Dynamic elements: Drone & Drops
    drone_dot, = ax.plot([], [], "^", color="#a855f7", ms=10, mec="#ffffff", label="Drone")
    drone_cone, = ax.plot([], [], "o", color="#38bdf8", alpha=0.2, ms=28)
    trail_line, = ax.plot([], [], "-", color="#38bdf8", alpha=0.7, lw=1.8, label="Search Path")
    drop_dots, = ax.plot([], [], "s", color="#06b6d4", ms=8, mec="#ffffff", label="Relief Airdrops")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="#38bdf8",
                        fontsize=10, fontfamily="monospace", weight="bold")

    traj = mission_data.get("trajectory", [])
    n_frames = min(120, max(30, len(traj)))
    duration = mission_data.get("duration_s", 60.0)

    def update(frame_idx: int):
        t = (frame_idx / n_frames) * duration
        frac = frame_idx / max(1, n_frames - 1)
        idx = int(frac * (len(traj) - 1))

        if traj:
            cur_p = traj[idx]
            drone_dot.set_data([cur_p["e"]], [cur_p["n"]])
            drone_cone.set_data([cur_p["e"]], [cur_p["n"]])
            es = [p["e"] for p in traj[:idx+1]]
            ns = [p["n"] for p in traj[:idx+1]]
            trail_line.set_data(es, ns)

        # Active Drops
        active_drops = [d for d in mission_data.get("drops", []) if t >= d["release_time"]]
        if active_drops:
            des = [d["impact_pos_ned"][1] for d in active_drops]
            dns = [d["impact_pos_ned"][0] for d in active_drops]
            drop_dots.set_data(des, dns)

        time_text.set_text(f"MISSION TIME: {t:.1f} s | SPEED: {traj[idx].get('gs', 8.0):.1f} m/s")
        return drone_dot, drone_cone, trail_line, drop_dots, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(str(output_path), writer="pillow", fps=fps)
        log.info("Saved animated GIF to %s", output_path)
        return output_path
    except Exception as exc:
        log.warning("Could not export GIF via pillow: %r", exc)
        return None
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthetic Mission Builder
# --------------------------------------------------------------------------- #
def build_mission_animation_data(scenario_name: str = "flood") -> Dict[str, Any]:
    """Generate comprehensive trajectory, perception, identification, and rescue data."""
    world, lwir_spec, rgb_spec = build_reference_scenario(scenario_name)
    human_id_engine = HumanIdentifier()
    drop_calc = BallisticDropCalculator()
    router = GroundRescueRouter()

    # Synthetic Survey Flight Trajectory (lawnmower pattern)
    duration_s = 60.0
    dt = 0.5
    times = np.arange(0, duration_s, dt)
    traj = []

    # Generate survey lanes
    n_lanes = 6
    lane_length = min(350.0, world.north_m)
    lane_width = min(350.0, world.east_m)
    lane_spacing = lane_width / n_lanes

    total_pts = len(times)
    pts_per_lane = total_pts // n_lanes

    for i, t in enumerate(times):
        l_idx = min(i // max(1, pts_per_lane), n_lanes - 1)
        sub_frac = (i % max(1, pts_per_lane)) / max(1, pts_per_lane)
        east = l_idx * lane_spacing + 20.0
        if l_idx % 2 == 0:
            north = sub_frac * lane_length + 20.0
            yaw = 0.0
        else:
            north = (1.0 - sub_frac) * lane_length + 20.0
            yaw = 180.0

        traj.append({
            "t": round(float(t), 2),
            "n": round(float(north), 1),
            "e": round(float(east), 1),
            "alt": 45.0,
            "yaw": yaw,
            "gs": 8.0,
        })

    # Human Identification Profiles & Triage
    survivors = []
    airdrops = []
    routes = []

    for i, v in enumerate(getattr(world, "victims", [])):
        class MockObs:
            def __init__(self, t_val):
                self.t = t_val
                self.gsd_m = 0.12
                self.confidence = 0.88
                self.peak_temp_c = 36.8
                self.background_temp_c = 22.0
                self.modality = "rgb"
                self.area_px = 45.0
                self.north = float(v.north)
                self.east = float(v.east)
                self.attributes = {"aspect": 1.2, "elongation": 1.2, "waving_motion": 0.85 if v.category == "hypothermia_risk" else 0.1}

        class MockTrack:
            def __init__(self, victim, idx):
                self.tid = idx + 1
                self.label = "person"
                self.confidence = 0.89
                self.north = float(victim.north)
                self.east = float(victim.east)
                self.position_sigma_m = 1.4
                self.observations = [MockObs(0.0), MockObs(1.0), MockObs(2.0)]
                self.n_obs = 3
                self.immobility = 0.9 if victim.on_rooftop else 0.2
                self.speed_ms = 0.05
                self.age_s = 30.0

        tr = MockTrack(v, i)
        prof = human_id_engine.identify(
            track=tr,
            water_depth_m=0.8 if v.in_water else 0.0,
            ambient_temp_c=22.0,
            wind_speed_ms=3.0,
        )
        p_dict = prof.to_dict()
        survivors.append(p_dict)

        # Ballistic Airdrop solution
        spec = PAYLOAD_SPECS[RescuePayloadType.FLOTATION_BUOY if v.in_water else RescuePayloadType.FIRST_AID_TRAUMA_KIT]
        rel_pos = np.array([v.north - 15.0, v.east - 8.0, -45.0])
        rel_vel = np.array([6.0, 3.0, 0.0])

        drop_res = drop_calc.simulate_drop(
            spec=spec,
            release_pos_ned=rel_pos,
            release_vel_ned=rel_vel,
            release_time=12.0 + i * 4.0,
        )
        drop_res.target_id = i + 1
        drop_res.miss_distance_m = float(math.hypot(drop_res.impact_pos_ned[0] - v.north, drop_res.impact_pos_ned[1] - v.east))
        drop_res.success = drop_res.miss_distance_m < 20.0
        airdrops.append(drop_res.to_dict())

        # Safe Ground / Boat Route
        rt = router.plan_route(
            start_ned=(0.0, 0.0, 0.0),
            target_ned=(v.north, v.east, 0.0),
            team_type=RescueTeamType.AMPHIBIOUS_RESCUE_BOAT if v.in_water else RescueTeamType.FOOT_RESCUE_TEAM,
            world=world,
            target_id=i + 1,
        )
        routes.append(rt.to_dict())

    # Hazards
    hazards = []
    for f in getattr(world, "fires", []):
        hazards.append({
            "hazard_class": "fire",
            "north_m": float(f.north),
            "east_m": float(f.east),
            "radius_m": float(f.radius_m),
            "severity": 0.9,
        })

    return {
        "scenario": scenario_name,
        "world_width": float(world.east_m),
        "world_height": float(world.north_m),
        "duration_s": duration_s,
        "trajectory": traj,
        "survivors": survivors,
        "drops": airdrops,
        "routes": routes,
        "hazards": hazards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAR Mission Animation")
    parser.add_argument("--scenario", default="flood", help="Scenario name (flood, earthquake, wildfire, landslide)")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    parser.add_argument("--gif", action="store_true", help="Also generate animated GIF")
    args = parser.parse_args()

    art_dir = Path(args.artifacts)
    art_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating SAR rescue mission animation data for '%s'...", args.scenario)
    mission_data = build_mission_animation_data(args.scenario)

    html_path = art_dir / "rescue_mission_animation.html"
    generate_html_animation(args.scenario, mission_data, html_path)

    if args.gif:
        gif_path = art_dir / "rescue_mission.gif"
        generate_matplotlib_gif(args.scenario, mission_data, gif_path)

    log.info("Mission animation generation completed successfully!")


if __name__ == "__main__":
    main()
