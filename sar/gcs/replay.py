"""Mission replay dashboard.

The live command centre (:mod:`sar.gcs.dashboard`) draws from an in-flight
:class:`~sar.comms.TelemetryUplink`.  Every sortie also writes a complete report
artifact under ``artifacts/``, and this module serves a read-only, offline
replay of one: a map, a timeline scrubber, and the physics-informed survivor
profiles (posture, triage, core temperature, payload drop, ground route).

It is deliberately self-contained — no runner, no uplink, no world — so a
mission can be reviewed on a laptop with no connectivity, exactly like the
field conditions the system is designed for.

Run it directly::

    python scripts/serve_replay.py --artifact artifacts/mission_flood.json

or::

    python -m sar.gcs.replay --artifact artifacts/rescue_mission_flood.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, wgs84_to_local

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    _HAS_FASTAPI = True
except Exception:                                # pragma: no cover
    _HAS_FASTAPI = False

log = logging.getLogger("sar.gcs.replay")

__all__ = ["ReplayDashboard", "load_artifact", "project_survivors"]

_PRIORITY_COLORS = {
    "IMMEDIATE": "#dc2626", "DELAYED": "#f59e0b", "MINOR": "#facc15",
    "EXPECTANT": "#94a3b8", "UNKNOWN": "#3b82f6", "D": "#dc2626",
    "M": "#f59e0b", "m": "#f59e0b", "i": "#dc2626",
}

_HAZARD_COLORS = {
    "CRITICAL": "#7f1d1d", "DANGEROUS": "#dc2626", "CAUTION": "#ea580c",
    "INFORMATIONAL": "#3b82f6", "NONE": "#94a3b8",
}


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
def _as_origin(origin: Optional[Dict[str, Any]]) -> Optional[GeoPoint]:
    if not origin or origin.get("lat") is None:
        return None
    return GeoPoint(float(origin["lat"]), float(origin["lon"]),
                    float(origin.get("alt", 0.0)))


def _project(lat: Any, lon: Any, origin: Optional[GeoPoint]) -> Optional[Tuple[float, float]]:
    if origin is None or lat is None or lon is None:
        return None
    try:
        n, e, _ = wgs84_to_local(GeoPoint(float(lat), float(lon), 0.0), origin)
        return float(n), float(e)
    except Exception:
        return None


def project_survivors(data: Dict[str, Any],
                      origin: Optional[GeoPoint]) -> List[Dict[str, Any]]:
    """Build a time-indexed, locally-projected survivor list from an artifact."""
    out: List[Dict[str, Any]] = []
    for s in data.get("survivors_reported", []) or []:
        ne = _project(s.get("la"), s.get("lo"), origin)
        out.append({
            "id": s.get("_key") or s.get("id"),
            "north_m": round(ne[0], 1) if ne else None,
            "east_m": round(ne[1], 1) if ne else None,
            "lat": s.get("la"), "lon": s.get("lo"),
            "sigma_m": s.get("sg", s.get("sigma_m")),
            "priority": s.get("pr", s.get("priority")),
            "group_size": s.get("gs", s.get("group_size", 1)),
            "confidence": s.get("cf", s.get("confidence")),
            "time_critical_s": s.get("tc", s.get("time_critical_s")),
            "needs": s.get("nb", s.get("needs")),
            "cross_modal": bool(s.get("xm")),
            "t": float(s.get("tm") or s.get("last_seen_t") or 0.0),
            "n_updates": s.get("_n_updates", 1),
            "source": "ground",
        })
    return out


def _operations(data: Dict[str, Any],
                origin: Optional[GeoPoint]) -> List[Dict[str, Any]]:
    """The rich rescue-operation view: one entry per survivor with profile,
    payload drop and ground routes, if the artifact has them."""
    out: List[Dict[str, Any]] = []
    ops = (data.get("rescue") or {}).get("active_operations", []) or []
    for op in ops:
        prof = op.get("profile") or {}
        drop = op.get("drop_result")
        routes = op.get("ground_routes") or {}
        out.append({
            "id": prof.get("id_tag") or f"S{op.get('target_id')}",
            "target_id": op.get("target_id"),
            "north_m": prof.get("north_m"),
            "east_m": prof.get("east_m"),
            "lat": prof.get("lat"), "lon": prof.get("lon"),
            "sigma_m": None,
            "priority": (prof.get("triage_priority") or "").upper(),
            "group_size": prof.get("group_size", 1),
            "confidence": prof.get("confidence"),
            "time_critical_s": prof.get("time_critical_s"),
            "needs": prof.get("recommended_equipment"),
            "cross_modal": None,
            "posture": prof.get("posture"),
            "posture_confidence": prof.get("posture_confidence"),
            "distress": prof.get("distress_level"),
            "sos": prof.get("sos_waving_detected"),
            "demographic": prof.get("demographic"),
            "core_temp_c": prof.get("estimated_core_temp_c"),
            "hypothermia_risk": prof.get("hypothermia_risk"),
            "t": float(op.get("created_at") or 0.0),
            "n_updates": 1,
            "source": "operation",
            "state": op.get("state"),
            "payload": op.get("assigned_payload"),
            "drop": drop,
            "routes": [
                {
                    "team_type": team,
                    "safety_score": r.get("safety_score"),
                    "distance_m": r.get("total_distance_m"),
                    "waypoints": r.get("waypoints") or [],
                }
                for team, r in routes.items()
            ],
        })
    # Sort by first appearance so the scrubber reveals them in order.
    out.sort(key=lambda e: float(e["t"]))
    return out


def _hazards(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for h in data.get("hazards_reported", []) or []:
        out.append({
            "hid": h.get("hid"), "label": h.get("label"),
            "north_m": h.get("north_m"), "east_m": h.get("east_m"),
            "severity": (h.get("severity") or "CAUTION").upper(),
            "exclusion_m": h.get("exclusion_radius_m", h.get("exclusion_m")),
            "confidence": h.get("confidence"),
            "t": float(h.get("first_seen_t") or h.get("last_seen_t") or 0.0),
        })
    return out


def _bounds(entities: Sequence[Tuple[float, float]],
            default: float = 1000.0) -> Dict[str, float]:
    """A padded bounding box that contains every entity (plus the origin)."""
    pts = [(float(n), float(e)) for n, e in entities if n is not None and e is not None]
    pts.append((0.0, 0.0))          # the GCS / mission origin
    if not pts:
        return {"n_min": -default / 2, "n_max": default / 2,
                "e_min": -default / 2, "e_max": default / 2}
    ns = [p[0] for p in pts]
    es = [p[1] for p in pts]
    span = max(max(ns) - min(ns), max(es) - min(es), 40.0)
    pad = span * 0.08 + 10.0
    return {"n_min": min(ns) - pad, "n_max": max(ns) + pad,
            "e_min": min(es) - pad, "e_max": max(es) + pad}


def load_artifact(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    origin = _as_origin(data.get("origin"))

    ops = _operations(data, origin)
    survivors = ops if ops else project_survivors(data, origin)
    hazards = _hazards(data)
    truth = [v for v in (data.get("truth") or {}).get("victims", [])]
    lanes = (data.get("plan") or {}).get("lanes", []) or []

    entities: List[Tuple[float, float]] = []
    for s in survivors:
        entities.append((s.get("north_m"), s.get("east_m")))
    for h in hazards:
        entities.append((h.get("north_m"), h.get("east_m")))
    for v in truth:
        entities.append((v.get("north_m"), v.get("east_m")))
    for ln in lanes:
        st, en = ln.get("start"), ln.get("end")
        if st:
            entities.append((st[0], st[1]))
        if en:
            entities.append((en[0], en[1]))
    for op in ops:
        drop = op.get("drop") or {}
        for key in ("impact_north_m", "impact_east_m", "release_north_m", "release_east_m"):
            pass
        if drop.get("impact_north_m") is not None:
            entities.append((drop["impact_north_m"], drop["impact_east_m"]))

    return {
        "scenario": data.get("scenario", ""),
        "duration_s": float(data.get("duration_s") or data.get("flight_time_s") or 0.0),
        "flight_time_s": float(data.get("flight_time_s") or 0.0),
        "bounds": _bounds(entities),
        "origin": data.get("origin"),
        "survivors": survivors,
        "hazards": hazards,
        "truth": [
            {"vid": v.get("vid"), "north_m": v.get("north_m"),
             "east_m": v.get("east_m"), "alive": v.get("alive"),
             "category": v.get("category")}
            for v in truth
        ],
        "lanes": lanes,
        "search_box": ((data.get("plan") or {}).get("search_box")
                       or (data.get("coverage") or {}).get("search_box")),
        "coverage": data.get("coverage") or {},
        "scoring": data.get("scoring") or {},
        "comms": data.get("comms") or {},
        "rescue": data.get("rescue") or {},
        "perception": data.get("perception") or {},
        "timeline": data.get("timeline", []) or [],
        "faults": data.get("faults", []) or [],
    }


# --------------------------------------------------------------------------- #
# The web page
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SAHYOG SAR — Mission Replay</title>
<style>
  :root{--bg:#0b1220;--panel:#111a2b;--panel2:#16213a;--ink:#e5edf8;
        --muted:#8ea3c0;--accent:#22c55e;--route:#38bdf8;--line:#243350;
        --red:#dc2626;--amber:#f59e0b;--yellow:#facc15}
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column}
  header{display:flex;align-items:center;gap:14px;padding:10px 16px;
         background:linear-gradient(180deg,#12203a,#0e1728);border-bottom:1px solid var(--line)}
  header .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent)}
  header h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:12px}
  header .spacer{flex:1}
  .badge{background:var(--panel2);border:1px solid var(--line);color:var(--muted);
         padding:3px 10px;border-radius:999px;font-size:12px}
  main{flex:1;display:flex;min-height:0}
  .mapwrap{flex:1;position:relative;background:#0a1120}
  canvas{display:block;width:100%;height:100%}
  .side{width:340px;background:var(--panel);border-left:1px solid var(--line);
        display:flex;flex-direction:column;min-height:0}
  .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
  .stat .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  .stat .v{font-size:19px;font-weight:650;margin-top:1px}
  .stat .v small{font-size:11px;color:var(--muted);font-weight:400}
  .controls{padding:10px 12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
  .row{display:flex;align-items:center;gap:8px}
  button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
         border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px}
  button:hover{border-color:var(--accent)}
  button.active{background:var(--accent);color:#04130a;border-color:var(--accent);font-weight:600}
  input[type=range]{flex:1;accent-color:var(--accent)}
  .clock{font-variant-numeric:tabular-nums;color:var(--accent);font-weight:600;min-width:72px;text-align:right}
  .panelhead{padding:8px 12px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px;
             border-bottom:1px solid var(--line)}
  .details{flex:1;overflow:auto;padding:10px 12px}
  .details .empty{color:var(--muted);font-size:13px}
  .card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px}
  .card h4{margin:0 0 6px;font-size:13px}
  .kv{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:2px 0;border-bottom:1px dashed #1d2b44}
  .kv:last-child{border-bottom:none}
  .kv .k{color:var(--muted)}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600}
  .log{font-size:12px;color:var(--muted)}
  .log .t{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:6px}
  .foot{padding:8px 12px;color:var(--muted);font-size:11px;border-top:1px solid var(--line)}
  .legend{position:absolute;left:12px;bottom:12px;background:rgba(17,26,43,.9);
          border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:11px;color:var(--muted)}
  .legend .l{display:flex;align-items:center;gap:6px;margin:3px 0}
  .legend .sw{width:10px;height:10px;border-radius:3px;display:inline-block}
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <div>
    <h1>SAHYOG SAR — Mission Replay</h1>
    <div class="sub" id="meta">loading…</div>
  </div>
  <div class="spacer"></div>
  <span class="badge">offline · artifact replay</span>
</header>
<main>
  <div class="mapwrap">
    <canvas id="map"></canvas>
    <div class="legend">
      <div class="l"><span class="sw" style="background:#dc2626"></span> Immediate</div>
      <div class="l"><span class="sw" style="background:#f59e0b"></span> Delayed</div>
      <div class="l"><span class="sw" style="background:#facc15"></span> Minor</div>
      <div class="l"><span class="sw" style="background:#38bdf8"></span> Safe route</div>
      <div class="l"><span class="sw" style="background:#64748b"></span> Ground truth</div>
    </div>
  </div>
  <div class="side">
    <div class="stats">
      <div class="stat"><div class="k">Scenario</div><div class="v" id="st-scenario">–</div></div>
      <div class="stat"><div class="k">Flight time</div><div class="v" id="st-flight">–</div></div>
      <div class="stat"><div class="k">Survivors</div><div class="v" id="st-surv">–</div></div>
      <div class="stat"><div class="k">Hazards</div><div class="v" id="st-haz">–</div></div>
      <div class="stat"><div class="k">Payloads dropped</div><div class="v" id="st-drop">–</div></div>
      <div class="stat"><div class="k">Recall / Precision</div><div class="v" id="st-score">–</div></div>
    </div>
    <div class="controls">
      <div class="row">
        <button id="btn-play" class="active">▶ Play</button>
        <button id="btn-restart">↺</button>
        <select id="speed">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
          <option value="8">8×</option>
        </select>
        <span class="clock" id="clock">00:00</span>
      </div>
      <div class="row" style="margin-top:8px">
        <input id="scrub" type="range" min="0" max="100" value="0"/>
      </div>
    </div>
    <div class="panelhead">Survivor detail &nbsp;·&nbsp; event log</div>
    <div class="details" id="details"><div class="empty">scrub the timeline to reveal detections…</div></div>
    <div class="foot">Read-only replay of a recorded sortie. Live command centre: <code>run_mission.py --live</code>.</div>
  </div>
</main>
<script>
const EV = {bounds:{n_min:0,n_max:1000,e_min:0,e_max:1000},duration_s:0,survivors:[],hazards:[],
            truth:[],lanes:[],search_box:null,coverage:{},scoring:{},comms:{},rescue:{},
            timeline:[],faults:[],scenario:"",flight_time_s:0};
let T = 0, playing = true, speed = 1, lastTick = performance.now();
const cv = document.getElementById("map"), ctx = cv.getContext("2d");
let W = 0, H = 0;

function fit(){const r=cv.getBoundingClientRect();W=cv.width=r.width*devicePixelRatio;H=cv.height=r.height*devicePixelRatio;}
window.addEventListener("resize",fit);

function X(e){const b=EV.bounds;return ((e-b.e_min)/(b.e_max-b.e_min))*W;}
function Y(n){const b=EV.bounds;return H-((n-b.n_min)/(b.n_max-b.n_min))*H;}

function prioColor(p){return ({IMMEDIATE:"#dc2626",DELAYED:"#f59e0b",MINOR:"#facc15",
  EXPECTANT:"#94a3b8",UNKNOWN:"#3b82f6",D:"#dc2626",M:"#f59e0b",I:"#dc2626"})[(p||"").toUpperCase()]||"#3b82f6";}
function sevColor(s){return ({CRITICAL:"#7f1d1d",DANGEROUS:"#dc2626",CAUTION:"#ea580c",
  INFORMATIONAL:"#3b82f6",NONE:"#94a3b8"})[(s||"").toUpperCase()]||"#94a3b8";}

function draw(){
  ctx.clearRect(0,0,W,H);
  // grid
  const b=EV.bounds, gs=50; ctx.strokeStyle="rgba(56,89,140,.12)"; ctx.lineWidth=1;
  for(let e=Math.floor(b.e_min/gs)*gs;e<=b.e_max;e+=gs){ctx.beginPath();ctx.moveTo(X(e),0);ctx.lineTo(X(e),H);ctx.stroke();}
  for(let n=Math.floor(b.n_min/gs)*gs;n<=b.n_max;n+=gs){ctx.beginPath();ctx.moveTo(0,Y(n));ctx.lineTo(W,Y(n));ctx.stroke();}

  // search box
  if(EV.search_box && EV.search_box.north_m){
    const sb=EV.search_box; ctx.strokeStyle="rgba(34,197,94,.55)";ctx.setLineDash([6,5]);ctx.lineWidth=1.4;
    ctx.strokeRect(X(sb.east_m[0]),Y(sb.north_m[1]),X(sb.east_m[1])-X(sb.east_m[0]),Y(sb.north_m[0])-Y(sb.north_m[1]));
    ctx.setLineDash([]);
  }
  // lanes
  ctx.strokeStyle="rgba(56,189,248,.28)";ctx.lineWidth=1;
  for(const ln of EV.lanes){if(!ln.start||!ln.end)continue;
    ctx.beginPath();ctx.moveTo(X(ln.start[1]),Y(ln.start[0]));ctx.lineTo(X(ln.end[1]),Y(ln.end[0]));ctx.stroke();}

  // ground truth (ghost)
  for(const v of EV.truth){if(v.north_m==null)continue;
    ctx.fillStyle="rgba(148,163,184,.4)";ctx.beginPath();ctx.arc(X(v.east_m),Y(v.north_m),3.5,0,7);ctx.fill();}

  // hazards (up to now)
  for(const h of EV.hazards){if(h.t>T||h.north_m==null)continue;
    const r=h.exclusion_m?Math.max(6,h.exclusion_m*1.0):8;
    ctx.strokeStyle=sevColor(h.severity);ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(X(h.east_m),Y(h.north_m),Math.min(r,40),0,7);ctx.stroke();
    ctx.fillStyle=sevColor(h.severity)+"33";ctx.beginPath();ctx.arc(X(h.east_m),Y(h.north_m),Math.min(r,40),0,7);ctx.fill();
    ctx.fillStyle=sevColor(h.severity);ctx.font="10px system-ui";ctx.fillText((h.label||"HAZARD").toUpperCase(),X(h.east_m)+r+2,Y(h.north_m)+3);
  }

  // routes + drops
  for(const s of EV.survivors){if(s.t>T||s.north_m==null)continue;
    for(const rt of (s.routes||[])){
      const wps=rt.waypoints||[];if(wps.length<2)continue;
      ctx.strokeStyle="rgba(56,189,248,.8)";ctx.lineWidth=1.6;
      ctx.beginPath();
      for(let i=0;i<wps.length;i++){const w=wps[i];const nx=w.east_m??w.east_m;
        if(i===0)ctx.moveTo(X(w.east_m),Y(w.north_m));else ctx.lineTo(X(w.east_m),Y(w.north_m));}
      ctx.stroke();
    }
    if(s.drop && s.drop.impact_north_m!=null){
      ctx.strokeStyle="#f43f5e";ctx.lineWidth=1.5;
      const ix=X(s.drop.impact_east_m), iy=Y(s.drop.impact_north_m);
      ctx.beginPath();ctx.moveTo(ix-5,iy-5);ctx.lineTo(ix+5,iy+5);ctx.moveTo(ix+5,iy-5);ctx.lineTo(ix-5,iy+5);ctx.stroke();
    }
  }

  // survivors (up to now), urgent first so they stay on top
  const vis=[...EV.survivors].filter(s=>s.t<=T&&s.north_m!=null)
      .sort((a,b)=>(b.priority==="IMMEDIATE"||b.priority==="D")-((a.priority==="IMMEDIATE"||a.priority==="D")));
  for(const s of vis){
    const x=X(s.east_m), y=Y(s.north_m), col=prioColor(s.priority);
    if(s.sigma_m){ctx.strokeStyle=col+"66";ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,Math.min(s.sigma_m,60),0,7);ctx.stroke();}
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill();
    ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle=col;ctx.font="11px system-ui";
    ctx.fillText((s.id||("S"+s.target_id))+" "+(s.posture?s.posture.replace(/_/g," "):""),x+8,y-4);
  }

  // GCS origin
  ctx.fillStyle="#22c55e";ctx.beginPath();ctx.arc(X(0),Y(0),4,0,7);ctx.fill();
  ctx.fillStyle="#22c55e";ctx.font="10px system-ui";ctx.fillText("GCS",X(0)+6,Y(0)+3);
}

function renderDetails(){
  const d=document.getElementById("details");
  const vis=EV.survivors.filter(s=>s.t<=T).sort((a,b)=>b.t-a.t);
  const evs=EV.timeline.filter(e=>(e.t??0)<=T).slice(-24);
  let html="";
  if(vis.length){
    for(const s of vis.slice(0,4)){
      const col=prioColor(s.priority);
      html+=`<div class="card"><h4><span class="pill" style="background:${col}22;color:${col}">${s.priority||"?"}</span> ${s.id||"S"+s.target_id}</h4>`;
      const kv=[["Posture",s.posture?s.posture.replace(/_/g," "):"—"],
                ["Triage",s.priority||"—"],["Core temp",s.core_temp_c?s.core_temp_c+" °C":"—"],
                ["Distress",s.distress||"—"],["SOS waving",s.sos?"yes":(s.sos===false?"no":"—")],
                ["Group",s.group_size??"—"],["Equipment",s.needs||"—"],
                ["Confidence",s.confidence!=null?Number(s.confidence).toFixed(2):"—"]];
      if(s.drop)kv.push(["Drop miss",s.drop.miss_distance_m!=null?s.drop.miss_distance_m+" m":"—"]);
      if(s.routes&&s.routes.length)kv.push(["Safe route",s.routes[0].distance_m?Number(s.routes[0].distance_m).toFixed(0)+" m · "+s.routes[0].team_type:"—"]);
      for(const [k,v] of kv)html+=`<div class="kv"><span class="k">${k}</span><span>${v}</span></div>`;
      html+=`</div>`;
    }
  } else html+=`<div class="empty">no survivors reported yet at t=${T.toFixed(0)}s</div>`;
  html+=`<div class="panelhead" style="margin:4px 0 6px">events (last 24)</div><div class="log">`;
  for(const e of evs)html+=`<div><span class="t">${Number(e.t).toFixed(0)}s</span>${e.event}${e.hazard_class?" · "+e.hazard_class:""}${e.track_id?" · track "+e.track_id:""}</div>`;
  html+=`</div>`;
  d.innerHTML=html;
}

function tick(){
  const now=performance.now(), dt=(now-lastTick)/1000; lastTick=now;
  if(playing){T=Math.min(T+dt*speed, EV.duration_s); if(T>=EV.duration_s)playing=false;}
  document.getElementById("clock").textContent=(T/60).toFixed(1).padStart(4,"0")+"m";
  document.getElementById("scrub").value=T/Math.max(EV.duration_s,1)*100;
  document.getElementById("btn-play").textContent=playing?"⏸ Pause":"▶ Play";
  fit(); draw(); renderDetails();
  requestAnimationFrame(tick);
}

fetch("/api/events").then(r=>r.json()).then(d=>{
  Object.assign(EV,d);
  document.getElementById("meta").textContent=`${d.scenario||"mission"} · ${d.duration_s.toFixed(0)}s sortie · ${new Date().toLocaleString()}`;
  document.getElementById("st-scenario").textContent=d.scenario||"—";
  document.getElementById("st-flight").innerHTML=(d.flight_time_s||d.duration_s||0).toFixed(0)+"<small> s</small>";
  document.getElementById("st-surv").textContent=d.survivors.length;
  document.getElementById("st-haz").textContent=d.hazards.length;
  const drops=d.survivors.filter(s=>s.drop).length;
  document.getElementById("st-drop").textContent=drops;
  const sc=d.scoring||{};
  document.getElementById("st-score").textContent=(sc.recall!=null?Math.round(sc.recall*100)+"%":"—")+" / "+(sc.precision!=null?Math.round(sc.precision*100)+"%":"—");
  fit(); draw(); renderDetails();
  requestAnimationFrame(tick);
});
document.getElementById("btn-play").onclick=()=>{playing=!playing;lastTick=performance.now();};
document.getElementById("btn-restart").onclick=()=>{T=0;playing=true;lastTick=performance.now();};
document.getElementById("speed").onchange=e=>{speed=parseFloat(e.target.value);};
document.getElementById("scrub").oninput=e=>{T=parseFloat(e.target.value)/100*EV.duration_s;lastTick=performance.now();draw();renderDetails();};
</script>
</body>
</html>
"""


class ReplayDashboard:
    """Serves the replay page plus the one-shot ``/api/events`` payload."""

    def __init__(self, artifact: str | Path, host: str = "0.0.0.0",
                 port: int = 8090) -> None:
        if not _HAS_FASTAPI:                       # pragma: no cover
            raise RuntimeError("the replay dashboard needs fastapi and uvicorn")
        self.artifact = Path(artifact)
        self.host = host
        self.port = int(port)
        self.events = load_artifact(self.artifact)
        self._uvicorn: Any = None
        self.app = self._build_app()

    def _build_app(self) -> Any:
        app = FastAPI(title="SAHYOG SAR Mission Replay", docs_url=None,
                      redoc_url=None, openapi_url=None)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _PAGE

        @app.get("/api/events")
        def events() -> JSONResponse:
            return JSONResponse(self.events)

        @app.get("/api/health")
        def health() -> Dict[str, Any]:
            return {"ok": True, "artifact": str(self.artifact),
                    "t": time.time()}

        return app

    def serve(self) -> None:                        # pragma: no cover
        log.info("replaying %s at http://%s:%d", self.artifact, self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port,
                    log_level="warning", access_log=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a SAHYOG mission replay")
    parser.add_argument("--artifact", required=True, help="path to a mission JSON artifact")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    if not Path(args.artifact).exists():
        parser.error(f"artifact not found: {args.artifact}")
    dash = ReplayDashboard(args.artifact, host=args.host, port=args.port)
    dash.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
