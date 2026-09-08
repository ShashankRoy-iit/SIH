"""Command-centre dashboard: zero-internet, fed only by what the link delivered.

The design constraint is in the name. Nothing on this page is fetched from
anywhere - no CDN, no font service, no map tiles, no analytics. Every byte is
served from the same process that owns the receive queue, because the scenario
this system exists for is one where the cellular network is part of what the
disaster destroyed. A dashboard that needs the internet to display the results of
a search conducted because the internet is down is not a dashboard for this
problem.

The second constraint is that it shows **what the ground station received**, not
what the aircraft knows. The aircraft may have twenty survivors queued and no
link; the command centre has zero, and that is the operationally true state. The
queue depth is displayed next to the survivor list precisely so the difference is
visible rather than silently absorbed - a controller who cannot see that reports
are pending has no way to know that the picture in front of them is incomplete.

Map rendering is a plain canvas in local metres. No tiles means no projection
service and no attribution, and the coordinates that matter for a rescue are
metres-from-origin anyway; WGS84 is shown per survivor for anyone who needs to
put it in something else.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("sar.gcs")

__all__ = ["Dashboard"]

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    _HAS_FASTAPI = True
except Exception:                                   # pragma: no cover
    _HAS_FASTAPI = False


# --------------------------------------------------------------------------- #
_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAR Command Centre</title>
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --dim:#8b949e;
  --life:#ff5c5c; --tact:#f0a94a; --situ:#58a6ff; --bulk:#6e7681; --ok:#3fb950;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:14px;padding:8px 14px;
  border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header h1{font-size:13px;margin:0;letter-spacing:.08em;text-transform:uppercase}
.pill{padding:2px 8px;border:1px solid var(--line);border-radius:10px;
  font-size:11px;color:var(--dim);white-space:nowrap}
.pill b{color:var(--fg);font-weight:600}
.pill.up{border-color:var(--ok);color:var(--ok)}
.pill.down{border-color:var(--life);color:var(--life)}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:1px;
  background:var(--line);height:calc(100vh - 45px)}
.left{background:var(--bg);display:flex;flex-direction:column;min-width:0}
.right{background:var(--panel);overflow-y:auto;min-width:0}
#mapwrap{position:relative;flex:1;min-height:0}
canvas{position:absolute;inset:0;width:100%;height:100%}
.legend{position:absolute;left:10px;bottom:10px;background:rgba(13,17,23,.86);
  border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-size:11px}
.legend div{display:flex;align-items:center;gap:6px;margin:2px 0}
.sw{width:9px;height:9px;border-radius:50%;display:inline-block}
section{border-bottom:1px solid var(--line);padding:9px 12px}
section h2{font-size:11px;margin:0 0 7px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim)}
.kv{display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:12px}
.kv span:nth-child(even){text-align:right;color:var(--fg)}
.kv span:nth-child(odd){color:var(--dim)}
.bar{height:5px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;background:var(--situ)}
.card{border:1px solid var(--line);border-left-width:3px;border-radius:5px;
  padding:7px 9px;margin-bottom:7px;background:#0d1117;cursor:pointer}
.card:hover{border-color:var(--dim)}
.card.sel{background:#1c2431}
.card.p-immediate{border-left-color:var(--life)}
.card.p-delayed{border-left-color:var(--tact)}
.card.p-minor{border-left-color:var(--situ)}
.card .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card .id{font-weight:700}
.card .age{color:var(--dim);font-size:11px}
.card .row{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;
  color:var(--dim);margin-top:2px}
.card .row b{color:var(--fg);font-weight:600}
.sigma{color:var(--tact)}
.denied .sigma{color:var(--life)}
.empty{color:var(--dim);font-size:12px;padding:6px 0}
.tag{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10.5px;
  border:1px solid var(--line);color:var(--dim)}
.tag.life{border-color:var(--life);color:var(--life)}
.tag.tact{border-color:var(--tact);color:var(--tact)}
#log{font-size:11px;color:var(--dim);max-height:150px;overflow-y:auto}
#log div{padding:1px 0;border-bottom:1px dotted #21262d}
#log time{color:var(--bulk)}
.warn{color:var(--tact)}
.crit{color:var(--life);font-weight:700}
@media(max-width:900px){main{grid-template-columns:1fr;height:auto}
  .left{height:60vh}.right{max-height:none}}
</style></head>
<body>
<header>
  <h1>SAR Command Centre</h1>
  <span class="pill" id="p-link">link <b>-</b></span>
  <span class="pill" id="p-nav">nav <b>-</b></span>
  <span class="pill" id="p-veh">vehicle <b>-</b></span>
  <span class="pill" id="p-q">queue <b>-</b></span>
  <span class="pill" id="p-rx">received <b>-</b></span>
  <span class="pill" id="p-clock">-</span>
</header>
<main>
  <div class="left">
    <div id="mapwrap">
      <canvas id="cv"></canvas>
      <div class="legend">
        <div><i class="sw" style="background:var(--life)"></i> survivor (immediate)</div>
        <div><i class="sw" style="background:var(--tact)"></i> survivor (delayed/minor)</div>
        <div><i class="sw" style="background:var(--situ)"></i> hazard</div>
        <div><i class="sw" style="background:#7ee787"></i> aircraft</div>
        <div><i class="sw" style="background:var(--bulk)"></i> radio range</div>
        <div style="margin-top:5px;color:var(--dim)">shading = P(detect) coverage</div>
      </div>
    </div>
    <section><h2>Event log</h2><div id="log"></div></section>
  </div>
  <div class="right">
    <section>
      <h2>Survivors reported to this station</h2>
      <div id="surv"></div>
    </section>
    <section>
      <h2>Coverage &amp; search progress</h2>
      <div class="kv" id="cov"></div>
    </section>
    <section>
      <h2>Hazards</h2>
      <div id="haz"></div>
    </section>
    <section>
      <h2>Data link</h2>
      <div class="kv" id="link"></div>
    </section>
    <section>
      <h2>Navigation</h2>
      <div class="kv" id="nav"></div>
    </section>
  </div>
</main>
<script>
const S={state:null,sel:null,log:[]};
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const PRIO_COLOR={immediate:'#ff5c5c',I:'#ff5c5c',delayed:'#f0a94a',D:'#f0a94a',
  minor:'#58a6ff',M:'#58a6ff'};
function fit(){const r=cv.parentElement.getBoundingClientRect(),d=devicePixelRatio||1;
  cv.width=r.width*d;cv.height=r.height*d;cx.setTransform(d,0,0,d,0,0);draw();}
addEventListener('resize',fit);

function push(m){S.log.unshift({t:new Date(),m});if(S.log.length>60)S.log.pop();
  const e=document.getElementById('log');
  e.innerHTML=S.log.map(x=>`<div><time>${x.t.toLocaleTimeString()}</time> ${x.m}</div>`).join('');}

function kv(el,pairs){el.innerHTML=pairs.filter(Boolean).map(([k,v,c])=>
  `<span>${k}</span><span${c?` class="${c}"`:''}>${v}</span>`).join('');}

function draw(){
  const st=S.state; if(!st)return;
  const w=cv.clientWidth,h=cv.clientHeight; cx.clearRect(0,0,w,h);
  const W=st.world.east_m||1,H=st.world.north_m||1;
  const s=Math.min(w/W,h/H)*0.94, ox=(w-W*s)/2, oy=(h-H*s)/2;
  // NED: north up, east right. Flip y because canvas y grows downward.
  const X=e=>ox+e*s, Y=n=>oy+(H-n)*s;
  // coverage shading
  const g=st.coverage_grid;
  if(g&&g.rows&&g.rows.length){
    const cs=g.res*s;
    for(let r=0;r<g.rows.length;r++)for(let c=0;c<g.rows[r].length;c++){
      const v=g.rows[r][c]; if(v<0.02)continue;
      const n0=(r)*g.res, e0=(c)*g.res;
      cx.fillStyle=`rgba(88,166,255,${0.06+0.5*Math.min(v,1)})`;
      cx.fillRect(X(e0),Y(n0+g.res),Math.max(cs,1),Math.max(cs,1));}
  }
  // search box
  const b=st.plan&&st.plan.search_box;
  if(b){cx.strokeStyle='#6e7681';cx.setLineDash([5,4]);cx.lineWidth=1;
    cx.strokeRect(X(b.east_m[0]),Y(b.north_m[1]),
      (b.east_m[1]-b.east_m[0])*s,(b.north_m[1]-b.north_m[0])*s);cx.setLineDash([]);}
  // ground station + radio range
  const gs=st.gcs||{n:0,e:0};
  if(st.link&&st.link.range_m){cx.beginPath();cx.arc(X(gs.e),Y(gs.n),st.link.range_m*s,0,7);
    cx.strokeStyle='rgba(110,118,129,.55)';cx.setLineDash([3,5]);cx.stroke();cx.setLineDash([]);}
  cx.fillStyle='#6e7681';cx.fillRect(X(gs.e)-4,Y(gs.n)-4,8,8);
  cx.font='10px monospace';cx.fillText('GCS',X(gs.e)+7,Y(gs.n)+3);
  // ground truth (only when the sim supplies it - grey, for scoring)
  (st.truth||[]).forEach(v=>{cx.beginPath();cx.arc(X(v.east_m),Y(v.north_m),2.5,0,7);
    cx.fillStyle='rgba(139,148,158,.5)';cx.fill();});
  // hazards
  (st.hazards||[]).forEach(hz=>{const n=hz.north_m,e=hz.east_m;
    if(n==null||e==null)return;
    cx.beginPath();cx.arc(X(e),Y(n),Math.max((hz.radius_m||15)*s,4),0,7);
    cx.strokeStyle='#58a6ff';cx.lineWidth=1.5;cx.stroke();
    cx.fillStyle='rgba(88,166,255,.18)';cx.fill();});
  // survivors with their reported uncertainty ellipse
  (st.survivors||[]).forEach(sv=>{
    const n=sv.north_m,e=sv.east_m; if(n==null||e==null)return;
    const col=PRIO_COLOR[sv.pr||sv.priority]||'#f0a94a';
    const sg=(sv.sg||sv.sigma_m||0);
    if(sg>0){cx.beginPath();cx.arc(X(e),Y(n),Math.max(sg*2*s,5),0,7);
      cx.strokeStyle=col+'88';cx.setLineDash([3,3]);cx.lineWidth=1;cx.stroke();cx.setLineDash([]);}
    cx.beginPath();cx.arc(X(e),Y(n),S.sel===sv.id?7:5,0,7);
    cx.fillStyle=col;cx.fill();
    cx.strokeStyle='#0d1117';cx.lineWidth=1.5;cx.stroke();
    cx.fillStyle=col;cx.font='10px monospace';
    cx.fillText(sv.id,X(e)+9,Y(n)+3);});
  // aircraft
  const v=st.vehicle;
  if(v&&v.n_m!=null){cx.save();cx.translate(X(v.e_m),Y(v.n_m));
    cx.beginPath();cx.moveTo(0,-8);cx.lineTo(5,6);cx.lineTo(0,3);cx.lineTo(-5,6);
    cx.closePath();cx.fillStyle='#7ee787';cx.fill();cx.restore();}
}

async function tick(){
  try{
    const r=await fetch('api/state',{cache:'no-store'});
    const st=await r.json();
    const prev=S.state;
    S.state=st;
    const nS=(st.survivors||[]).length, nP=prev?(prev.survivors||[]).length:0;
    if(nS>nP){const nw=(st.survivors||[]).slice(nP);
      nw.forEach(x=>push(`<span class="crit">SURVIVOR ${x.id}</span> prio ${x.pr||x.priority}
        &plusmn;${(x.sg||x.sigma_m||0).toFixed(1)}m${x.denied?' <span class="warn">(GPS-denied)</span>':''}`));}
    if(prev&&prev.link&&st.link&&prev.link.in_range&&!st.link.in_range)
      push('<span class="warn">data link LOST - queueing</span>');
    if(prev&&prev.link&&st.link&&!prev.link.in_range&&st.link.in_range)
      push('<span style="color:var(--ok)">data link reacquired - draining queue</span>');

    // header pills
    const L=st.link||{},Q=st.queue||{},V=st.vehicle||{},N=st.nav||{};
    const lp=document.getElementById('p-link');
    lp.className='pill '+(L.in_range?'up':'down');
    lp.innerHTML=`link <b>${L.in_range?L.transport:'OUT OF RANGE'}</b>`;
    document.getElementById('p-nav').innerHTML=
      `nav <b>${N.source||'-'}</b> &plusmn;${(N.pos_sigma_m||0).toFixed(1)}m`+
      (N.denied?` <span class="warn">${(N.since_fix_s||0).toFixed(0)}s since fix</span>`:'');
    document.getElementById('p-veh').innerHTML=
      `vehicle <b>${V.mode||'-'}</b> ${V.armed?'ARMED':'disarmed'} `+
      `${(V.alt_rel_m||0).toFixed(0)}m ${(V.gs_ms||0).toFixed(1)}m/s batt ${(V.batt_pct||0).toFixed(0)}%`;
    const pend=(Q.items||0);
    document.getElementById('p-q').className='pill'+(pend>0?' warn':'');
    document.getElementById('p-q').innerHTML=
      `queue <b>${pend}</b> msg / ${((Q.bytes||0)/1024).toFixed(1)} KiB`;
    document.getElementById('p-rx').innerHTML=`received <b>${st.received||0}</b> pkt`;
    document.getElementById('p-clock').textContent=new Date().toLocaleTimeString();

    // survivor cards
    const el=document.getElementById('surv');
    const list=st.survivors||[];
    el.innerHTML=list.length?list.map(sv=>{
      const pr=(sv.pr||sv.priority||'?');
      const pc=PRIO_COLOR[pr]||'#8b949e';
      const age=sv._rx_t?Math.round((Date.now()/1000)-sv._rx_t):0;
      const sg=(sv.sg||sv.sigma_m||0);
      return `<div class="card p-${({'I':'immediate','D':'delayed','M':'minor'})[pr]||pr} ${S.sel===sv.id?'sel':''}"
        data-n="${sv.north_m}" data-e="${sv.east_m}" onclick="sel('${sv.id}')">
        <div class="top"><span class="id" style="color:${pc}">${sv.id}</span>
          <span class="age">${age}s ago &middot; rev ${sv._rev||1}</span></div>
        <div class="row"><span>${sv.lat!=null?sv.lat.toFixed(5):'-'}, ${sv.lon!=null?sv.lon.toFixed(5):'-'}</span>
          <span class="sigma ${sv.denied?'':''}">&plusmn;${sg.toFixed(1)}m</span></div>
        <div class="row"><span>group <b>${sv.gs||1}</b> &middot; ${sv.nb==='M'?'medical':(sv.nb||'-')}</span>
          <span>${sv.xm?'<span class="tag life">RGB+thermal</span>':'<span class="tag">thermal only</span>'}</span></div>
        <div class="row"><span>time-critical <b>${sv.tc?Math.round(sv.tc/60)+' min':'-'}</b></span>
          <span>${(sv._n_updates||1)>1?sv._n_updates+' updates':'first report'}</span></div>
        ${sv.lz&&sv.lz.length?`<div class="row"><span>landing zone ${sv.lz[0].toFixed(0)} m N</span></div>`:''}
        ${sv.hz&&sv.hz.length?`<div class="row warn"><span>hazard nearby: ${sv.hz[0]}</span></div>`:''}
      </div>`;}).join('')
      :`<div class="empty">No survivors received. ${Q.items>0?
        `<span class="warn">${Q.items} message(s) still queued on the aircraft.</span>`:
        'The aircraft has reported none.'}</div>`;

    // coverage
    const C=st.coverage||{};
    kv(document.getElementById('cov'),[
      ['effective (searched box)',C.box_effective!=null?
        `${(100*C.box_effective).toFixed(1)}%`:'-'],
      ['effective (whole world)',C.effective_coverage!=null?
        `${(100*C.effective_coverage).toFixed(1)}%`:'-'],
      ['P(detect) &ge; 0.5',C.fraction_covered_p50!=null?
        `${(100*C.fraction_covered_p50).toFixed(1)}%`:'-'],
      ['mean best GSD',C.mean_best_gsd_m!=null?`${C.mean_best_gsd_m} m/px`:'-'],
      ['survivors still believed unfound',C.expected_remaining!=null?
        C.expected_remaining.toFixed(1):'-'],
      ['search progress',C.progress!=null?`${(100*C.progress).toFixed(0)}%`:'-'],
    ]);
    const cb=document.getElementById('cov');
    if(C.progress!=null){cb.insertAdjacentHTML('beforeend',
      `<div class="bar"><i style="width:${Math.min(100,100*C.progress)}%"></i></div>`);}

    // hazards
    const hz=st.hazards||[];
    document.getElementById('haz').innerHTML=hz.length?hz.map(h=>
      `<div class="row" style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0">
        <span class="tag tact">${h.hazard_class||'?'}</span>
        <span style="color:var(--dim)">n ${Math.round(h.north_m||0)} e ${Math.round(h.east_m||0)}</span>
        <span>${h.severity!=null?('sev '+h.severity):''}</span></div>`).join('')
      :'<div class="empty">None reported.</div>';

    // link
    kv(document.getElementById('link'),[
      ['transport',L.transport||'-'],
      ['in range',L.in_range?'<span style="color:var(--ok)">yes</span>':
        '<span class="crit">NO</span>'],
      ['packets delivered',`${L.sent_packets||0}`],
      ['bytes delivered',`${L.sent_bytes||0}`],
      ['packet success',`${L.packet_success_pct||0}%`],
      ['total outage',`${(L.outage_s||0).toFixed(0)} s`],
      ['air time used',`${(L.air_time_s||0).toFixed(0)} s`],
      ['updates coalesced',Q.stats?Q.stats.coalesced:0],
      ['duplicates suppressed',Q.stats?Q.stats.suppressed_duplicate:0],
      ['bulk shed to protect alerts',Q.stats?Q.stats.dropped_evicted:0],
    ]);

    // nav
    kv(document.getElementById('nav'),[
      ['source',N.source||'-'],
      ['position sigma',`${(N.pos_sigma_m||0).toFixed(2)} m`,N.denied?'crit':''],
      ['seconds since fix',`${(N.since_fix_s||0).toFixed(0)} s`,
        (N.since_fix_s||0)>30?'crit':((N.since_fix_s||0)>5?'warn':'')],
      ['satellites',N.n_satellites!=null?N.n_satellites:'-'],
      ['HDOP',N.hdop!=null?N.hdop.toFixed(2):'-'],
      ['drift rate',N.drift_rate_ms!=null?`${N.drift_rate_ms} m/s`:'-'],
      ['state',N.denied?'<span class="crit">GPS DENIED</span>':'<span style="color:var(--ok)">nominal</span>'],
    ]);
    draw();
  }catch(e){push('<span class="warn">dashboard poll failed: '+e+'</span>');}
}
function sel(id){S.sel=(S.sel===id?null:id);draw();}
window.sel=sel;
fit();tick();setInterval(tick,1000);
</script></body></html>
"""


# --------------------------------------------------------------------------- #
class Dashboard:
    """Serves the command-centre page and the state it draws from.

    Parameters
    ----------
    uplink
        The :class:`~sar.comms.link.TelemetryUplink` whose receive buffer is the
        ground picture.  The dashboard reads ``uplink.report`` and nothing else
        about the aircraft, so it cannot accidentally show knowledge the ground
        does not have.
    runner
        Optional mission runner, used for the plan outline, the coverage grid and
        - in simulation only - the ground-truth victim markers used for scoring.
    """

    def __init__(self, uplink: Any, runner: Any = None, world: Any = None,
                 host: str = "0.0.0.0", port: int = 8088,
                 coverage_downsample: int = 12) -> None:
        if not _HAS_FASTAPI:
            raise RuntimeError(
                "the dashboard needs fastapi and uvicorn; install them or run "
                "without --live (the mission still records everything to "
                "artifacts/)")
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
        app = FastAPI(title="SAR Command Centre", docs_url=None, redoc_url=None,
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

        @app.get("/api/health")
        def health() -> Dict[str, Any]:
            return {"ok": True, "t": time.time()}

        return app

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """Everything the page draws, in one object.

        Assembled from the uplink's *received* state.  The coverage grid is
        downsampled hard: at full resolution a 900 m basin on a 4 m grid is
        22500 cells, and serialising that once a second to a browser over the
        same radio-constrained network the aircraft is using would be an
        own-goal.
        """
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
        }
        # The uplink's own view of range, which the aircraft sets from its
        # distance to this station.
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
                        self.runner.coverage_in_box(box).get("effective_coverage",
                                                             0.0), 4)
            belief = getattr(self.runner, "belief", None)
            if belief is not None:
                bs = belief.summary()
                out["coverage"]["expected_remaining"] = bs["expected_remaining"]
                out["coverage"]["progress"] = bs["progress"]
            # Ground truth, for scoring.  Clearly separated and only present when
            # a world model supplied it - a real station never has this array, and
            # the page renders it grey so it is never mistaken for a report.
            if self.world is not None:
                out["truth"] = [{"vid": v.vid, "north_m": round(float(v.north), 1),
                                 "east_m": round(float(v.east), 1),
                                 "alive": bool(v.alive)}
                                for v in getattr(self.world, "victims", [])]
        if self.world is not None:
            out["world"] = {"north_m": float(self.world.north_m),
                            "east_m": float(self.world.east_m),
                            "name": getattr(self.world.spec, "name", "")}
        # Survivor ids and a stable local position for each, derived the same way
        # the runner's scorer does it: from the compact lat/lon against the
        # mission origin.  The id is the queue key, which survives across
        # revisions, so a marker does not change identity when an update lands.
        origin = getattr(self.world, "origin", None) if self.world is not None \
            else None
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
        """Coverage as a coarse 2-D array of rounded probabilities."""
        import numpy as np
        step = max(1, self.coverage_downsample)
        g = cov.p_detected[::step, ::step]
        g = np.where(g > 0.02, np.round(g, 2), 0.0)
        return {"res": cov.res * step, "rows": g.tolist()}

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Serve on a daemon thread so the mission loop keeps running."""
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

        self._server = threading.Thread(target=_run, daemon=True,
                                        name="sar-dashboard")
        self._server.start()
        # Wait for it to actually bind rather than reporting success and letting
        # the first request fail: a dashboard that is not listening is the one
        # thing here that has to work on the first try in a demo.
        for _ in range(60):
            if getattr(self._uvicorn, "started", False):
                log.info("dashboard listening on http://%s:%d", self.host,
                         self.port)
                return
            time.sleep(0.1)
        log.warning("dashboard did not confirm startup within 6 s")

    def stop(self) -> None:
        self._stop.set()
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        self._server = None
