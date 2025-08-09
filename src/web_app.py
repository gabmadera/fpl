from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
import pandas as pd
from pathlib import Path
import numpy as np
from .ml_pipeline import MLPipeline
from .team_optimizer import TeamOptimizer
from .chip_strategy import ChipStrategyManager
from .fpl_client import FPLClient
from .scheduler import FPLScheduler
from .config import config
from .settings import load_exclusions


app = FastAPI(title="FPL AI Dashboard", version="0.1")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/predictions")
def predictions(limit: int = 0) -> list[dict]:
    path = Path("data/processed/predictions_current.csv")
    if not path.exists():
        # Generate on the fly if missing
        ml = MLPipeline()
        df = ml.predict_current()
    else:
        df = pd.read_csv(path)
    # Merge FPL status/chance so UI can show availability badges
    try:
        fpl = FPLClient()
        bs = fpl.bootstrap_static()
        elems = pd.DataFrame(bs.get("elements", []))
        teams = pd.DataFrame(bs.get("teams", []))
        if not elems.empty:
            elems = elems.rename(columns={
                "id": "player_id",
                "status": "fpl_status",
                "chance_of_playing_next_round": "chance_next",
                "team": "team_id",
            })
            # Only take team_id from elems if predictions lack it
            elem_cols = ["player_id", "fpl_status", "chance_next"]
            if "team_id" not in df.columns and "team_id" in elems.columns:
                elem_cols.append("team_id")
            df = df.merge(elems[elem_cols], on="player_id", how="left")
            # If duplicate team_id columns exist, coalesce
            if "team_id_x" in df.columns or "team_id_y" in df.columns:
                df["team_id"] = df.get("team_id_x").fillna(df.get("team_id_y"))
                df.drop(columns=[c for c in ["team_id_x", "team_id_y"] if c in df.columns], inplace=True)
        if not teams.empty:
            teams = teams.rename(columns={"id": "team_id", "short_name": "team_short"})
            if "team_short" in teams.columns:
                teams["team_short"] = teams["team_short"].astype(str).str.upper()
            if "team_id" in df.columns:
                df = df.merge(teams[["team_id", "team_short"]], on="team_id", how="left")
    except Exception as _:
        pass

    # Full list by default (limit=0); otherwise cap
    cols = [c for c in ["player_id", "name", "position", "team_id", "team_short", "price", "predicted_points", "fpl_status", "chance_next", "xg_per90", "xa_per90", "ep_component", "ep_ml"] if c in df.columns]
    out = df[cols]
    if limit and limit > 0:
        out = out.head(limit)
    return out.to_dict(orient="records")


@app.get("/leaders")
def leaders() -> dict:
    """Top players by xG/90 and xA/90 from current predictions file."""
    path = Path("data/processed/predictions_current.csv")
    if not path.exists():
        ml = MLPipeline()
        df = ml.predict_current()
    else:
        df = pd.read_csv(path)
    for col in ["xg_per90", "xa_per90"]:
        if col not in df.columns:
            df[col] = np.nan
    # Merge team short if missing
    try:
        fpl = FPLClient()
        teams = pd.DataFrame(fpl.bootstrap_static().get("teams", []))
        teams = teams.rename(columns={"id": "team_id", "short_name": "team_short"})
        teams["team_short"] = teams.get("team_short", "").astype(str).str.upper()
        if "team_id" in df.columns:
            df = df.merge(teams[["team_id", "team_short"]], on="team_id", how="left")
    except Exception:
        pass
    k = [c for c in ["player_id", "name", "position", "team_id", "team_short", "price", "xg_per90", "xa_per90", "predicted_points"] if c in df.columns]
    sort_xg = df.sort_values("xg_per90", ascending=False).head(10)[k].to_dict(orient="records")
    sort_xa = df.sort_values("xa_per90", ascending=False).head(10)[k].to_dict(orient="records")
    return {"top_xg90": sort_xg, "top_xa90": sort_xa}


@app.post("/refresh")
def refresh() -> dict:
    """Refresh pipeline: collect data, retrain if possible, predict."""
    try:
        ml = MLPipeline()
        ml.pipe.collect_fpl_snapshots()
        train_info = ml.train()
        ml.predict_current()
        return {"status": "ok", "train": train_info}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary")
def summary(budget: float = 100.0) -> dict:
    """Return predictions, optimized XI, captain/vice, and chip recommendations."""
    # Predictions
    path = Path("data/processed/predictions_current.csv")
    if not path.exists():
        ml = MLPipeline()
        preds = ml.predict_current()
    else:
        preds = pd.read_csv(path)

    # Team optimization
    optimizer = TeamOptimizer()
    team = optimizer.optimize_team(preds, budget_cap=budget)

    # Chip recommendations
    chip_mgr = ChipStrategyManager()
    fpl = FPLClient()
    try:
        current_gw = fpl.current_gameweek()
    except Exception:
        current_gw = 1
    chips = [r.__dict__ for r in chip_mgr.recommend_chips(preds, current_gw, {})]
    # Expose available chips (simple state for now)
    available_chips = {
        'wildcard': True,
        'free_hit': True,
        'triple_captain': True,
        'bench_boost': True,
    }

    # Bench suggestion: best 4 not in XI
    try:
        selected_ids = {
            p.get("player_id") for p in team.get("selected", []) if isinstance(p, dict)
        }
        bench = (
            preds[~preds["player_id"].isin(list(selected_ids))]
            .sort_values("predicted_points", ascending=False)
            .head(4)
            [[c for c in ["player_id", "name", "position", "team_id", "price", "predicted_points"] if c in preds.columns]]
            .to_dict(orient="records")
        )
    except Exception:
        bench = []

    # Team mapping (id -> short_name)
    team_map = {}
    try:
        bs = fpl.bootstrap_static()
        tdf = pd.DataFrame(bs.get("teams", []))
        if not tdf.empty:
            for _, row in tdf.iterrows():
                tid = int(row.get("id")) if row.get("id") is not None else None
                if tid is not None:
                    team_map[tid] = row.get("short_name") or row.get("name")
    except Exception:
        team_map = {}

    return {
        "top_predictions": preds.head(20)[[c for c in ["player_id", "name", "position", "team_id", "price", "predicted_points"] if c in preds.columns]].to_dict(orient="records"),
        "team": team,
        "bench": bench,
        "chips": chips,
        "gameweek": current_gw,
        "exclusions": list(load_exclusions()),
        "teams": team_map,
        "available_chips": available_chips,
        "chip_set": 1 if current_gw<=19 else 2,
    }


@app.get("/fixtures")
def fixtures() -> dict:
    """Return current and next GW fixtures in a simple calendar form."""
    fpl = FPLClient()
    try:
        current_gw = fpl.current_gameweek()
        raw = fpl.fixtures()
        fdf = pd.DataFrame(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def pack(gw: int):
        try:
            cur = fdf[fdf.get("event") == gw]
            cols = [
                c
                for c in [
                    "team_h",
                    "team_a",
                    "team_h_difficulty",
                    "team_a_difficulty",
                    "kickoff_time",
                ]
                if c in cur.columns
            ]
            return cur[cols].to_dict(orient="records")
        except Exception:
            return []

    return {"gameweek": current_gw, "current": pack(current_gw), "next": pack(current_gw + 1)}


@app.post("/run-now")
def run_now(token: str) -> dict:
    """Public trigger to run the full weekly analysis pipeline (requires token)."""
    if not config.RUN_NOW_TOKEN or token != config.RUN_NOW_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        scheduler = FPLScheduler()
        scheduler.weekly_analysis_pipeline()
        return {"status": "ok"}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>FPL AI Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Minimal extra styling; Tailwind covers most */
    .badge { display:inline-block; padding:2px 6px; font-size:11px; border-radius:9999px; }
    .badge-pos-GKP { background:#cffafe; color:#075985; }
    .badge-pos-DEF { background:#d1fae5; color:#065f46; }
    .badge-pos-MID { background:#fef3c7; color:#92400e; }
    .badge-pos-FWD { background:#ffe4e6; color:#9f1239; }
    .pitch { background: linear-gradient(180deg,#166534 0%, #065f46 100%); border-radius:14px; padding:16px; color:#e5e7eb; }
    .line { display:flex; justify-content:center; gap:14px; margin:10px 0; }
    .player-chip { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2); padding:8px 10px; border-radius:12px; backdrop-filter: blur(2px); font-size:13px; }
    .capt { border:1px solid #fbbf24; }
  </style>
  <script>
async function loadData() {
  const res = await fetch('/predictions');
  const data = await res.json();
  window.predData = data;
  try { console.log('pred sample', data.slice(0,5)); } catch(e){}
  renderTable(data);
  renderChart(data);
  loadSummary();
}

function renderTable(data) {
  const table = document.getElementById('table');
  const headers = ['player', 'team', 'price', 'predicted_points'];
  const mapTeam = (id, short) => {
    if (short) return short;
    if (window.teamMap && window.teamMap[id]) return window.teamMap[id];
    return (id !== undefined && id !== null) ? String(id) : '';
  };
  let html = '<table class="w-full text-sm"><thead><tr>'
    + headers.map(h => `<th class=\"text-left py-2 pr-2 border-b cursor-pointer\" data-sort=\"${h}\">${h}</th>`).join('')
    + '</tr></thead><tbody>';
  for (const row of data) {
    const pos = row.position || '';
    const badge = `<span class=\"badge badge-pos-${pos}\">${pos}</span>`;
    let status = '';
    if (row.fpl_status && row.fpl_status !== 'a') {
      const label = row.fpl_status === 'i' ? 'Inj' : (row.fpl_status === 's' ? 'Sus' : row.fpl_status.toUpperCase());
      status = ` <span class=\"ml-1 text-xs text-rose-700 bg-rose-100 px-1 rounded\">${label}${row.chance_next ? ' ' + row.chance_next + '%' : ''}</span>`;
    }
    const player = `${badge} <span class=\"ml-1\">${row.name ?? ''}</span>${status}`;
    const teamShortRaw = mapTeam(row.team_id, row.team_short);
    const teamShort = teamShortRaw || ((row.team_id!==undefined && row.team_id!==null)? String(row.team_id): '-');
    const file = /^[A-Z]{2,4}$/.test(String(teamShort).toUpperCase()) ? String(teamShort).toUpperCase() : '';
    const badgeImg = file ? `<img src=\"/static/badges/${file}.png\" alt=\"${file}\" class=\"inline-block w-5 h-5 mr-1 align-[-2px]\" onerror=\"this.style.display='none'\"/>` : '';
    html += '<tr>'
      + `<td class=\"py-2 pr-2 border-b\">${player}</td>`
      + `<td class=\"py-2 pr-2 border-b text-gray-700\">${badgeImg}${teamShort}</td>`
      + `<td class=\"py-2 pr-2 border-b\">${row.price ?? ''}</td>`
      + `<td class=\"py-2 pr-2 border-b font-medium\">${row.predicted_points ?? ''}</td>`
      + '</tr>';
  }
  html += '</tbody></table>';
  table.innerHTML = html;

  // Sort handlers
  const thead = table.querySelector('thead');
  if (thead) {
    thead.addEventListener('click', (e) => {
      const th = e.target.closest('th');
      if (!th || !th.dataset.sort) return;
      const key = th.dataset.sort;
      const rows = [...(window.predData || [])];
      const mapKey = {
        'player': r => r.name?.toLowerCase() || '',
        'team': r => (window.teamMap && window.teamMap[r.team_id]) ? window.teamMap[r.team_id] : '',
        'price': r => Number(r.price) || 0,
        'predicted_points': r => Number(r.predicted_points) || 0,
      };
      const getter = mapKey[key];
      if (!getter) return;
      const sorted = rows.sort((a,b) => getter(b) - getter(a));
      renderTable(sorted);
      renderChart(sorted);
    });
  }
}

function renderChart(data) {
  const subset = data.slice(0, 20);
  const y = subset.map(r => r.predicted_points);
  const x = subset.map(r => r.name);
  const trace = { x, y, mode: 'lines+markers', type: 'scatter', marker: { color: '#4f46e5' }, line: { color: '#4f46e5' } };
  Plotly.newPlot('chart', [trace], {margin: {t: 20, b: 80}, yaxis: {title: 'Predicted Points'}, xaxis: {tickangle: -45}});
}

async function loadSummary() {
  const res = await fetch('/summary');
  const sum = await res.json();
  // store team map for fixtures rendering
  window.teamMap = sum.teams || {};
  window.lastSummary = sum;
  document.getElementById('gw').textContent = `Gameweek ${sum.gameweek}`;
  updateTeamUI();
  // Re-render table with names/badges once team map available
  if (window.predData) {
    renderTable(window.predData);
  }
  // Chips
  const chips = sum.chips || [];
  const chipHtml = chips.length ? '<ul>' + chips.map(c => `<li>${c.chip_type.toUpperCase()} (Set ${c.chip_set}): ${c.reasoning}</li>`).join('') + '</ul>' : 'Hold chips - no clear opportunities';
  document.getElementById('chips').innerHTML = chipHtml;
  // Chips available
  const cav = sum.available_chips || {};
  const avail = Object.keys(cav).filter(k => cav[k]).map(k => k.replace('_',' ').toUpperCase()).join(', ');
  const chipAvailNode = document.getElementById('chips_available');
  if (chipAvailNode) chipAvailNode.innerHTML = `Available (Set ${sum.chip_set || 1}): ${avail || 'Unknown'}`;
  // Exclusions
  const ex = sum.exclusions || [];
  const exNode = document.getElementById('exclusions');
  if (exNode) exNode.innerHTML = ex.length ? ('<code>' + ex.join(', ') + '</code>') : 'None';
  // After summary loads, load fixtures so we can annotate opponents
  loadFixtures();
}

async function loadFixtures() {
  try {
    const res = await fetch('/fixtures');
    const fx = await res.json();
    // Build team id -> opponent label map for current GW
    window.oppMap = {};
    const mapTeam = (id) => (window.teamMap && window.teamMap[id]) ? window.teamMap[id] : id;
    (fx.current || []).forEach(r => {
      if (r.team_h != null && r.team_a != null) {
        window.oppMap[r.team_h] = `${mapTeam(r.team_a)} (H)`;
        window.oppMap[r.team_a] = `${mapTeam(r.team_h)} (A)`;
      }
    });
    const fmt = (rows) => rows.map(r => `<div class="text-sm text-gray-800 py-1 border-b border-gray-100">${mapTeam(r.team_h)} (H) vs ${mapTeam(r.team_a)} (A) — H:${r.team_h_difficulty ?? '-'} A:${r.team_a_difficulty ?? '-'}</div>`).join('');
    const cur = document.getElementById('fx_current');
    const nxt = document.getElementById('fx_next');
    if (cur) cur.innerHTML = fmt(fx.current || []);
    if (nxt) nxt.innerHTML = fmt(fx.next || []);
    // Re-render team lists with opponent labels now available
    updateTeamUI();
  } catch (e) {
    const cur = document.getElementById('fx_current');
    const nxt = document.getElementById('fx_next');
    if (cur) cur.innerHTML = '—';
    if (nxt) nxt.innerHTML = '—';
  }
}

function updateTeamUI() {
  const sum = window.lastSummary || {};
  const opp = window.oppMap || {};
  const mapTeam = (id) => (window.teamMap && window.teamMap[id]) ? window.teamMap[id] : id;
  // Team XI
  const team = sum.team?.selected || [];
  const gk = team.filter(p=>p.position==='GKP');
  const def_ = team.filter(p=>p.position==='DEF');
  const mid = team.filter(p=>p.position==='MID');
  const fwd = team.filter(p=>p.position==='FWD');
  const chip = (p, isCapt=false) => {
    const oppLabel = opp[p.team_id] ? ` · vs ${opp[p.team_id]}` : '';
    const c = (isCapt? ' capt':'');
    return `<div class="player-chip${c}"><span class="badge badge-pos-${p.position}">${p.position}</span> <span class="ml-1 font-medium">${p.name}</span>${oppLabel} · <span class="text-gray-200">${p.predicted_points} pts</span></div>`;
  };
  const captain = sum.team?.captain || '';
  const vice = sum.team?.vice_captain || '';
  const row = (arr) => `<div class="line">${arr.map(p=>chip(p, p.name===captain)).join('')}</div>`;
  const pitch = `<div class="pitch">${row(gk)}${row(def_)}${row(mid)}${row(fwd)}<div class="text-xs mt-2">Captain: ${captain || 'N/A'} · Vice: ${vice || 'N/A'}</div></div>`;
  const teamNode = document.getElementById('team');
  if (teamNode) teamNode.innerHTML = pitch;
  // Bench
  const bench = sum.bench || [];
  const benchNode = document.getElementById('bench');
  if (benchNode) benchNode.innerHTML = bench.length ? ('<ol>' + bench.map(p => {
    const oppLabel = opp[p.team_id] ? ` — vs ${opp[p.team_id]}` : '';
    return `<li>${p.name} (${p.position})${oppLabel} — ${p.predicted_points} pts</li>`;
  }).join('') + '</ol>') : '—';
}

window.addEventListener('load', () => { loadData(); });
  </script>
</head>
<body>
  <div class="max-w-6xl mx-auto p-6">
    <h1 class="text-3xl font-semibold mb-1">FPL AI Dashboard</h1>
    <p class="text-gray-600">Top predicted players (auto-generates predictions if missing)</p>
    <div class="flex items-center gap-3 mt-4">
      <button id="refreshBtn" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md">Refresh Data</button>
      <span id="status" class="text-gray-600"></span>
      <input id="filterName" placeholder="Type to filter by name…" class="px-3 py-2 border border-gray-300 rounded-md w-64" />
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white">
        <h3 id="gw" class="font-medium mb-2">Gameweek</h3>
        <div id="chart" style="height:360px;"></div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white">
        <h3 class="font-medium mb-2">Suggested XI & Captaincy</h3>
        <div id="team"></div>
        <h4 class="font-medium mt-3">Bench</h4>
        <div id="bench"></div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white md:col-span-2">
        <h3 class="font-medium mb-2">Chip Recommendations</h3>
        <div id="chips"></div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white md:col-span-2">
        <h3 class="font-medium mb-2">Fixtures (Current & Next GW)</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 class="font-medium mb-1">Current GW</h4>
            <div id="fx_current"></div>
          </div>
          <div>
            <h4 class="font-medium mb-1">Next GW</h4>
            <div id="fx_next"></div>
          </div>
        </div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white md:col-span-2">
        <h3 class="font-medium mb-2">Exclusions (Injuries/Unavailable)</h3>
        <p class="text-sm text-gray-600">Place FPL player_ids in <code>data/config/exclusions.json</code> to exclude.</p>
        <div id="exclusions"></div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white md:col-span-2">
        <h3 class="font-medium mb-2">Top Predictions (Table)</h3>
        <div id="table"></div>
      </div>
      <div class="border border-gray-200 rounded-xl p-4 shadow bg-white md:col-span-2">
        <h3 class="font-medium mb-2">Leaders: xG/90 and xA/90</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 class="font-medium mb-1">Top xG/90</h4>
            <div id="leaders_xg"></div>
          </div>
          <div>
            <h4 class="font-medium mb-1">Top xA/90</h4>
            <div id="leaders_xa"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
    """
    # Inject small script to wire refresh button and filter + fixtures loader
    inject = """
<script>
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('refreshBtn');
  const status = document.getElementById('status');
  const filterName = document.getElementById('filterName');
  btn.addEventListener('click', async () => {
    status.textContent = 'Refreshing...';
    try {
      const res = await fetch('/refresh', { method: 'POST' });
      const out = await res.json();
      status.textContent = 'Done';
      await loadData();
      await loadFixtures();
    } catch (e) {
      status.textContent = 'Error';
    }
  });
  filterName?.addEventListener('input', () => {
    const data = window.predData || [];
    const q = (filterName.value || '').toLowerCase();
    const filtered = q ? data.filter(r => String(r.name || '').toLowerCase().includes(q)) : data;
    renderTable(filtered);
    renderChart(filtered);
  });
  // Initial fixtures load
  loadFixtures();
  // Leaders
  (async ()=>{
    try {
      const res = await fetch('/leaders');
      const { top_xg90, top_xa90 } = await res.json();
      const fmt = (rows)=> '<table class="w-full text-sm"><thead><tr><th class="text-left py-1 border-b">Player</th><th class="text-left py-1 border-b">Team</th><th class="text-right py-1 border-b">xG/90</th></tr></thead><tbody>' + rows.map(r => `<tr><td class="py-1 border-b">${r.name}</td><td class="py-1 border-b">${(r.team_short||'')}</td><td class="py-1 border-b text-right">${(r.xg_per90??'').toFixed ? r.xg_per90.toFixed(2) : r.xg_per90}</td></tr>`).join('') + '</tbody></table>';
      const fmt2 = (rows)=> '<table class="w-full text-sm"><thead><tr><th class="text-left py-1 border-b">Player</th><th class="text-left py-1 border-b">Team</th><th class="text-right py-1 border-b">xA/90</th></tr></thead><tbody>' + rows.map(r => `<tr><td class="py-1 border-b">${r.name}</td><td class="py-1 border-b">${(r.team_short||'')}</td><td class="py-1 border-b text-right">${(r.xa_per90??'').toFixed ? r.xa_per90.toFixed(2) : r.xa_per90}</td></tr>`).join('') + '</tbody></table>';
      document.getElementById('leaders_xg').innerHTML = fmt(top_xg90 || []);
      document.getElementById('leaders_xa').innerHTML = fmt2(top_xa90 || []);
    } catch(e) {}
  })();
});
</script>
"""
    return html.replace('</body>', inject + '\n</body>')

