import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel
from itertools import product

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17", "2026-04-20"]
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- PARAMETER GRID ---
SMOOTH_RANGE   = list(range(40, 101, 5))           # 40, 45, ..., 100
ENTRY_SPEEDS   = [round(-0.3 + i * (-0.05), 2) for i in range(15)]  # -0.30, -0.35, ..., -1.00
EXIT_SPEEDS    = [round(-0.2 + i * 0.05, 2)   for i in range(9)]    # -0.20, -0.15, ..., 0.20
SL_RANGE       = list(range(10, 0, -1))            # 10, 9, ..., 1

BROKERAGE_PER_TRADE = 200  # Rs per straddle

# --- JSON FAIL-SAFE ENCODER ---
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(obj)


def get_slippage(offset: int) -> float:
    """Slippage scales linearly: 0 at ATM (offset=0), 1 point at offset=±400."""
    return abs(offset) / 400.0


def get_history(symbol, date, res):
    clean_sym = symbol.replace(":", "_")
    filename = f"{clean_sym}_{res}_{date}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    time.sleep(0.6)
    print(f"Fetching {symbol} ({res}) for {date} from API...")
    data = {
        "symbol": symbol, "resolution": res, "date_format": "1",
        "range_from": date, "range_to": date, "cont_flag": "1"
    }
    resp = fyers.history(data=data)

    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s")
                      .dt.tz_localize("UTC")
                      .dt.tz_convert(IST)
                      .dt.tz_localize(None))
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()


def prepare_simulator_data():
    master_data = {}
    for date in DATES_TO_TEST:
        master_data[date] = {"strikes": {}, "spot": []}
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty:
            continue

        master_data[date]["spot"] = nifty[['time', 'o', 'c']].to_dict('records')

        open_p  = nifty.iloc[0]['o']
        price_b = nifty[nifty['time'].dt.hour < 11].iloc[-1]['c']
        base_atm = (int(round(price_b / 100) * 100)
                    if abs(open_p - price_b) > 200
                    else int(round(open_p / 100) * 100))

        for offset in OFFSETS:
            strike = base_atm + offset
            df5_ce = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "5")
            df5_pe = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "5")
            df1_ce = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "1")
            df1_pe = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "1")

            if not (df5_ce.empty or df5_pe.empty or df1_ce.empty or df1_pe.empty):
                m5 = pd.merge(df5_ce[['time', 'c']], df5_pe[['time', 'c']], on='time')
                m1 = pd.merge(df1_ce[['time', 'c']], df1_pe[['time', 'c']], on='time')

                master_data[date]["strikes"][str(strike)] = {
                    "data5m":  (m5['c_x'] + m5['c_y']).tolist(),
                    "times5m": m5['time'].dt.strftime("%H:%M").tolist(),
                    "data1m":  (m1['c_x'] + m1['c_y']).tolist(),
                    "times1m": m1['time'].dt.strftime("%H:%M").tolist(),
                    "offset":  offset   # store offset for slippage calc
                }
    return master_data


def generate_interactive_html(data):
    json_data = json.dumps(data, cls=DateTimeEncoder)

    smooth_opts   = SMOOTH_RANGE
    espeed_opts   = ENTRY_SPEEDS
    xspeed_opts   = EXIT_SPEEDS
    sl_opts        = SL_RANGE

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Straddle Strategy Optimizer</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #080c10;
    --surface: #0e1420;
    --border: #1e2d3d;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --profit: #00ff88;
    --loss: #ff4444;
    --text: #c8d8e8;
    --muted: #5a7a9a;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Space Grotesk', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); padding: 24px; min-height: 100vh; }}
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background: radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,212,255,0.04) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,107,53,0.04) 0%, transparent 60%);
    pointer-events: none;
  }}

  h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 4px; }}
  h1 span {{ color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 13px; font-family: var(--mono); margin-bottom: 28px; }}

  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .panel-title {{ font-size: 11px; font-family: var(--mono); color: var(--accent); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; }}

  /* Config grid */
  .config-row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
  .cfg-group {{ display: flex; flex-direction: column; gap: 5px; }}
  .cfg-group label {{ font-size: 10px; font-family: var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
  .cfg-group select {{
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 7px 10px; border-radius: 5px; font-family: var(--mono); font-size: 12px;
    appearance: none; cursor: pointer; min-width: 100px;
  }}
  .cfg-group select:focus {{ outline: none; border-color: var(--accent); }}
  .info-tag {{
    font-size: 10px; font-family: var(--mono); color: var(--muted);
    background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.15);
    padding: 4px 8px; border-radius: 4px; align-self: flex-end;
  }}

  /* Buttons */
  .btn-row {{ display: flex; gap: 10px; margin-top: 4px; }}
  .btn {{
    padding: 10px 20px; border-radius: 6px; border: none; font-family: var(--mono);
    font-size: 12px; font-weight: 600; cursor: pointer; letter-spacing: 0.5px;
    transition: all 0.15s; white-space: nowrap;
  }}
  .btn-primary {{ background: var(--accent); color: #000; }}
  .btn-primary:hover {{ background: #00b8d9; transform: translateY(-1px); }}
  .btn-optimize {{ background: var(--accent2); color: #fff; }}
  .btn-optimize:hover {{ background: #e0562a; transform: translateY(-1px); }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; }}

  /* Progress */
  #progressWrap {{ display: none; margin-top: 14px; }}
  #progressBar {{ height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }}
  #progressFill {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); width: 0%; transition: width 0.1s; }}
  #progressLabel {{ font-size: 11px; font-family: var(--mono); color: var(--muted); margin-top: 6px; }}

  /* Metrics grid */
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }}
  .metric {{
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px; position: relative; overflow: hidden;
  }}
  .metric::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent); opacity: 0.4;
  }}
  .metric.highlight::before {{ opacity: 1; background: var(--accent2); }}
  .metric-label {{ font-size: 9px; font-family: var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .metric-val {{ font-size: 20px; font-family: var(--mono); font-weight: 600; }}
  .metric-val.profit {{ color: var(--profit); }}
  .metric-val.loss {{ color: var(--loss); }}
  .metric-val.neutral {{ color: var(--accent); }}

  /* Tabs */
  .tabs {{ display: flex; gap: 2px; margin-bottom: 0; }}
  .tab {{
    padding: 8px 16px; font-size: 11px; font-family: var(--mono); cursor: pointer;
    border-radius: 6px 6px 0 0; border: 1px solid var(--border); border-bottom: none;
    background: var(--bg); color: var(--muted); transition: all 0.15s;
  }}
  .tab.active {{ background: var(--surface); color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Tables */
  .tbl-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); }}
  thead th {{
    background: rgba(0,212,255,0.05); padding: 10px 12px; text-align: left;
    font-size: 10px; color: var(--accent); letter-spacing: 1px; text-transform: uppercase;
    border-bottom: 1px solid var(--border); white-space: nowrap; cursor: pointer;
  }}
  thead th:hover {{ color: #fff; }}
  tbody tr {{ border-bottom: 1px solid rgba(30,45,61,0.5); transition: background 0.1s; }}
  tbody tr:hover {{ background: rgba(0,212,255,0.03); }}
  td {{ padding: 9px 12px; white-space: nowrap; }}
  .profit {{ color: var(--profit); }}
  .loss {{ color: var(--loss); }}
  .rank-badge {{
    display: inline-block; background: var(--accent2); color: #000;
    font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 3px; margin-right: 4px;
  }}
  .best-row {{ background: rgba(255,107,53,0.05) !important; }}

  /* Score bar */
  .score-bar {{ display: flex; align-items: center; gap: 6px; }}
  .score-fill {{
    height: 6px; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2));
    min-width: 2px;
  }}
  .score-txt {{ font-size: 11px; color: var(--text); }}

  /* Heatmap */
  #heatmapContainer {{ overflow-x: auto; }}
  .heatmap-table {{ font-size: 11px; }}
  .heatmap-table th {{ font-size: 9px; padding: 6px 8px; }}
  .heatmap-table td {{ padding: 6px 8px; text-align: center; border-radius: 3px; }}

  /* Status */
  #status {{ font-size: 11px; font-family: var(--mono); color: var(--muted); margin-top: 8px; }}
  .tag {{ display: inline-block; font-size: 9px; font-family: var(--mono); padding: 2px 7px; border-radius: 3px; }}
  .tag-info {{ background: rgba(0,212,255,0.1); color: var(--accent); border: 1px solid rgba(0,212,255,0.2); }}
  .tag-warn {{ background: rgba(255,107,53,0.1); color: var(--accent2); border: 1px solid rgba(255,107,53,0.2); }}
</style>
</head>
<body>

<h1>Straddle <span>Strategy Optimizer</span></h1>
<p class="subtitle">// NIFTY · Grid Search · Brokerage ₹200/trade · Slippage 0→1pt (ATM→±400)</p>

<!-- CONFIG PANEL -->
<div class="panel">
  <div class="panel-title">⚙ Optimizer Config</div>
  <div class="config-row">
    <div class="cfg-group">
      <label>Capital (₹)</label>
      <select id="capital">
        <option value="50000">50,000</option>
        <option value="100000" selected>1,00,000</option>
        <option value="200000">2,00,000</option>
        <option value="500000">5,00,000</option>
      </select>
    </div>
    <div class="cfg-group">
      <label>Lots per Trade</label>
      <select id="lots">
        <option value="1" selected>1 lot (75 qty)</option>
        <option value="2">2 lots (150 qty)</option>
        <option value="3">3 lots (225 qty)</option>
      </select>
    </div>
    <div class="cfg-group">
      <label>Rank By</label>
      <select id="rankBy">
        <option value="score">Composite Score</option>
        <option value="totalPnL">Total P&L</option>
        <option value="sharpe">Sharpe Ratio</option>
        <option value="profitFactor">Profit Factor</option>
        <option value="winRate">Win Rate</option>
        <option value="maxDD_pct" selected>Min Drawdown</option>
      </select>
    </div>
    <div class="info-tag">
      Smooth: {len(smooth_opts)} vals · ESpeed: {len(espeed_opts)} vals · XSpeed: {len(xspeed_opts)} vals · SL: {len(sl_opts)} vals<br>
      <strong style="color:var(--accent2)">{len(smooth_opts)*len(espeed_opts)*len(xspeed_opts)*len(sl_opts):,} combinations</strong>
    </div>
    <div class="btn-row">
      <button class="btn btn-optimize" onclick="runOptimizer()" id="btnOpt">▶ RUN OPTIMIZER</button>
      <button class="btn btn-primary" onclick="runSingle()" id="btnSingle">⟳ SINGLE RUN</button>
    </div>
  </div>
  <div id="progressWrap">
    <div id="progressBar"><div id="progressFill"></div></div>
    <div id="progressLabel">Initializing...</div>
  </div>
  <div id="status"></div>
</div>

<!-- METRICS -->
<div class="panel">
  <div class="panel-title">◈ Performance Metrics</div>
  <div class="metrics-grid" id="metricsDash">
    <div class="metric"><div class="metric-label">Status</div><div class="metric-val neutral" style="font-size:14px">Run optimizer →</div></div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('opt')">Optimization Results</div>
  <div class="tab" onclick="switchTab('trades')">Trade Log</div>
  <div class="tab" onclick="switchTab('heatmap')">Heatmap</div>
</div>

<div class="panel" style="border-radius: 0 8px 8px 8px; margin-top: 0;">
  <!-- OPT RESULTS -->
  <div class="tab-content active" id="tab-opt">
    <div class="tbl-wrap">
      <table id="optTable">
        <thead>
          <tr>
            <th onclick="sortOpt('rank')">#</th>
            <th onclick="sortOpt('smooth')">Smooth%</th>
            <th onclick="sortOpt('eSpeed')">E.Speed</th>
            <th onclick="sortOpt('xSpeed')">X.Speed</th>
            <th onclick="sortOpt('sl')">SL</th>
            <th onclick="sortOpt('totalPnL')">Net P&L ₹</th>
            <th onclick="sortOpt('winRate')">Win%</th>
            <th onclick="sortOpt('sharpe')">Sharpe</th>
            <th onclick="sortOpt('profitFactor')">PF</th>
            <th onclick="sortOpt('maxDD_pct')">MaxDD%</th>
            <th onclick="sortOpt('trades')">Trades</th>
            <th onclick="sortOpt('avgDaily')">Avg/Day</th>
            <th onclick="sortOpt('score')">Score</th>
          </tr>
        </thead>
        <tbody id="optBody">
          <tr><td colspan="13" style="text-align:center;color:var(--muted);padding:40px">Run optimizer to see results</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- TRADE LOG -->
  <div class="tab-content" id="tab-trades">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Strike</th><th>Offset</th><th>Entry Time</th>
            <th>Entry ₹</th><th>Exit Time</th><th>Exit ₹</th>
            <th>Gross P&L</th><th>Slip</th><th>Brok</th><th>Net P&L</th><th>Reason</th>
          </tr>
        </thead>
        <tbody id="tradeBody">
          <tr><td colspan="12" style="text-align:center;color:var(--muted);padding:40px">Run single simulation first</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- HEATMAP -->
  <div class="tab-content" id="tab-heatmap">
    <div style="margin-bottom:12px;display:flex;gap:10px;align-items:center">
      <div class="cfg-group">
        <label>X Axis</label>
        <select id="hmX" onchange="renderHeatmap()">
          <option value="smooth">Smooth %</option>
          <option value="eSpeed" selected>Entry Speed</option>
          <option value="xSpeed">Exit Speed</option>
          <option value="sl">Init SL</option>
        </select>
      </div>
      <div class="cfg-group">
        <label>Y Axis</label>
        <select id="hmY" onchange="renderHeatmap()">
          <option value="smooth" selected>Smooth %</option>
          <option value="eSpeed">Entry Speed</option>
          <option value="xSpeed">Exit Speed</option>
          <option value="sl">Init SL</option>
        </select>
      </div>
      <div class="cfg-group">
        <label>Metric</label>
        <select id="hmMetric" onchange="renderHeatmap()">
          <option value="totalPnL">Net P&L</option>
          <option value="sharpe">Sharpe</option>
          <option value="score" selected>Score</option>
          <option value="winRate">Win Rate</option>
        </select>
      </div>
    </div>
    <div id="heatmapContainer"><p style="color:var(--muted);font-family:var(--mono);font-size:12px">Run optimizer first</p></div>
  </div>
</div>

<script>
const masterData = {json_data};
const SMOOTH_VALS  = {json.dumps(smooth_opts)};
const ESPEED_VALS  = {json.dumps(espeed_opts)};
const XSPEED_VALS  = {json.dumps(xspeed_opts)};
const SL_VALS      = {json.dumps(sl_opts)};
const BROKERAGE    = {BROKERAGE_PER_TRADE};   // per straddle (one-way already includes both legs)
const LOT_SIZE     = 75;

let allOptResults = [];
let sortCol = 'score', sortDir = -1;

// ─── UTILS ───────────────────────────────────────────────────────────────────
function std(arr) {{
  const mu = arr.reduce((a,b)=>a+b,0)/arr.length;
  return Math.sqrt(arr.map(x=>Math.pow(x-mu,2)).reduce((a,b)=>a+b,0)/arr.length)||1e-9;
}}

function calcMetrics(prices, idx, window) {{
  if (idx < window) return null;
  const slice = prices.slice(idx-window+1, idx+1);
  const net = Math.abs(slice.at(-1)-slice[0]);
  let total = 0;
  for(let i=1;i<slice.length;i++) total += Math.abs(slice[i]-slice[i-1]);
  const smooth = total > 0 ? net/total*100 : 0;
  const speed  = (slice.at(-1)-slice[0])/(window*5);
  const n = slice.length, xm=(n-1)/2;
  let num=0,den=0;
  for(let i=0;i<n;i++){{num+=(i-xm)*(slice[i]-slice.reduce((a,b)=>a+b)/n);den+=Math.pow(i-xm,2);}}
  const angle = Math.atan(num/(den||1e-9))*180/Math.PI;
  return {{ smooth, speed, trend: angle>5?"UP":(angle<-5?"DOWN":"FLAT") }};
}}

// ─── SINGLE SIMULATION ───────────────────────────────────────────────────────
function simulate(smooth, eSpeed, xSpeed, sl, lots) {{
  const qty  = lots * LOT_SIZE;
  let trades = [];

  for (let date in masterData) {{
    for (let strikeStr in masterData[date].strikes) {{
      const s      = masterData[date].strikes[strikeStr];
      const offset = s.offset !== undefined ? s.offset : 0;
      const slip   = Math.abs(offset)/400;   // 0 at ATM, 1 at ±400
      let active   = null;

      for (let i=0; i<s.data1m.length; i++) {{
        const time=s.times1m[i], price=s.data1m[i];
        const idx5m = s.times5m.indexOf(time);
        const m30   = idx5m!==-1 ? calcMetrics(s.data5m,idx5m,6)  : null;
        const m60   = idx5m!==-1 ? calcMetrics(s.data5m,idx5m,12) : null;

        if (!active) {{
          if (time>="11:15" && time<="13:30" && m30 && m60) {{
            if (m30.smooth>=smooth && m30.speed<=eSpeed &&
                m30.trend==="DOWN"  && m60.trend==="DOWN") {{
              const entryPrice = price + slip;   // adverse slippage on entry
              active = {{ date, strike:strikeStr, offset, slip, entryTime:time,
                          entryPrice, rawEntry:price,
                          tsl: entryPrice+sl }};
            }}
          }}
        }} else {{
          let pft = active.entryPrice - price;
          // TSL ladder (applied to entryPrice so slip-adjusted)
          const tsl_ladder = [
            [50, 40],[40, 30],[30, 20],[20, 10],[15, 8],[10, 5],[8, 3],[5, 2]
          ];
          for (const [thr, lock] of tsl_ladder) {{
            if (pft >= thr) {{ active.tsl = Math.min(active.tsl, active.entryPrice-lock); break; }}
          }}

          let reason = null;
          if (price>=active.tsl)                                         reason="TSL Hit";
          else if (m30 && m30.speed>xSpeed)                              reason="Speed";
          else if (m30&&m60&&(m30.trend==="UP"||m60.trend==="UP"))       reason="Trend UP";
          else if (time==="15:29")                                        reason="EOD";

          if (reason) {{
            const exitPrice  = price + slip;   // adverse slippage on exit
            const grossPnL   = (active.entryPrice - exitPrice) * qty;
            const totalSlip  = slip * 2 * qty; // entry + exit slippage × qty
            const netPnL     = grossPnL - BROKERAGE;
            trades.push({{
              date, strike:strikeStr, offset, entryTime:active.entryTime,
              entryPrice:active.entryPrice, exitTime:time, exitPrice,
              grossPnL, slippage:totalSlip, brokerage:BROKERAGE,
              netPnL, reason
            }});
            active = null;
          }}
        }}
      }}
      // Force close at EOD
      if (active) {{
        const last = s.data1m.at(-1) + active.slip;
        const netPnL = (active.entryPrice - last)*qty - BROKERAGE;
        trades.push({{ date, strike:active.strike, offset:active.offset,
          entryTime:active.entryTime, entryPrice:active.entryPrice,
          exitTime:"15:30", exitPrice:last,
          grossPnL:(active.entryPrice-last)*qty, slippage:active.slip*2*qty,
          brokerage:BROKERAGE, netPnL, reason:"EOD" }});
        active = null;
      }}
    }}
  }}
  return trades;
}}

function computeStats(trades, capital) {{
  if (!trades.length) return null;
  trades.sort((a,b)=>new Date(a.date+' '+a.entryTime)-new Date(b.date+' '+b.entryTime));

  let totalPnL=0, wins=0, losses=0, grossWin=0, grossLoss=0;
  let equity=0, peak=0, maxDD=0, streak=0, maxStreak=0;
  let dailyPnL={{}};

  trades.forEach(t=>{{
    totalPnL+=t.netPnL; equity+=t.netPnL;
    dailyPnL[t.date]=(dailyPnL[t.date]||0)+t.netPnL;
    if(t.netPnL>0){{wins++;grossWin+=t.netPnL;maxStreak=Math.max(maxStreak,streak);streak=0;}}
    else{{losses++;grossLoss+=Math.abs(t.netPnL);streak++;}}
    if(equity>peak)peak=equity;
    maxDD=Math.max(maxDD,peak-equity);
  }});
  maxStreak=Math.max(maxStreak,streak);

  const dailyArr=Object.values(dailyPnL);
  const avgDaily=dailyArr.length?totalPnL/dailyArr.length:0;
  const sharpe=dailyArr.length>1?avgDaily/std(dailyArr):0;
  const pf=grossLoss>0?grossWin/grossLoss:(grossWin>0?999:0);
  const wr=trades.length?wins/trades.length*100:0;

  // Composite score: normalised weighted blend
  const score = sharpe*0.3 + pf*0.25 + (wr/100)*0.25 + (totalPnL/capital)*0.2 - (maxDD/capital)*0.5;

  return {{ totalPnL, wins, losses, winRate:wr, sharpe, profitFactor:pf,
            maxDD, maxDD_pct:maxDD/capital*100, avgDaily,
            maxLosingStreak:maxStreak, trades:trades.length, score }};
}}

// ─── OPTIMIZER ───────────────────────────────────────────────────────────────
async function runOptimizer() {{
  const capital = parseFloat(document.getElementById('capital').value);
  const lots    = parseInt(document.getElementById('lots').value);
  const total   = SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;

  document.getElementById('btnOpt').disabled=true;
  document.getElementById('btnSingle').disabled=true;
  document.getElementById('progressWrap').style.display='block';
  document.getElementById('status').textContent='';
  allOptResults=[];

  let done=0;
  const t0=Date.now();

  for (const smooth of SMOOTH_VALS) {{
    for (const eSpeed of ESPEED_VALS) {{
      for (const xSpeed of XSPEED_VALS) {{
        for (const sl of SL_VALS) {{
          const trades = simulate(smooth, eSpeed, xSpeed, sl, lots);
          const stats  = computeStats(trades, capital);
          if (stats && stats.trades>0) {{
            allOptResults.push({{ smooth, eSpeed, xSpeed, sl, ...stats }});
          }}
          done++;
          if (done%50===0) {{
            const pct=done/total*100;
            document.getElementById('progressFill').style.width=pct+'%';
            document.getElementById('progressLabel').textContent=
              `${{done.toLocaleString()}} / ${{total.toLocaleString()}} combinations · ${{pct.toFixed(1)}}% · ${{((Date.now()-t0)/1000).toFixed(0)}}s`;
            await new Promise(r=>setTimeout(r,0)); // yield to browser
          }}
        }}
      }}
    }}
  }}

  document.getElementById('progressFill').style.width='100%';
  document.getElementById('progressLabel').textContent=`Complete · ${{allOptResults.length}} viable configs · ${{((Date.now()-t0)/1000).toFixed(1)}}s`;

  rankAndDisplay(capital, lots);
  document.getElementById('btnOpt').disabled=false;
  document.getElementById('btnSingle').disabled=false;
}}

function rankAndDisplay(capital, lots) {{
  const rankBy = document.getElementById('rankBy').value;
  const sorted = [...allOptResults].sort((a,b)=>
    rankBy==='maxDD_pct'?(a[rankBy]-b[rankBy]):(b[rankBy]-a[rankBy])
  );

  const best = sorted[0];
  if (!best) return;

  // Show metrics for best config
  updateMetrics(best, capital);
  renderOptTable(sorted.slice(0,200));  // top 200
  renderHeatmap();

  document.getElementById('status').innerHTML =
    `<span class="tag tag-info">Best: Smooth ${{best.smooth}}% · ESpeed ${{best.eSpeed}} · XSpeed ${{best.xSpeed}} · SL ${{best.sl}}</span>`;
}}

function renderOptTable(results) {{
  const maxScore=Math.max(...results.map(r=>r.score));
  let html='';
  results.forEach((r,i)=>{{
    const cls = i===0?'best-row':'';
    const pnlCls = r.totalPnL>=0?'profit':'loss';
    const scoreW = (r.score/maxScore*100).toFixed(0);
    html+=`<tr class="${{cls}}">
      <td>${{i===0?'<span class="rank-badge">★</span>':''}}${{i+1}}</td>
      <td>${{r.smooth}}</td>
      <td>${{r.eSpeed.toFixed(2)}}</td>
      <td>${{r.xSpeed.toFixed(2)}}</td>
      <td>${{r.sl}}</td>
      <td class="${{pnlCls}}">${{r.totalPnL.toFixed(0)}}</td>
      <td>${{r.winRate.toFixed(1)}}%</td>
      <td>${{r.sharpe.toFixed(3)}}</td>
      <td>${{r.profitFactor.toFixed(2)}}</td>
      <td class="loss">${{r.maxDD_pct.toFixed(2)}}%</td>
      <td>${{r.trades}}</td>
      <td>${{r.avgDaily.toFixed(0)}}</td>
      <td><div class="score-bar"><div class="score-fill" style="width:${{scoreW}}px"></div><span class="score-txt">${{r.score.toFixed(3)}}</span></div></td>
    </tr>`;
  }});
  document.getElementById('optBody').innerHTML=html;
}}

function updateMetrics(stats, capital) {{
  const m = [
    ['Net P&L ₹', stats.totalPnL.toFixed(2), stats.totalPnL>=0?'profit':'loss', true],
    ['Win Rate',  stats.winRate.toFixed(1)+'%', '', false],
    ['Sharpe',    stats.sharpe.toFixed(3), 'neutral', false],
    ['Profit Factor', stats.profitFactor.toFixed(2), 'neutral', false],
    ['Max DD',    `${{stats.maxDD.toFixed(0)}} (${{stats.maxDD_pct.toFixed(2)}}%)`, 'loss', false],
    ['Avg Day ₹', stats.avgDaily.toFixed(0), '', false],
    ['Avg Win ₹', (stats.totalPnL/stats.wins||0).toFixed(0), 'profit', false],
    ['Avg Loss ₹',(stats.grossLoss/stats.losses||0) ? '-'+(Math.abs(stats.grossLoss||0)/stats.losses).toFixed(0) : '0', 'loss', false],
    ['Trades',    `${{stats.trades}} / ${{stats.wins}}W / ${{stats.losses}}L`, 'neutral', false],
    ['Max Streak',stats.maxLosingStreak, 'loss', false],
  ];
  document.getElementById('metricsDash').innerHTML =
    m.map(([l,v,c,hi])=>
      `<div class="metric ${{hi?'highlight':''}}">
        <div class="metric-label">${{l}}</div>
        <div class="metric-val ${{c}}">${{v}}</div>
      </div>`).join('');
}}

// ─── SINGLE RUN ──────────────────────────────────────────────────────────────
function runSingle() {{
  const capital=parseFloat(document.getElementById('capital').value);
  const lots   =parseInt(document.getElementById('lots').value);

  // use best params if available, else defaults
  const cfg = allOptResults.length ? allOptResults[0] :
    {{smooth:70, eSpeed:-0.9, xSpeed:-0.1, sl:10}};

  const trades=simulate(cfg.smooth,cfg.eSpeed,cfg.xSpeed,cfg.sl,lots);
  const stats=computeStats(trades,capital);
  if(!stats)return;
  updateMetrics(stats,capital);
  renderTradeLog(trades);
  switchTab('trades');
}}

function renderTradeLog(trades) {{
  let html='';
  trades.forEach(t=>{{
    const nc=t.netPnL>=0?'profit':'loss';
    html+=`<tr>
      <td>${{t.date}}</td><td>${{t.strike}}</td>
      <td style="color:var(--muted)">${{t.offset>0?'+':''}}${{t.offset}}</td>
      <td>${{t.entryTime}}</td><td>${{t.entryPrice.toFixed(2)}}</td>
      <td>${{t.exitTime}}</td><td>${{t.exitPrice.toFixed(2)}}</td>
      <td class="${{t.grossPnL>=0?'profit':'loss'}}">${{t.grossPnL.toFixed(2)}}</td>
      <td style="color:var(--muted)">-${{t.slippage.toFixed(2)}}</td>
      <td style="color:var(--muted)">-${{t.brokerage}}</td>
      <td class="${{nc}}">${{t.netPnL.toFixed(2)}}</td>
      <td style="font-size:10px;color:var(--muted)">${{t.reason}}</td>
    </tr>`;
  }});
  document.getElementById('tradeBody').innerHTML=html||'<tr><td colspan="12" style="text-align:center;color:var(--muted);padding:20px">No trades</td></tr>';
}}

// ─── HEATMAP ─────────────────────────────────────────────────────────────────
function renderHeatmap() {{
  if(!allOptResults.length) return;
  const xKey   = document.getElementById('hmX').value;
  const yKey   = document.getElementById('hmY').value;
  const metric = document.getElementById('hmMetric').value;
  if(xKey===yKey) return;

  const xVals = [...new Set(allOptResults.map(r=>r[xKey]))].sort((a,b)=>a-b);
  const yVals = [...new Set(allOptResults.map(r=>r[yKey]))].sort((a,b)=>a-b);

  // Average metric for each x/y combo
  const grid = {{}};
  allOptResults.forEach(r=>{{
    const k=`${{r[xKey]}}|${{r[yKey]}}`;
    if(!grid[k]) grid[k]={{sum:0,n:0}};
    grid[k].sum+=r[metric]; grid[k].n++;
  }});

  const vals=Object.values(grid).map(v=>v.sum/v.n);
  const mn=Math.min(...vals), mx=Math.max(...vals);
  const norm=v=>(mx-mn)>0?(v-mn)/(mx-mn):0.5;
  const colorFn=v=>{{
    const n=norm(v);
    const r=Math.round(255*(1-n)), g=Math.round(255*n);
    return `rgba(${{r}},${{g}},50,0.7)`;
  }};

  let html=`<table class="heatmap-table" style="border-collapse:collapse;font-family:var(--mono)">`;
  html+=`<thead><tr><th style="color:var(--muted)">${{yKey}} \\ ${{xKey}}</th>`;
  xVals.forEach(x=>html+=`<th>${{typeof x==='number'&&!Number.isInteger(x)?x.toFixed(2):x}}</th>`);
  html+=`</tr></thead><tbody>`;

  yVals.forEach(y=>{{
    html+=`<tr><td style="color:var(--muted);font-size:10px;padding:6px 10px">${{typeof y==='number'&&!Number.isInteger(y)?y.toFixed(2):y}}</td>`;
    xVals.forEach(x=>{{
      const k=`${{x}}|${{y}}`, d=grid[k];
      if(d) {{
        const v=d.sum/d.n;
        html+=`<td style="background:${{colorFn(v)}};color:#fff">${{Math.round(v)}}</td>`;
      }} else {{
        html+=`<td style="color:var(--muted)">-</td>`;
      }}
    }});
    html+=`</tr>`;
  }});
  html+=`</tbody></table>`;
  document.getElementById('heatmapContainer').innerHTML=html;
}}

// ─── TABS & SORT ─────────────────────────────────────────────────────────────
function switchTab(id) {{
  document.querySelectorAll('.tab').forEach((t,i)=>{{
    t.classList.toggle('active', ['opt','trades','heatmap'][i]===id);
  }});
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  if(id==='heatmap') renderHeatmap();
}}

function sortOpt(col) {{
  if(sortCol===col) sortDir*=-1; else {{ sortCol=col; sortDir=-1; }}
  const sorted=[...allOptResults].sort((a,b)=>
    col==='maxDD_pct'?sortDir*(a[col]-b[col]):sortDir*(b[col]-a[col])
  );
  renderOptTable(sorted.slice(0,200));
}}
</script>
</body>
</html>"""

    with open("simulator_optimizer.html", "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    print("Preparing simulator data...")
    sim_data = prepare_simulator_data()
    print(f"Data ready. Generating optimizer HTML...")
    generate_interactive_html(sim_data)
    print("Done! Open simulator_optimizer.html in your browser.")
    print(f"\nGrid search will test:")
    print(f"  Smooth %   : {SMOOTH_RANGE[0]}–{SMOOTH_RANGE[-1]} (step 5)  → {len(SMOOTH_RANGE)} values")
    print(f"  Entry Speed: {ENTRY_SPEEDS[0]}–{ENTRY_SPEEDS[-1]} (step -0.05) → {len(ENTRY_SPEEDS)} values")
    print(f"  Exit Speed : {EXIT_SPEEDS[0]}–{EXIT_SPEEDS[-1]} (step +0.05) → {len(EXIT_SPEEDS)} values")
    print(f"  Init SL    : {SL_RANGE[0]}–{SL_RANGE[-1]} (step -1) → {len(SL_RANGE)} values")
    print(f"  Total combos: {len(SMOOTH_RANGE)*len(ENTRY_SPEEDS)*len(EXIT_SPEEDS)*len(SL_RANGE):,}")
