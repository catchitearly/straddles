import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
# Selected dates for backtesting
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17","2026-04-20"]
EXPIRY = "26421"  # Nifty April Expiry
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Fyers
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70, 80]
ENTRY_SPEEDS = [round(-0.3 - (i * 0.1), 2) for i in range(8)]
EXIT_SPEEDS  = [-0.1, -0.15, -0.2]
SL_RANGE     = [10, 8, 6, 5]

# 15-Minute Entry Increments: 10:15 AM to 2:45 PM
ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(obj)

def get_history(symbol, date, res):
    """Fetches data from cache or API."""
    filepath = os.path.join(DATA_DIR, f"{symbol.replace(':', '_')}_{res}_{date}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    time.sleep(0.5)  # Rate limit protection
    print(f"Fetching {symbol} for {date}...")
    arg = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=arg)
    
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s")
                      .dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None))
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_data():
    """Builds the master dataset for simulation."""
    master = {}
    for date in DATES_TO_TEST:
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        open_p = nifty.iloc[0]['o']
        morning = nifty[nifty['time'].dt.hour < 11]
        price_b = morning.iloc[-1]['c'] if not morning.empty else open_p
        base_atm = int(round(price_b / 100) * 100)
        
        master[date] = {"strikes": {}}
        for off in OFFSETS:
            strike = base_atm + off
            ce_sym, pe_sym = f"NSE:NIFTY{EXPIRY}{strike}CE", f"NSE:NIFTY{EXPIRY}{strike}PE"
            
            d5ce, d5pe = get_history(ce_sym, date, "5"), get_history(pe_sym, date, "5")
            d1ce, d1pe = get_history(ce_sym, date, "1"), get_history(pe_sym, date, "1")
            
            if not (d5ce.empty or d5pe.empty or d1ce.empty or d1pe.empty):
                m5 = pd.merge(d5ce[['time', 'c']], d5pe[['time', 'c']], on='time')
                m1 = pd.merge(d1ce[['time', 'c']], d1pe[['time', 'c']], on='time')
                master[date]["strikes"][str(strike)] = {
                    "data5m": (m5['c_x'] + m5['c_y']).tolist(),
                    "times5m": m5['time'].dt.strftime("%H:%M").tolist(),
                    "data1m": (m1['c_x'] + m1['c_y']).tolist(),
                    "times1m": m1['time'].dt.strftime("%H:%M").tolist(),
                    "offset": off
                }
    return master

def generate_html(data):
    """Generates the advanced interactive dashboard."""
    json_data = json.dumps(data, cls=DateTimeEncoder)
    
    html_start = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Professional Straddle Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #00d4ff; --profit: #00ff88; --loss: #ff4d4d; --text: #e1e8ed; --muted: #8899a6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; margin: 0; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; overflow-x: auto; gap: 5px; margin-bottom: 1px; }
        .tab { padding: 10px 20px; background: #1a2026; cursor: pointer; border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0; font-size: 12px; font-weight: bold; white-space: nowrap; }
        .tab.active { background: var(--surface); color: var(--accent); border-top: 2px solid var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #1a2026; padding: 15px; border-radius: 8px; border-left: 4px solid var(--accent); }
        .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
        .stat-val { font-size: 18px; font-weight: 700; margin-top: 5px; font-family: 'IBM Plex Mono'; }
        table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; }
        th { text-align: left; background: rgba(255,255,255,0.03); padding: 12px; color: var(--muted); border-bottom: 1px solid var(--border); }
        td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.02); }
        .btn-run { background: #ff6b35; color: white; border: none; padding: 15px 40px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }
        .chart-container { display: flex; align-items: flex-end; gap: 2px; height: 100px; border-bottom: 1px solid var(--border); margin: 20px 0; }
        .bar { background: var(--accent); opacity: 0.6; flex: 1; min-width: 4px; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1 style="margin:0; color:var(--accent);">Nifty Straddle Analytics v3.0</h1>
            <p style="color:var(--muted); font-size:12px; margin-top:5px;">5 Lots (325 Qty) | Capital: ₹12.5L | Multi-Window Stats</p>
        </div>
        <button class="btn-run" onclick="runOptimizer()">▶ CALCULATE DASHBOARD</button>
    </div>
    <div id="progressBar" style="height:4px; background:#1a2026; border-radius:2px; margin-bottom:20px; overflow:hidden;"><div id="progressFill" style="height:100%; background:var(--accent); width:0%;"></div></div>
    <div class="tabs" id="tabHeaders">
        <div class="tab active" onclick="switchTab('overall')">OVERALL STATS</div>
        <div class="tab" onclick="switchTab('analytics')">P&L GRAPHS</div>
    </div>
    <div class="panel">
        <div id="overall" class="tab-content active">
            <div class="stat-grid" id="mainStats"></div>
            <table id="overallTable"><thead><tr><th>RANK</th><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>DD%</th><th>SCORE</th></tr></thead><tbody id="overallBody"></tbody></table>
        </div>
        <div id="analytics" class="tab-content">
            <h3>Top 100 Setup P&L Distribution</h3>
            <div class="chart-container" id="pnlChart"></div>
            <div class="stat-grid" id="distributionStats"></div>
        </div>
    </div>
    <script>
"""
    script_data = f"""
        const masterData = {json_data};
        const TIME_VALS = {ENTRY_TIMES};
        const SMOOTH_VALS = {SMOOTH_RANGE};
        const ESPEED_VALS = {ENTRY_SPEEDS};
        const XSPEED_VALS = {EXIT_SPEEDS};
        const SL_VALS = {SL_RANGE};
        const CAPITAL = 1250000;
        const QTY = 325;
    """

    html_end = r"""
        let allResults = [];
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }
        function calculateStats(arr) {
            if (arr.length === 0) return { mean: 0, median: 0, mode: 0 };
            const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
            const sorted = [...arr].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            return { mean, median };
        }
        function simulate(entryTime, smooth, eSpeed, xSpeed, sl) {
            let trades = [];
            for (let date in masterData) {
                for (let strike in masterData[date].strikes) {
                    const s = masterData[date].strikes[strike];
                    const slip = Math.abs(s.offset) / 400;
                    let active = null;
                    for (let i=0; i < s.data1m.length; i++) {
                        const time = s.times1m[i];
                        const price = s.data1m[i];
                        if (time < entryTime) continue;
                        const idx5 = s.times5m.indexOf(time);
                        const m30 = idx5 !== -1 ? (function(p, idx, w) {
                            const slice = p.slice(idx - w + 1, idx + 1);
                            const net = slice[slice.length-1] - slice[0];
                            let total = 0; for(let j=1;j<slice.length;j++) total += Math.abs(slice[j]-slice[j-1]);
                            return { smooth: (Math.abs(net)/total)*100, speed: net/(w*5) };
                        })(s.data5m, idx5, 6) : null;
                        if (!active) {
                            if (time >= entryTime && time <= "14:45" && m30 && m30.smooth >= smooth && m30.speed <= eSpeed) 
                                active = { entry: price - slip, tsl: (price - slip) + sl };
                        } else {
                            if (active.entry - price >= 20) active.tsl = Math.min(active.tsl, active.entry - 10);
                            if (price >= active.tsl || (m30 && m30.speed > xSpeed) || time === "15:25") {
                                trades.push(((active.entry - (price + slip)) * QTY) - 200);
                                active = null;
                            }
                        }
                    }
                }
            }
            return trades;
        }
        async function runOptimizer() {
            allResults = [];
            const headers = document.getElementById('tabHeaders');
            const total = TIME_VALS.length * SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;
            for (let t of TIME_VALS) {
                if(!document.getElementById('pane-'+t)) {
                    const tab = document.createElement('div');
                    tab.className = 'tab'; tab.innerText = t; tab.onclick = (e) => { switchTab('pane-'+t); };
                    headers.appendChild(tab);
                    const pane = document.createElement('div');
                    pane.id = 'pane-'+t; pane.className = 'tab-content panel';
                    pane.innerHTML = `<h3>Window: ${t}</h3><table><thead><tr><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>SCORE</th></tr></thead><tbody id="body-${t}"></tbody></table>`;
                    document.body.appendChild(pane);
                }
                for (let sm of SMOOTH_VALS) {
                    for (let es of ESPEED_VALS) {
                        for (let xs of XSPEED_VALS) {
                            for (let sl of SL_VALS) {
                                const trades = simulate(t, sm, es, xs, sl);
                                if (trades.length > 2) {
                                    const pnl = trades.reduce((a,b) => a+b, 0);
                                    const wr = (trades.filter(x => x > 0).length / trades.length) * 100;
                                    let pk=0, cur=0, mdd=0;
                                    trades.forEach(x => { cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                    const score = (pnl/CAPITAL)*20 + (wr/10) - (mdd/CAPITAL)*100;
                                    allResults.push({ t, sm, es, xs, sl, pnl, wr, mdd:(mdd/CAPITAL)*100, score });
                                }
                                count++;
                                if(count % 2500 === 0) {
                                    document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                    await new Promise(r => setTimeout(r, 0));
                                }
                            }
                        }
                    }
                }
            }
            allResults.sort((a,b) => b.score - a.score);
            document.getElementById('overallBody').innerHTML = allResults.slice(0, 50).map((r,i) => `<tr><td>#${i+1}</td><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td><td style="color:${r.pnl>=0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.pnl).toLocaleString()}</td><td>${r.wr.toFixed(1)}%</td><td>${r.mdd.toFixed(2)}%</td><td>${r.score.toFixed(2)}</td></tr>`).join('');
            const stats = calculateStats(allResults.map(r => r.pnl));
            document.getElementById('distributionStats').innerHTML = `<div class="stat-card"><div class="stat-label">Mean P&L</div><div class="stat-val">₹${Math.round(stats.mean).toLocaleString()}</div></div><div class="stat-card"><div class="stat-label">Median P&L</div><div class="stat-val">₹${Math.round(stats.median).toLocaleString()}</div></div>`;
            document.getElementById('pnlChart').innerHTML = allResults.slice(0, 100).map(r => `<div class="bar" style="height:${(r.pnl/allResults[0].pnl*100)}%"></div>`).join('');
            TIME_VALS.forEach(t => {
                const timeData = allResults.filter(r => r.t === t).sort((a,b) => b.score - a.score).slice(0, 20);
                const tbody = document.getElementById('body-'+t);
                if(tbody) tbody.innerHTML = timeData.map(r => `<tr><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td><td style="color:${r.pnl>=0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.pnl).toLocaleString()}</td><td>${r.wr.toFixed(1)}%</td><td>${r.score.toFixed(2)}</td></tr>`).join('');
            });
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_start + script_data + html_end)

if __name__ == "__main__":
    print("--- STARTING NIFTY STRADDLE OPTIMIZATION ---")
    master_data = prepare_data()
    if not master_data:
        print("CRITICAL ERROR: No data found. Check your API credentials or cache.")
    else:
        print(f"Success: Data processed for {len(master_data)} dates.")
        generate_html(master_data)
        if os.path.exists("simulator_optimizer.html"):
            print("REPORT GENERATED: simulator_optimizer.html")
        else:
            print("ERROR: Failed to save the report file.")
