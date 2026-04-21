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
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17","2026-04-20"]
EXPIRY = "26421" 
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40,45, 50,55, 60,65, 70,75, 80]
ENTRY_SPEEDS = [round(-0.3 - (i * 0.1), 2) for i in range(8)]
EXIT_SPEEDS  = [-0.1, -0.15, -0.2,0]
SL_RANGE     = [10, 8, 6, 5]

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
    filepath = os.path.join(DATA_DIR, f"{symbol.replace(':', '_')}_{res}_{date}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df
    time.sleep(0.5)
    print(f"Fetching {symbol} for {date}...")
    arg = {"symbol": symbol, "resolution": res, "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=arg)
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None))
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_data():
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
    json_data = json.dumps(data, cls=DateTimeEncoder)
    html_start = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nifty Strategy Distribution Optimizer</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #00d4ff; --profit: #00ff88; --loss: #ff4d4d; --text: #e1e8ed; --muted: #8899a6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; margin: 0; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; overflow-x: auto; gap: 5px; margin-bottom: 1px; }
        .tab { padding: 10px 20px; background: #1a2026; cursor: pointer; border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0; font-size: 11px; font-weight: bold; white-space: nowrap; }
        .tab.active { background: var(--surface); color: var(--accent); border-top: 2px solid var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #1a2026; padding: 15px; border-radius: 8px; border-left: 4px solid var(--accent); }
        .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
        .stat-val { font-size: 18px; font-weight: 700; margin-top: 5px; font-family: 'IBM Plex Mono'; }
        table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; }
        th { text-align: left; background: rgba(255,255,255,0.03); padding: 10px; color: var(--muted); border-bottom: 1px solid var(--border); }
        td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.02); }
        .btn-run { background: #ff6b35; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; }
        .histogram-container { display: flex; align-items: flex-end; gap: 4px; height: 300px; border-bottom: 2px solid var(--border); border-left: 2px solid var(--border); padding: 0 10px; margin: 40px 0 60px 40px; position: relative; }
        .histo-bar { background: var(--accent); opacity: 0.7; flex: 1; min-width: 20px; position: relative; transition: opacity 0.2s; }
        .histo-bar:hover { opacity: 1; }
        .histo-label { position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); font-size: 9px; white-space: nowrap; color: var(--muted); }
        .histo-val { position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 10px; font-weight: bold; color: var(--accent); }
        .y-axis-label { position: absolute; left: -45px; top: 50%; transform: translateY(-50%) rotate(-90deg); font-size: 10px; color: var(--muted); }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1 style="margin:0; color:var(--accent);">Nifty Strategy Distribution</h1>
            <p style="color:var(--muted); font-size:12px; margin-top:5px;">Frequency Analysis (All Combinations) | ₹10k P&L Step | Capital: ₹12.5L</p>
        </div>
        <button class="btn-run" onclick="runOptimizer()">▶ RUN FULL CALCULATION</button>
    </div>
    <div id="progressBar" style="height:4px; background:#1a2026; border-radius:2px; margin-bottom:20px; overflow:hidden;"><div id="progressFill" style="height:100%; background:var(--accent); width:0%;"></div></div>
    <div class="tabs" id="tabHeaders">
        <div class="tab active" onclick="switchTab('overall')">OVERALL STATS</div>
        <div class="tab" onclick="switchTab('distribution')">STRATEGY DISTRIBUTION</div>
    </div>
    <div class="panel">
        <div id="overall" class="tab-content active">
            <div class="stat-grid" id="mainStats"></div>
            <table id="overallTable"><thead><tr><th>RANK</th><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>ROI%</th><th>WIN%</th><th>SCORE</th></tr></thead><tbody id="overallBody"></tbody></table>
        </div>
        <div id="distribution" class="tab-content">
            <div class="stat-grid" id="distStats"></div>
            <div class="histogram-container" id="pnlHistogram">
                <div class="y-axis-label">No. of Combinations</div>
            </div>
            <p style="text-align:center; color:var(--muted); font-size:11px;">X-Axis: Net P&L (₹10,000 Step) | Hover to see exact count</p>
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
                                if (trades.length > 1) {
                                    const pnl = trades.reduce((a,b) => a+b, 0);
                                    const wr = (trades.filter(x => x > 0).length / trades.length) * 100;
                                    let pk=0, cur=0, mdd=0;
                                    trades.forEach(x => { cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                    const score = (pnl/CAPITAL)*20 + (wr/10) - (mdd/CAPITAL)*100;
                                    allResults.push({ t, sm, es, xs, sl, pnl, wr, mdd:(mdd/CAPITAL)*100, score });
                                }
                                count++;
                                if(count % 5000 === 0) {
                                    document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                    await new Promise(r => setTimeout(r, 0));
                                }
                            }
                        }
                    }
                }
            }

            allResults.sort((a,b) => b.score - a.score);
            document.getElementById('overallBody').innerHTML = allResults.slice(0, 50).map((r,i) => `<tr><td>#${i+1}</td><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td><td style="color:${r.pnl>=0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.pnl).toLocaleString()}</td><td>${(r.pnl/CAPITAL*100).toFixed(1)}%</td><td>${r.wr.toFixed(1)}%</td><td>${r.score.toFixed(2)}</td></tr>`).join('');

            // BUCKETIZING FOR HISTOGRAM (Step: 10,000)
            const minPnL = Math.floor(Math.min(...allResults.map(r => r.pnl)) / 10000) * 10000;
            const maxPnL = Math.ceil(Math.max(...allResults.map(r => r.pnl)) / 10000) * 10000;
            const bins = {};
            for (let i = minPnL; i <= maxPnL; i += 10000) bins[i] = 0;
            
            allResults.forEach(r => {
                const b = Math.floor(r.pnl / 10000) * 10000;
                bins[b]++;
            });

            const maxBinCount = Math.max(...Object.values(bins));
            const histogram = document.getElementById('pnlHistogram');
            histogram.innerHTML = '<div class="y-axis-label">No. of Combinations</div>' + Object.keys(bins).map(b => {
                const count = bins[b];
                const height = (count / maxBinCount) * 100;
                return `<div class="histo-bar" style="height:${height}%" title="P&L: ₹${b} | Count: ${count}">
                            <div class="histo-val">${count}</div>
                            <div class="histo-label">₹${parseInt(b).toLocaleString()}</div>
                        </div>`;
            }).join('');

            // Statistical Cards
            const pnls = allResults.map(r => r.pnl);
            const mean = pnls.reduce((a,b)=>a+b,0) / pnls.length;
            const sorted = [...pnls].sort((a,b)=>a-b);
            const median = sorted[Math.floor(sorted.length/2)];
            document.getElementById('mainStats').innerHTML = `
                <div class="stat-card"><div class="stat-label">Total Combinations</div><div class="stat-val">${allResults.length.toLocaleString()}</div></div>
                <div class="stat-card"><div class="stat-label">Mean P&L</div><div class="stat-val">₹${Math.round(mean).toLocaleString()}</div></div>
                <div class="stat-card"><div class="stat-label">Median P&L</div><div class="stat-val">₹${Math.round(median).toLocaleString()}</div></div>
            `;
            
            TIME_VALS.forEach(t => {
                const timeData = allResults.filter(r => r.t === t).sort((a,b) => b.score - a.score).slice(0, 15);
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
    master_data = prepare_data()
    if master_data:
        generate_html(master_data)
        print("SUCCESS: Full Histogram Dashboard Generated.")
