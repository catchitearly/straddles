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
# Backtesting window
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17","2026-04-20","2026-04-21"]
EXPIRY = "26421" 
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Fyers
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70, 80,45,55,65,75]
ENTRY_SPEEDS = [round(-0.3 - (i * 0.1), 2) for i in range(8)]
EXIT_SPEEDS  = [-0.1, -0.15, -0.2,0]
SL_RANGE     = [10, 8, 6, 5]

ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

def get_history(symbol, date, res):
    """Fetches data from cache or API."""
    filepath = os.path.join(DATA_DIR, f"{symbol.replace(':', '_')}_{res}_{date}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    time.sleep(0.6) # Safety delay
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
        
        # Determine ATM based on 10:15 AM price
        morning = nifty[nifty['time'].dt.strftime("%H:%M") <= "10:15"]
        price_b = morning.iloc[-1]['c'] if not morning.empty else nifty.iloc[0]['o']
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
    """Generates the drill-down interactive dashboard."""
    json_data = json.dumps(data)
    html_start = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Interactive Strategy Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #00d4ff; --profit: #00ff88; --loss: #ff4d4d; --text: #e1e8ed; --muted: #8899a6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; margin: 0; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; gap: 5px; margin-bottom: 1px; overflow-x: auto; }
        .tab { padding: 10px 20px; background: #1a2026; cursor: pointer; border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0; font-size: 11px; white-space: nowrap; }
        .tab.active { background: var(--surface); color: var(--accent); border-top: 2px solid var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .btn-run { background: #ff6b35; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        .histogram-container { display: flex; align-items: flex-end; gap: 4px; height: 250px; border-bottom: 2px solid var(--border); margin: 50px 0 60px 40px; position: relative; }
        .histo-bar { background: var(--accent); opacity: 0.6; flex: 1; min-width: 25px; cursor: pointer; position: relative; }
        .histo-bar:hover { opacity: 1; background: #fff; }
        .histo-label { position: absolute; bottom: -45px; left: 50%; transform: translateX(-50%) rotate(45deg); font-size: 8px; color: var(--muted); }
        .histo-val { position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 10px; font-weight: bold; color: var(--accent); }
        table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; }
        th { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); color: var(--muted); }
        td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.02); }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1 style="margin:0;">Nifty Strategy Explorer</h1>
            <p style="color:var(--muted); font-size:12px;">5 Lots | Capital ₹12.5L | Step: ₹10k P&L</p>
        </div>
        <button class="btn-run" onclick="runOptimizer()">▶ ANALYZE CLUSTERS</button>
    </div>
    <div id="progressBar" style="height:4px; background:#1a2026; margin-bottom:20px; overflow:hidden;"><div id="progressFill" style="height:100%; background:var(--accent); width:0%;"></div></div>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('distribution')">STRATEGY CLUSTERS</div>
        <div class="tab" onclick="switchTab('overall')">RANKED LIST</div>
    </div>

    <div class="panel">
        <div id="distribution" class="tab-content active">
            <div class="histogram-container" id="pnlHistogram"></div>
            <div id="drillDownSection" style="display:none; border-top: 1px solid var(--border); padding-top: 20px;">
                <h3 id="drillTitle" style="color:var(--accent)"></h3>
                <table id="drillTable">
                    <thead><tr><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>DD (₹)</th></tr></thead>
                    <tbody id="drillBody"></tbody>
                </table>
            </div>
        </div>
        <div id="overall" class="tab-content">
            <table id="overallTable"><thead><tr><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>SCORE</th></tr></thead><tbody id="overallBody"></tbody></table>
        </div>
    </div>

    <script>
        const masterData = """ + json_data + r""";
        const TIME_VALS = """ + str(ENTRY_TIMES) + r""";
        const SMOOTH_VALS = """ + str(SMOOTH_RANGE) + r""";
        const ESPEED_VALS = """ + str(ENTRY_SPEEDS) + r""";
        const XSPEED_VALS = """ + str(EXIT_SPEEDS) + r""";
        const SL_VALS = """ + str(SL_RANGE) + r""";
        const CAPITAL = 1250000;
        const QTY = 325;

        let allResults = [];

        function switchTab(t) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(h => h.classList.remove('active'));
            document.getElementById(t).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        function simulate(entryTime, smooth, eSpeed, xSpeed, sl) {
            let trades = [];
            for (let d in masterData) {
                for (let stk in masterData[d].strikes) {
                    const s = masterData[d].strikes[stk];
                    const slip = Math.abs(s.offset) / 400;
                    let active = null;
                    for (let i=0; i < s.data1m.length; i++) {
                        const time = s.times1m[i];
                        const price = s.data1m[i];
                        if (time < entryTime) continue;
                        const idx5 = s.times5m.indexOf(time);
                        const m30 = idx5 !== -1 ? (function(p, idx, w) {
                            if (idx < w) return null;
                            const slice = p.slice(idx - w + 1, idx + 1);
                            const net = slice[slice.length-1] - slice[0];
                            let total = 0; for(let j=1;j<slice.length;j++) total += Math.abs(slice[j]-slice[j-1]);
                            return { sm: (Math.abs(net)/total)*100, sp: net/(w*5) };
                        })(s.data5m, idx5, 6) : null;

                        if (!active) {
                            if (time >= entryTime && time <= "14:45" && m30 && m30.sm >= smooth && m30.sp <= eSpeed) 
                                active = { ent: price - slip, tsl: (price - slip) + sl };
                        } else {
                            if (active.ent - price >= 20) active.tsl = Math.min(active.tsl, active.ent - 10);
                            if (price >= active.tsl || (m30 && m30.sp > xSpeed) || time === "15:25") {
                                trades.push(((active.ent - (price + slip)) * QTY) - 200);
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
            const total = TIME_VALS.length * SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;
            for (let t of TIME_VALS) {
                for (let sm of SMOOTH_VALS) {
                    for (let es of ESPEED_VALS) {
                        for (let xs of XSPEED_VALS) {
                            for (let sl of SL_VALS) {
                                const trds = simulate(t, sm, es, xs, sl);
                                if (trds.length > 0) {
                                    const pnl = trds.reduce((a,b)=>a+b, 0);
                                    const wr = (trds.filter(x=>x>0).length / trds.length)*100;
                                    let pk=0, cur=0, mdd=0;
                                    trds.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                    allResults.push({ t, sm, es, xs, sl, pnl, wr, mdd, pnlBin: Math.floor(pnl/10000)*10000 });
                                }
                                count++;
                                if (count % 5000 === 0) {
                                    document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                    await new Promise(r => setTimeout(r, 0));
                                }
                            }
                        }
                    }
                }
            }
            renderHistogram();
            renderOverall();
        }

        function renderHistogram() {
            const bins = {};
            allResults.forEach(r => bins[r.pnlBin] = (bins[r.pnlBin] || 0) + 1);
            const sortedBins = Object.keys(bins).sort((a,b) => a-b);
            const max = Math.max(...Object.values(bins));
            document.getElementById('pnlHistogram').innerHTML = sortedBins.map(b => `
                <div class="histo-bar" style="height:${(bins[b]/max)*100}%" onclick="showDrill(${b})">
                    <div class="histo-val">${bins[b]}</div>
                    <div class="histo-label">₹${parseInt(b).toLocaleString()}</div>
                </div>`).join('');
        }

        function showDrill(bin) {
            const filtered = allResults.filter(r => r.pnlBin === bin).sort((a,b) => b.pnl - a.pnl);
            document.getElementById('drillDownSection').style.display = 'block';
            document.getElementById('drillTitle').innerText = `Cluster: ₹${bin.toLocaleString()} (${filtered.length} Setups)`;
            document.getElementById('drillBody').innerHTML = filtered.map(r => `
                <tr><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td>
                <td style="color:${r.pnl>=0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.pnl).toLocaleString()}</td>
                <td>${r.wr.toFixed(1)}%</td><td>₹${Math.round(r.mdd).toLocaleString()}</td></tr>`).join('');
        }

        function renderOverall() {
            allResults.sort((a,b) => b.pnl - a.pnl);
            document.getElementById('overallBody').innerHTML = allResults.slice(0, 50).map(r => `
                <tr><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td>
                <td>₹${Math.round(r.pnl).toLocaleString()}</td><td>${r.wr.toFixed(1)}%</td><td>${(r.pnl/CAPITAL*100).toFixed(2)}</td></tr>`).join('');
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_start)

if __name__ == "__main__":
    print("--- PREPARING DATA ---")
    data = prepare_data()
    if data:
        print(f"--- GENERATING DASHBOARD FOR {len(data)} DATES ---")
        generate_html(data)
        print("--- SUCCESS ---")
    else:
        print("--- ERROR: NO DATA RETRIEVED ---")
