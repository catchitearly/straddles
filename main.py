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

EXPIRY_MAP = {
    "2026-04-07": "26421", "2026-04-08": "26421", "2026-04-09": "26421",
    "2026-04-13": "26421", "2026-04-15": "26421", "2026-04-16": "26421",
    "2026-04-20": "26421", "2026-04-21": "26421", "2026-04-22": "26APR"
}

DATES_TO_TEST = list(EXPIRY_MAP.keys())
OFFSETS = [-300, -200, -100, 0, 100, 200, 300] 
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_history(symbol, date, res):
    filepath = os.path.join(DATA_DIR, f"{symbol.replace(':', '_')}_{res}_{date}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df
    time.sleep(0.7)
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
        expiry = EXPIRY_MAP.get(date)
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        nifty['sma30'] = nifty['c'].rolling(30).mean()
        nifty['sma60'] = nifty['c'].rolling(60).mean()
        nifty['trend30'] = nifty['sma30'].diff()
        nifty['trend60'] = nifty['sma60'].diff()
        nifty['time_str'] = nifty['time'].dt.strftime("%H:%M")

        morning = nifty[nifty['time_str'] <= "10:15"]
        if morning.empty: continue
        base_atm = int(round(morning.iloc[-1]['c'] / 100) * 100)
        
        trends_dict = nifty.set_index('time_str')[['trend30', 'trend60']].to_dict('index')
        master[date] = {"index_trends": trends_dict, "strikes": {}}
        
        for off in OFFSETS:
            strike = base_atm + off
            ce_sym, pe_sym = f"NSE:NIFTY{expiry}{strike}CE", f"NSE:NIFTY{expiry}{strike}PE"
            d5, d1 = get_history(ce_sym, date, "5"), get_history(ce_sym, date, "1")
            d5p, d1p = get_history(pe_sym, date, "5"), get_history(pe_sym, date, "1")
            
            if not (d5.empty or d1.empty or d5p.empty or d1p.empty):
                m5 = pd.merge(d5, d5p, on='epoch')
                m1 = pd.merge(d1, d1p, on='epoch')
                master[date]["strikes"][str(strike)] = {
                    "data5m": (m5['c_x'] + m5['c_y']).tolist(),
                    "times5m": pd.to_datetime(m5['epoch'], unit='s').dt.tz_localize("UTC").dt.tz_convert(IST).dt.strftime("%H:%M").tolist(),
                    "data1m": (m1['c_x'] + m1['c_y']).tolist(),
                    "times1m": pd.to_datetime(m1['epoch'], unit='s').dt.tz_localize("UTC").dt.tz_convert(IST).dt.strftime("%H:%M").tolist()
                }
    return master

def generate_html(data):
    json_str = json.dumps(data)
    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Elite Optimizer v4.6</title>
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #f0b90b; --profit: #02c076; --loss: #cf304a; --text: #eaecef; --muted: #848e9c; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #1e2329; cursor: pointer; border-radius: 4px; font-weight: bold; transition: 0.3s; }
        .tab.active { background: var(--accent); color: #000; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .chart-container { height: 450px; border-left: 2px solid var(--border); border-bottom: 2px solid var(--border); position: relative; margin: 40px; background: #0b0e11; }
        .dot { position: absolute; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); cursor: pointer; transform: translate(-50%, 50%); opacity: 0.8; border: 1px solid #000; }
        .dot:hover { border: 2px solid white; z-index: 100; opacity: 1; transform: scale(1.5) translate(-33%, 33%); }
        .axis-label { position: absolute; color: var(--muted); font-size: 12px; }
        input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
        table { width: 100%; border-collapse: collapse; font-size: 11px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); }
    </style>
</head>
<body>
    <h1>Strategy Analyzer v4.6</h1>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('controls')">Settings & Rank</div>
        <div class="tab" onclick="switchTab('graph')">Surface Graph View</div>
    </div>

    <div id="controls" class="tab-content active">
        <div class="panel">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <label>Entry Threshold (Speed): <span id="eVal" style="color:var(--accent)">-0.30</span></label>
                    <input type="range" id="eSpeed" min="-1.0" max="-0.1" step="0.05" value="-0.3" oninput="document.getElementById('eVal').innerText=this.value">
                </div>
                <div>
                    <label>Exit Threshold (Speed): <span id="xVal" style="color:var(--accent)">0.00</span></label>
                    <input type="range" id="xSpeed" min="-0.2" max="0.3" step="0.05" value="0.0" oninput="document.getElementById('xVal').innerText=this.value">
                </div>
            </div>
            <button onclick="runSim()" style="margin-top:20px; width:100%; padding:15px; background:var(--accent); border:none; font-weight:bold; border-radius:4px; cursor:pointer;">RUN BACKTEST ENGINE</button>
        </div>
        <div class="panel">
            <table id="rankTable">
                <thead><tr><th>SCORE</th><th>TIME</th><th>SM%</th><th>NET P&L</th><th>WIN%</th><th>TRADES</th><th>MAX DD%</th></tr></thead>
                <tbody id="rankBody"></tbody>
            </table>
        </div>
    </div>

    <div id="graph" class="tab-content">
        <div class="panel">
            <label>Filter: Only show setups with Drawdown < <span id="ddSliderVal" style="color:var(--accent)">5%</span></label>
            <input type="range" id="ddLimit" min="0.5" max="15" step="0.5" value="5" oninput="document.getElementById('ddSliderVal').innerText=this.value+'%'; updateGraph();">
            
            <div class="chart-container" id="surface">
                </div>
            <div style="text-align:center; color:var(--muted); font-size:12px; margin-top:10px;">X-Axis: Number of Trades | Y-Axis: Return on Capital (ROC %)</div>
        </div>
    </div>

    <script>
        const masterData = """ + json_str + r""";
        const CAPITAL = 500000;
        const QTY = 130;
        const TAX = 200;
        let allResults = [];

        function switchTab(id) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
            if(id === 'graph') updateGraph();
        }

        function runSim() {
            const eThr = parseFloat(document.getElementById('eSpeed').value);
            const xThr = parseFloat(document.getElementById('xSpeed').value);
            allResults = [];
            const times = ["10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15", "13:45", "14:15"];
            const smooths = [40, 50, 60, 70];

            for (let t of times) {
                for (let sm of smooths) {
                    let trades = [];
                    for (let d in masterData) {
                        for (let stk in masterData[d].strikes) {
                            trades.push(...simulate(d, masterData[d].strikes[stk], t, sm, eThr, xThr));
                        }
                    }
                    if (trades.length > 2) {
                        const net = trades.reduce((a,b)=>a+b, 0) - (trades.length * TAX);
                        const wr = (trades.filter(x=>x>0).length / trades.length)*100;
                        let pk=0, cur=0, mdd=0;
                        trades.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                        
                        const ddPct = (mdd / CAPITAL) * 100;
                        const score = net / (mdd + 500); 
                        allResults.push({ t, sm, net, wr, mdd: ddPct, count: trades.length, score, roc: (net/CAPITAL)*100 });
                    }
                }
            }
            renderTable();
            alert("Analysis Complete. Check Rank Table or Graph.");
        }

        function simulate(date, s, startTime, smooth, eThr, xThr) {
            let trds = []; let active = null; let spdHist = [0, 0];
            for (let i=1; i < s.data1m.length; i++) {
                const time = s.times1m[i]; const price = s.data1m[i];
                if (time < startTime) continue;
                const trends = masterData[date].index_trends[time] || {trend30:0, trend60:0};
                const idx5 = s.times5m.indexOf(time);
                if (idx5 < 6) continue;
                const slice = s.data5m.slice(idx5-5, idx5+1);
                const net = slice[5] - slice[0];
                let tot = 0; for(let j=1;j<6;j++) tot += Math.abs(slice[j]-slice[j-1]);
                const curSm = (Math.abs(net)/tot)*100;
                const curSpd = net/30;
                
                const isAccelerating = curSpd < spdHist[0] && spdHist[0] < spdHist[1];

                if (!active) {
                    if (trends.trend30 < 0 && trends.trend60 <= 0 && curSm >= smooth && curSpd <= eThr && isAccelerating) 
                        active = { ent: price, tsl: price + 6 };
                } else {
                    if (price >= active.tsl || curSpd > xThr || time === "15:25") {
                        trds.push((active.ent - price) * QTY); active = null;
                    }
                }
                spdHist.push(curSpd); spdHist.shift();
            }
            return trds;
        }

        function renderTable() {
            allResults.sort((a,b) => b.score - a.score);
            document.getElementById('rankBody').innerHTML = allResults.slice(0, 30).map(r => `
                <tr><td>${r.score.toFixed(2)}</td><td>${r.t}</td><td>${r.sm}</td>
                <td style="color:${r.net>0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.net).toLocaleString()}</td>
                <td>${r.wr.toFixed(1)}%</td><td>${r.count}</td><td>${r.mdd.toFixed(1)}%</td></tr>`).join('');
        }

        function updateGraph() {
            const limit = parseFloat(document.getElementById('ddLimit').value);
            const chart = document.getElementById('surface');
            chart.innerHTML = ''; // Clear
            
            if (allResults.length === 0) {
                chart.innerHTML = '<div style="padding:20px; color:var(--muted)">Run Analysis first...</div>';
                return;
            }

            const maxT = Math.max(...allResults.map(r => r.count)) || 1;
            const maxR = Math.max(...allResults.map(r => r.roc)) || 1;
            const minR = Math.min(...allResults.map(r => r.roc)) || 0;
            const rangeR = (maxR - minR) || 1;

            allResults.forEach(r => {
                if (r.mdd <= limit) {
                    const dot = document.createElement('div');
                    dot.className = 'dot';
                    // Calculate % from left (X = Count) and bottom (Y = ROC)
                    const leftPos = (r.count / maxT) * 95;
                    const bottomPos = ((r.roc - minR) / rangeR) * 95;
                    
                    dot.style.left = leftPos + '%';
                    dot.style.bottom = bottomPos + '%';
                    dot.style.background = r.roc > 0 ? 'var(--profit)' : 'var(--loss)';
                    dot.title = `Time: ${r.t} | SM: ${r.sm}% | ROC: ${r.roc.toFixed(1)}% | DD: ${r.mdd.toFixed(1)}%`;
                    
                    chart.appendChild(dot);
                }
            });
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    print("--- STEP 1: PREPARING DATA ---")
    data = prepare_data()
    if data:
        print("--- STEP 2: GENERATING DASHBOARD ---")
        generate_html(data)
        print("--- SUCCESS: Open simulator_optimizer.html ---")
    else:
        print("--- ERROR: No Data Found in cache ---")
