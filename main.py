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

# Updated Expiry Map per your instructions
EXPIRY_MAP = {
    "2026-04-07": "26421",
    "2026-04-08": "26421",
    "2026-04-09": "26421",
    "2026-04-13": "26421",
    "2026-04-15": "26421",
    "2026-04-16": "26421",
    "2026-04-20": "26421",  
    "2026-04-21": "26421",
    "2026-04-22": "26APR"
}

DATES_TO_TEST = list(EXPIRY_MAP.keys())
OFFSETS = [-300, -200, -100, 0, 100, 200, 300] 
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70]
ENTRY_TIMES = [ (datetime.strptime("10:15", "%H:%M") + timedelta(minutes=15*i)).strftime("%H:%M") for i in range(18) ]

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
        
        # Calculate MTF Trends
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
                    "times1m": pd.to_datetime(m1['epoch'], unit='s').dt.tz_localize("UTC").dt.tz_convert(IST).dt.strftime("%H:%M").tolist(),
                    "offset": off
                }
    return master

def generate_html(data):
    json_str = json.dumps(data)
    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Acceleration Optimizer v4.2</title>
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #00d4ff; --profit: #02c076; --loss: #cf304a; --text: #eaecef; --muted: #848e9c; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .slider-box { margin-bottom: 15px; }
        input[type=range] { width: 100%; accent-color: var(--accent); }
        table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--border); }
        .btn-run { background: var(--accent); color: #000; border: none; padding: 12px 30px; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; }
        .win { color: var(--profit); }
    </style>
</head>
<body>
    <h1>Acceleration & MTF Optimizer (₹5L)</h1>
    <div class="grid">
        <div class="panel">
            <h3>Backtest Parameters</h3>
            <div class="slider-box">
                <label>Entry Speed (Min): <span id="eSpdVal">-0.30</span></label>
                <input type="range" id="eSpd" min="-0.8" max="-0.1" step="0.05" value="-0.3" oninput="document.getElementById('eSpdVal').innerText=this.value">
            </div>
            <div class="slider-box">
                <label>Stop Loss (Pts): <span id="slVal">6</span></label>
                <input type="range" id="sl" min="2" max="15" step="1" value="6" oninput="document.getElementById('slVal').innerText=this.value">
            </div>
            <button class="btn-run" onclick="runSimulation()">RUN ANALYSIS</button>
        </div>
        <div class="panel">
            <h3>Configuration Summary</h3>
            <p>Capital: ₹5,00,000 | Lots: 2 (130 Qty)</p>
            <p>30m Trend: Down | 60m Trend: Flat/Down</p>
            <p>Condition: Speed<sub>t</sub> < Speed<sub>t-1</sub> (Acceleration)</p>
        </div>
    </div>
    <div class="panel">
        <table id="resTable">
            <thead><tr><th>SCORE</th><th>TIME</th><th>SM%</th><th>NET P&L</th><th>WIN%</th><th>DD (₹)</th><th>TRADES</th></tr></thead>
            <tbody id="resBody"></tbody>
        </table>
    </div>

    <script>
        const masterData = """ + json_str + r""";
        const CAPITAL = 500000;
        const QTY = 130;
        const TAX = 200;

        function runSimulation() {
            const targetESpd = parseFloat(document.getElementById('eSpd').value);
            const targetSL = parseFloat(document.getElementById('sl').value);
            const smooths = """ + str(SMOOTH_RANGE) + r""";
            const sampleDate = Object.keys(masterData)[0];
            const times = Object.keys(masterData[sampleDate].index_trends).filter((_,i) => i % 10 === 0);
            
            let results = [];
            for (let t of times) {
                for (let sm of smooths) {
                    let trades = [];
                    for (let d in masterData) {
                        for (let stk in masterData[d].strikes) {
                            trades.push(...simulate(d, masterData[d].strikes[stk], t, sm, targetESpd, targetSL));
                        }
                    }
                    if (trades.length > 3) {
                        const net = trades.reduce((a,b)=>a+b, 0) - (trades.length * TAX);
                        const wr = (trades.filter(x=>x>0).length / trades.length)*100;
                        let pk=0, cur=0, mdd=0;
                        trades.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                        if (wr >= 70 && net > 0) results.push({ t, sm, net, wr, mdd, count: trades.length, score: (net/mdd) });
                    }
                }
            }
            render(results);
        }

        function simulate(date, s, startTime, smooth, eSpeed, sl) {
            let trds = []; let active = null; let prevSpeed = 0;
            const slip = Math.abs(s.offset) / 400;
            for (let i=1; i < s.data1m.length; i++) {
                const time = s.times1m[i]; const price = s.data1m[i];
                if (time < startTime) continue;
                const trends = masterData[date].index_trends[time] || {trend30:0, trend60:0};
                const mtfPass = trends.trend30 < 0 && trends.trend60 <= 0;
                const idx5 = s.times5m.indexOf(time);
                if (idx5 < 6) continue;
                const slice = s.data5m.slice(idx5-5, idx5+1);
                const net = slice[5] - slice[0];
                let tot = 0; for(let j=1;j<6;j++) tot += Math.abs(slice[j]-slice[j-1]);
                const curSm = (Math.abs(net)/tot)*100;
                const curSpd = net/30;
                if (!active) {
                    if (mtfPass && curSm >= smooth && curSpd <= eSpeed && curSpd < prevSpeed) 
                        active = { ent: price - slip, tsl: (price - slip) + sl };
                } else {
                    if (active.ent - price >= 15) active.tsl = Math.min(active.tsl, active.ent - 5);
                    if (price >= active.tsl || curSpd > -0.1 || time === "15:25") {
                        trds.push((active.ent - (price + slip)) * QTY); active = null;
                    }
                }
                prevSpeed = curSpd;
            } return trds;
        }

        function render(res) {
            res.sort((a,b) => b.score - a.score);
            document.getElementById('resBody').innerHTML = res.slice(0, 30).map(r => `
                <tr><td>${r.score.toFixed(2)}</td><td>${r.t}</td><td>${r.sm}</td>
                <td class="win">₹${Math.round(r.net).toLocaleString()}</td>
                <td>${r.wr.toFixed(1)}%</td><td>₹${Math.round(r.mdd).toLocaleString()}</td><td>${r.count}</td></tr>`).join('');
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    print("--- PREPARING DATA ---")
    data = prepare_data()
    if data:
        generate_html(data)
        print("--- SUCCESS ---")
