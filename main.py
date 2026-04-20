import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = list(range(40, 101, 5))
# -0.3 to -2.5 in decrements of 0.05
ENTRY_SPEEDS = [round(-0.3 - (i * 0.05), 2) for i in range(45)] 
# -0.2 to 0.2 in increments of 0.05
EXIT_SPEEDS  = [round(-0.2 + (i * 0.05), 2) for i in range(9)]
# 10 to 5 in decrements of 1
SL_RANGE     = list(range(10, 4, -1))

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

    time.sleep(0.8) 
    print(f"Fetching {symbol} for {date}...")
    data = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s")
                      .dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None))
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
    <title>Nifty Advanced Optimizer</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #080c10; --surface: #0e1420; --border: #1e2d3d; --accent: #00d4ff; --accent2: #ff6b35; --profit: #00ff88; --loss: #ff4444; --text: #c8d8e8; --muted: #5a7a9a; }
        body { background: var(--bg); color: var(--text); font-family: 'Space Grotesk', sans-serif; padding: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        h1 { font-size: 24px; color: var(--accent); margin-bottom: 5px; }
        .metric-card { background: var(--bg); border: 1px solid var(--border); padding: 15px; border-radius: 8px; flex: 1; min-width: 150px; }
        .metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; font-family: 'IBM Plex Mono'; }
        .metric-val { font-size: 20px; font-weight: 700; margin-top: 5px; color: var(--accent); }
        table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; }
        th { text-align: left; padding: 12px; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 10px; }
        td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .profit { color: var(--profit); } .loss { color: var(--loss); }
        .btn { background: var(--accent2); color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: 700; transition: transform 0.1s; }
        .btn:active { transform: scale(0.98); }
        #progressBar { height: 6px; background: var(--border); width: 100%; border-radius: 3px; margin-top: 15px; overflow: hidden; }
        #progressFill { height: 100%; background: var(--accent); width: 0%; transition: width 0.2s; }
    </style>
</head>
<body>
    <h1>Straddle Optimizer Engine v2.0</h1>
    <p style="color:var(--muted); font-size: 12px; margin-bottom: 20px;">
        5 Lots @ 65 Qty | Slippage: Linear (0-1) | Brokerage: ₹200/Trade | Capital: ₹12.5L
    </p>
    
    <div class="panel">
        <div style="display: flex; gap: 20px; align-items: center;">
            <div class="metric-card">
                <div class="metric-label">Total Capital</div>
                <div class="metric-val">₹12,50,000</div>
            </div>
            <button class="btn" onclick="runOptimizer()">▶ RUN 31,590 SIMULATIONS</button>
        </div>
        <div id="progressBar"><div id="progressFill"></div></div>
        <p id="status" style="font-size: 11px; margin-top: 10px; color: var(--muted); font-family: 'IBM Plex Mono';">Waiting to start...</p>
    </div>

    <div class="panel">
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>SMOOTH%</th><th>E.SPEED</th><th>X.SPEED</th><th>SL</th><th>NET P&L</th><th>ROI%</th><th>WIN%</th><th>MAX DD%</th><th>SCORE</th>
                </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
        </table>
    </div>

    <script>
"""
    script_data = f"""
        const masterData = {json_data};
        const SMOOTH_VALS = {SMOOTH_RANGE};
        const ESPEED_VALS = {ENTRY_SPEEDS};
        const XSPEED_VALS = {EXIT_SPEEDS};
        const SL_VALS = {SL_RANGE};
        const TOTAL_QTY = 325; // 5 lots * 65 qty
        const CAPITAL = 1250000; 
    """

    html_end = r"""
        function calcMetrics(prices, idx, window) {
            if (idx < window) return null;
            const slice = prices.slice(idx - window + 1, idx + 1);
            const net = Math.abs(slice[slice.length-1] - slice[0]);
            let total = 0;
            for(let i=1; i<slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
            return { 
                smooth: total > 0 ? (net / total) * 100 : 0, 
                speed: (slice[slice.length-1] - slice[0]) / (window * 5)
            };
        }

        function simulate(smooth, eSpeed, xSpeed, sl) {
            let trades = [];
            for (let date in masterData) {
                for (let strike in masterData[date].strikes) {
                    const s = masterData[date].strikes[strike];
                    const slip = Math.abs(s.offset) / 400; // Slippage: 0 at ATM, 1 at 400 offset
                    let active = null;

                    for (let i=0; i < s.data1m.length; i++) {
                        const time = s.times1m[i];
                        const price = s.data1m[i];
                        const idx5 = s.times5m.indexOf(time);
                        const m30 = idx5 !== -1 ? calcMetrics(s.data5m, idx5, 6) : null;

                        if (!active) {
                            if (time >= "11:15" && time <= "13:30" && m30) {
                                if (m30.smooth >= smooth && m30.speed <= eSpeed) {
                                    active = { entry: price - slip, tsl: (price - slip) + sl };
                                }
                            }
                        } else {
                            let pft = active.entry - price;
                            if (pft >= 20) active.tsl = Math.min(active.tsl, active.entry - 10);
                            
                            let exitReason = null;
                            if (price >= active.tsl) exitReason = "SL";
                            else if (m30 && m30.speed > xSpeed) exitReason = "Speed";
                            else if (time === "15:29") exitReason = "EOD";

                            if (exitReason) {
                                const gross = (active.entry - (price + slip)) * TOTAL_QTY;
                                const net = gross - 200; // Brokerage deduction
                                trades.push(net);
                                active = null;
                            }
                        }
                    }
                }
            }
            return trades;
        }

        async function runOptimizer() {
            const results = [];
            const total = SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;

            for (let smooth of SMOOTH_VALS) {
                for (let es of ESPEED_VALS) {
                    for (let xs of XSPEED_VALS) {
                        for (let sl of SL_VALS) {
                            const trades = simulate(smooth, es, xs, sl);
                            if (trades.length > 0) {
                                const netPnl = trades.reduce((a,b) => a+b, 0);
                                const wr = (trades.filter(t => t > 0).length / trades.length) * 100;
                                let peak = 0, cur = 0, mdd = 0;
                                trades.forEach(t => { cur += t; if(cur > peak) peak = cur; mdd = Math.max(mdd, peak - cur); });
                                
                                const roi = (netPnl / CAPITAL) * 100;
                                const score = (roi * 2) + (wr / 10) - ((mdd/CAPITAL)*50);
                                
                                results.push({ smooth, es, xs, sl, netPnl, roi, wr, mdd: (mdd/CAPITAL)*100, score });
                            }
                            count++;
                            if(count % 500 === 0) {
                                document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                document.getElementById('status').innerText = "Simulating: " + count + " / " + total;
                                await new Promise(r => setTimeout(r, 0));
                            }
                        }
                    }
                }
            }

            results.sort((a,b) => b.score - a.score);
            let rows = "";
            results.slice(0, 100).forEach(r => {
                rows += `<tr>
                    <td>${r.smooth}%</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td>
                    <td class="${r.netPnl>=0?'profit':'loss'}">₹${r.netPnl.toLocaleString('en-IN', {maximumFractionDigits:0})}</td>
                    <td>${r.roi.toFixed(2)}%</td><td>${r.wr.toFixed(1)}%</td>
                    <td class="loss">${r.mdd.toFixed(2)}%</td><td>${r.score.toFixed(2)}</td>
                </tr>`;
            });
            document.getElementById('resultsBody').innerHTML = rows;
            document.getElementById('status').innerText = "Finished. Top 100 configurations shown.";
        }
    </script>
</body>
</html>
"""
    full_html = html_start + script_data + html_end
    with open("simulator_optimizer.html", "w") as f:
        f.write(full_html)

if __name__ == "__main__":
    print("--- NIFTY STRADDLE OPTIMIZER ---")
    data = prepare_data()
    generate_html(data)
    print("Dashboard created: simulator_optimizer.html")
