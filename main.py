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
ENTRY_SPEEDS = [round(-0.3 + i * (-0.05), 2) for i in range(15)]
EXIT_SPEEDS  = [round(-0.2 + i * 0.05, 2) for i in range(9)]
SL_RANGE     = list(range(10, 0, -1))

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
    
    # Using {{ }} to escape literal curly braces for CSS and JS
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nifty Straddle Optimizer</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{ --bg: #080c10; --surface: #0e1420; --border: #1e2d3d; --accent: #00d4ff; --accent2: #ff6b35; --profit: #00ff88; --loss: #ff4444; --text: #c8d8e8; --muted: #5a7a9a; }}
        body {{ background: var(--bg); color: var(--text); font-family: 'Space Grotesk', sans-serif; padding: 20px; }}
        .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
        h1 {{ font-size: 24px; color: var(--accent); margin-bottom: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }}
        .metric {{ background: var(--bg); border: 1px solid var(--border); padding: 15px; border-radius: 8px; }}
        .metric-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; font-family: 'IBM Plex Mono'; }}
        .metric-val {{ font-size: 18px; font-weight: 700; margin-top: 5px; }}
        .profit {{ color: var(--profit); }} .loss {{ color: var(--loss); }}
        table {{ width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 12px; }}
        th {{ text-align: left; padding: 10px; border-bottom: 1px solid var(--border); color: var(--accent); }}
        td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .btn {{ background: var(--accent2); color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
        #progressBar {{ height: 4px; background: var(--border); width: 100%; border-radius: 2px; margin-top: 10px; overflow: hidden; }}
        #progressFill {{ height: 100%; background: var(--accent); width: 0%; transition: width 0.1s; }}
    </style>
</head>
<body>
    <h1>Straddle Optimizer Engine</h1>
    <p style="color:var(--muted); font-size: 12px; margin-bottom: 20px;">Logic: Adverse Slippage (0→1pt) | Rs 200 Brokerage</p>
    
    <div class="panel">
        <div style="display: flex; gap: 15px; align-items: flex-end; flex-wrap: wrap;">
            <div>
                <label style="display:block; font-size: 10px; color: var(--muted);">CAPITAL (INR)</label>
                <select id="capital" class="btn" style="background: var(--bg); border: 1px solid var(--border);">
                    <option value="100000">1,00,000</option>
                    <option value="200000">2,00,000</option>
                    <option value="500000">5,00,000</option>
                </select>
            </div>
            <button class="btn" onclick="runOptimizer()">▶ START OPTIMIZATION</button>
        </div>
        <div id="progressBar"><div id="progressFill"></div></div>
        <p id="status" style="font-size: 11px; margin-top: 10px; color: var(--muted); font-family: 'IBM Plex Mono';"></p>
    </div>

    <div class="panel">
        <div class="metrics-grid" id="metrics">
            <div class="metric"><div class="metric-label">Status</div><div class="metric-val">Ready</div></div>
        </div>
    </div>

    <div class="panel">
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>Smooth%</th><th>E.Speed</th><th>X.Speed</th><th>SL</th><th>Net P&L</th><th>Win%</th><th>MaxDD%</th><th>Score</th>
                </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
        </table>
    </div>

    <script>
        const masterData = {json_data};
        const SMOOTH_VALS = {SMOOTH_RANGE};
        const ESPEED_VALS = {ENTRY_SPEEDS};
        const XSPEED_VALS = {EXIT_SPEEDS};
        const SL_VALS = {SL_RANGE};

        function calcMetrics(prices, idx, window) {{
            if (idx < window) return null;
            const slice = prices.slice(idx - window + 1, idx + 1);
            const net = Math.abs(slice[slice.length-1] - slice[0]);
            let total = 0;
            for(let i=1; i<slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
            return {{ 
                smooth: total > 0 ? (net / total) * 100 : 0, 
                speed: (slice[slice.length-1] - slice[0]) / (window * 5)
            }};
        }}

        function simulate(smooth, eSpeed, xSpeed, sl) {{
            let trades = [];
            for (let date in masterData) {{
                for (let strike in masterData[date].strikes) {{
                    const s = masterData[date].strikes[strike];
                    const slip = Math.abs(s.offset) / 400;
                    let active = null;

                    for (let i=0; i < s.data1m.length; i++) {{
                        const time = s.times1m[i];
                        const price = s.data1m[i];
                        const idx5 = s.times5m.indexOf(time);
                        const m30 = idx5 !== -1 ? calcMetrics(s.data5m, idx5, 6) : null;

                        if (!active) {{
                            if (time >= "11:15" && time <= "13:30" && m30) {{
                                if (m30.smooth >= smooth && m30.speed <= eSpeed) {{
                                    active = {{ entry: price - slip, tsl: (price - slip) + sl }};
                                }}
                            }}
                        }} else {{
                            let pft = active.entry - price;
                            if (pft >= 20) active.tsl = Math.min(active.tsl, active.entry - 10);
                            
                            let exitReason = null;
                            if (price >= active.tsl) exitReason = "SL";
                            else if (m30 && m30.speed > xSpeed) exitReason = "Speed";
                            else if (time === "15:29") exitReason = "EOD";

                            if (exitReason) {{
                                const net = ((active.entry - (price + slip)) * 75) - 200;
                                trades.push(net);
                                active = null;
                            }}
                        }}
                    }}
                }}
            }
            return trades;
        }}

        async function runOptimizer() {{
            const capital = parseInt(document.getElementById('capital').value);
            const results = [];
            const total = SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;

            for (let smooth of SMOOTH_VALS) {{
                for (let es of ESPEED_VALS) {{
                    for (let xs of XSPEED_VALS) {{
                        for (let sl of SL_VALS) {{
                            const trades = simulate(smooth, es, xs, sl);
                            if (trades.length > 0) {{
                                const netPnl = trades.reduce((a,b) => a+b, 0);
                                const wins = trades.filter(t => t > 0).length;
                                const wr = (wins / trades.length) * 100;
                                
                                let peak = 0, cur = 0, mdd = 0;
                                trades.forEach(t => {{ cur += t; if(cur > peak) peak = cur; mdd = Math.max(mdd, peak - cur); }});
                                
                                const score = (netPnl / capital) * 10 + (wr / 100);
                                results.push({{ smooth, es, xs, sl, netPnl, wr, mdd: (mdd/capital)*100, score }});
                            }}
                            count++;
                            if(count % 200 === 0) {{
                                document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                document.getElementById('status').innerText = "Processing: " + count + " / " + total;
                                await new Promise(r => setTimeout(r, 0));
                            }}
                        }}
                    }}
                }}
            }

            results.sort((a,b) => b.score - a.score);
            let html = "";
            results.slice(0, 100).forEach(r => {{
                html += `<tr><td>${{r.smooth}}%</td><td>${{r.es}}</td><td>${{r.xs}}</td><td>${{r.sl}}</td><td class="${{r.netPnl>=0?'profit':'loss'}}">₹${{r.netPnl.toFixed(0)}}</td><td>${{r.wr.toFixed(1)}}%</td><td class="loss">${{r.mdd.toFixed(2)}}%</td><td>${{r.score.toFixed(2)}}</td></tr>`;
            }});
            document.getElementById('resultsBody').innerHTML = html;
            document.getElementById('status').innerText = "Complete!";
        }}
    </script>
</body>
</html>"""

    with open("simulator_optimizer.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    print("Preparing Data...")
    m_data = prepare_data()
    generate_html(m_data)
    print("Optimization Report Ready.")
