import os
import json
import time
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
DATES_TO_TEST = ["2026-04-08","2026-04-09","2026-04-10", "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_history(symbol, date, res):
    clean_sym = symbol.replace(":", "_")
    filename = f"{clean_sym}_{res}_{date}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    # Respect rate limits
    time.sleep(0.5) 
    print(f"Fetching {symbol} ({res}) for {date} from API...")
    data = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch","o","h","l","c","v"])
        df["time"] = pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_simulator_data():
    master_data = {}

    for date in DATES_TO_TEST:
        master_data[date] = {"strikes": {}, "spot": []}
        
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        # Format spot timestamps for JSON
        spot_copy = nifty[['time', 'o', 'c']].copy()
        spot_copy['time'] = spot_copy['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        master_data[date]["spot"] = spot_copy.to_dict('records')
        
        open_p = nifty.iloc[0]['o']
        price_b = nifty[nifty['time'].dt.hour < 11].iloc[-1]['c']
        base_atm = int(round(price_b / 50) * 50) if abs(open_p - price_b) > 200 else int(round(open_p / 50) * 50)

        for offset in OFFSETS:
            strike = base_atm + offset
            df5_ce = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "5")
            df5_pe = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "5")
            df1_ce = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "1")
            df1_pe = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "1")

            if not (df5_ce.empty or df5_pe.empty or df1_ce.empty or df1_pe.empty):
                m5 = pd.merge(df5_ce[['time','c']], df5_pe[['time','c']], on='time')
                m1 = pd.merge(df1_ce[['time','c']], df1_pe[['time','c']], on='time')
                
                master_data[date]["strikes"][str(strike)] = {
                    "data5m": (m5['c_x'] + m5['c_y']).tolist(),
                    "times5m": m5['time'].dt.strftime("%H:%M").tolist(),
                    "data1m": (m1['c_x'] + m1['c_y']).tolist(),
                    "times1m": m1['time'].dt.strftime("%H:%M").tolist()
                }
    return master_data

def generate_interactive_html(data):
    json_data = json.dumps(data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Straddle Simulator Pro</title>
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; padding: 30px; line-height: 1.5; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            label {{ display: block; font-size: 12px; color: #8b949e; margin-bottom: 5px; }}
            input {{ background: #0d1117; border: 1px solid #30363d; color: white; padding: 8px; border-radius: 6px; width: 100%; box-sizing: border-box; }}
            button {{ background: #238636; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }}
            button:hover {{ background: #2ea043; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; border: 1px solid #30363d; }}
            th {{ background: #161b22; color: #8b949e; text-align: left; padding: 12px; font-size: 12px; border-bottom: 1px solid #30363d; }}
            td {{ padding: 12px; border-bottom: 1px solid #30363d; font-size: 14px; }}
            .profit {{ color: #3fb950; font-weight: bold; }}
            .loss {{ color: #f85149; font-weight: bold; }}
            .stat-box {{ font-size: 24px; font-weight: bold; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h2>📊 Straddle Strategy Simulator</h2>
        
        <div class="card">
            <div class="grid">
                <div><label>Min Smoothness (%)</label><input type="number" id="minSmooth" value="70"></div>
                <div><label>Entry Speed (pts/min)</label><input type="number" id="entrySpeed" value="-0.9" step="0.1"></div>
                <div><label>Exit Speed Threshold</label><input type="number" id="exitSpeed" value="-0.1" step="0.05"></div>
                <div><label>Initial SL (pts)</label><input type="number" id="slPoints" value="10"></div>
            </div>
            <div style="margin-top: 20px; text-align: right;"><button onclick="runSimulation()">Run Simulator</button></div>
        </div>

        <div id="summaryArea"></div>
        <div id="resultsArea"></div>

        <script>
            const masterData = {json_data};

            function calcMetrics(prices, idx, window) {{
                if (idx < window) return null;
                const slice = prices.slice(idx - window + 1, idx + 1);
                const net = Math.abs(slice[slice.length-1] - slice[0]);
                let total = 0;
                for(let i=1; i<slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
                const smooth = total > 0 ? (net / total * 100) : 0;
                const speed = (slice[slice.length-1] - slice[0]) / (window * 5);
                
                // Simple Linear Slope
                const x = Array.from({{length: slice.length}}, (_, i) => i);
                const xm = x.reduce((a,b)=>a+b)/x.length;
                const ym = slice.reduce((a,b)=>a+b)/slice.length;
                let num = 0, den = 0;
                for(let i=0; i<x.length; i++) {{
                    num += (x[i]-xm)*(slice[i]-ym);
                    den += Math.pow(x[i]-xm, 2);
                }}
                const angle = Math.atan(num/den) * (180/Math.PI);
                return {{ smooth, speed, trend: angle > 5 ? "UP" : (angle < -5 ? "DOWN" : "FLAT") }};
            }}

            function runSimulation() {{
                const cfg = {{
                    smooth: parseFloat(document.getElementById('minSmooth').value),
                    eSpeed: parseFloat(document.getElementById('entrySpeed').value),
                    xSpeed: parseFloat(document.getElementById('exitSpeed').value),
                    sl: parseFloat(document.getElementById('slPoints').value)
                }};

                let rows = '';
                let totalPts = 0, winCount = 0, totalCount = 0;

                for (let date in masterData) {{
                    const day = masterData[date];
                    for (let strike in day.strikes) {{
                        const s = day.strikes[strike];
                        let active = null;

                        for (let i = 0; i < s.data1m.length; i++) {{
                            const time = s.times1m[i], price = s.data1m[i];
                            const idx5m = s.times5m.findIndex(t => t === time);
                            const m30 = idx5m !== -1 ? calcMetrics(s.data5m, idx5m, 6) : null;
                            const m60 = idx5m !== -1 ? calcMetrics(s.data5m, idx5m, 12) : null;

                            if (!active) {{
                                if (time >= "11:15" && time <= "13:30" && m30 && m60) {{
                                    if (m30.smooth >= cfg.smooth && m30.speed <= cfg.eSpeed && m30.trend === "DOWN" && m60.trend === "DOWN") {{
                                        active = {{ entry: price, time: time, tsl: price + cfg.sl }};
                                    }}
                                }}
                            }} else {{
                                let pft = active.entry - price;
                                if (pft >= 20) active.tsl = Math.min(active.tsl, active.entry - 12);
                                else if (pft >= 15) active.tsl = Math.min(active.tsl, active.entry - 8);
                                else if (pft >= 10) active.tsl = Math.min(active.tsl, active.entry - 5);
                                else if (pft >= 8) active.tsl = Math.min(active.tsl, active.entry - 3);

                                let reason = null;
                                if (price >= active.tsl) reason = "TSL Hit";
                                else if (m30 && m30.speed > cfg.xSpeed) reason = "Speed Slowdown";
                                else if (m30 && m60 && (m30.trend === "UP" || m60.trend === "UP")) reason = "Trend UP";

                                if (reason) {{
                                    let pnl = active.entry - price;
                                    totalPts += pnl; totalCount++; if(pnl>0) winCount++;
                                    rows += `<tr><td>${{date}}</td><td>${{strike}}</td><td>${{active.time}} (@${{active.entry.toFixed(2)}})</td><td>${{time}} (@${{price.toFixed(2)}})</td><td class="${{pnl >= 0 ? 'profit' : 'loss'}}">${{pnl.toFixed(2)}}</td><td>${{reason}}</td></tr>`;
                                    active = null;
                                }}
                            }}
                        }}
                    }}
                }}
                document.getElementById('summaryArea').innerHTML = `
                    <div class="card grid">
                        <div><label>Total P&L (Points)</label><div class="stat-box ${{totalPts>=0?'profit':'loss'}}">${{totalPts.toFixed(2)}}</div></div>
                        <div><label>Win Rate</label><div class="stat-box">${{totalCount>0?((winCount/totalCount)*100).toFixed(1):0}}%</div></div>
                        <div><label>Total Trades</label><div class="stat-box">${{totalCount}}</div></div>
                    </div>`;
                document.getElementById('resultsArea').innerHTML = `<table><thead><tr><th>Date</th><th>Strike</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead><tbody>${{rows}}</tbody></table>`;
            }}
            window.onload = runSimulation;
        </script>
    </body>
    </html>
    """
    with open("simulator.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    print("Preparing simulator data...")
    sim_data = prepare_simulator_data()
    print("Generating interactive HTML...")
    generate_interactive_html(sim_data)
    print("Done!")
