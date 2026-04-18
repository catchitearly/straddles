import os
import json
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
DATES_TO_TEST = ["2026-04-06","2026-04-07","2026-04-08","2026-04-09","2026-04-10", "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_history(symbol, date, res):
    # Local Cache Path
    clean_sym = symbol.replace(":", "_")
    filename = f"{clean_sym}_{res}_{date}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    # 1. Check Local Cache
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    # 2. Fetch from API if not found
    print(f"Fetching {symbol} ({res}) for {date} from API...")
    data = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch","o","h","l","c","v"])
        df["time"] = pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
        # Save to CSV
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_simulator_data():
    master_data = {}

    for date in DATES_TO_TEST:
        day_key = date
        master_data[day_key] = {"strikes": {}, "spot": []}
        
        # Get Spot for ATM calculation
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        # Convert spot to simple list for JS
        master_data[day_key]["spot"] = nifty[['time', 'o', 'c']].to_dict('records')
        
        # ATM Logic (Python side prepares the base, JS uses it)
        open_p = nifty.iloc[0]['o']
        price_b = nifty[nifty['time'].dt.hour < 11].iloc[-1]['c']
        base_atm = int(round(price_b / 50) * 50) if abs(open_p - price_b) > 200 else int(round(open_p / 50) * 50)

        for offset in OFFSETS:
            strike = base_atm + offset
            df5 = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "5")
            pe5 = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "5")
            df1 = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date, "1")
            pe1 = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date, "1")

            if not (df5.empty or pe5.empty or df1.empty or pe1.empty):
                m5 = pd.merge(df5[['time','c']], pe5[['time','c']], on='time')
                m1 = pd.merge(df1[['time','c']], pe1[['time','c']], on='time')
                
                master_data[day_key]["strikes"][str(strike)] = {
                    "data5m": (m5['c_x'] + m5['c_y']).tolist(),
                    "times5m": m5['time'].dt.strftime("%H:%M").tolist(),
                    "data1m": (m1['c_x'] + m1['c_y']).tolist(),
                    "times1m": m1['time'].dt.strftime("%H:%M").tolist()
                }
    
    return master_data

# --- HTML GENERATOR ---
# This generates a dashboard where all logic is handled by JavaScript
def generate_interactive_html(data):
    json_data = json.dumps(data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Straddle Simulator</title>
        <style>
            body {{ background: #0b1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
            .panel {{ background: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 20px; }}
            .controls {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .control-group {{ display: flex; flex-direction: column; }}
            input {{ background: #0d1117; color: white; border: 1px solid #30363d; padding: 8px; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background: #21262d; padding: 10px; text-align: left; font-size: 12px; }}
            td {{ padding: 10px; border-bottom: 1px solid #30363d; }}
            .profit {{ color: #3fb950; font-weight: bold; }}
            .loss {{ color: #f85149; font-weight: bold; }}
            button {{ background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h2>⚡ Interactive Backtest Simulator</h2>
        
        <div class="panel">
            <div class="controls">
                <div class="control-group">
                    <label>Min Smoothness (%)</label>
                    <input type="number" id="minSmooth" value="70">
                </div>
                <div class="control-group">
                    <label>Entry Speed (pts/min)</label>
                    <input type="number" id="entrySpeed" value="-0.9" step="0.1">
                </div>
                <div class="control-group">
                    <label>Exit Speed (pts/min)</label>
                    <input type="number" id="exitSpeed" value="-0.1" step="0.1">
                </div>
                <div class="control-group">
                    <label>SL Points</label>
                    <input type="number" id="slPoints" value="10">
                </div>
                <div class="control-group" style="justify-content: flex-end;">
                    <button onclick="runSimulation()">Run Simulator</button>
                </div>
            </div>
        </div>

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
                
                // Simplified Trend (Slope)
                const x = Array.from({{length: slice.length}}, (_, i) => i);
                const xm = x.reduce((a,b)=>a+b)/x.length;
                const ym = slice.reduce((a,b)=>a+b)/slice.length;
                let num = 0, den = 0;
                for(let i=0; i<x.length; i++) {{
                    num += (x[i]-xm)*(slice[i]-ym);
                    den += Math.pow(x[i]-xm, 2);
                }}
                const angle = Math.atan(num/den) * (180/Math.PI);
                const trend = angle > 5 ? "UP" : (angle < -5 ? "DOWN" : "FLAT");
                
                return {{ smooth, speed, trend }};
            }}

            function runSimulation() {{
                const config = {{
                    smooth: parseFloat(document.getElementById('minSmooth').value),
                    entrySpeed: parseFloat(document.getElementById('entrySpeed').value),
                    exitSpeed: parseFloat(document.getElementById('exitSpeed').value),
                    sl: parseFloat(document.getElementById('slPoints').value)
                }};

                let html = '<table><thead><tr><th>Date</th><th>Strike</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead><tbody>';
                let totalPnL = 0;

                for (let date in masterData) {{
                    const day = masterData[date];
                    for (let strike in day.strikes) {{
                        const sData = day.strikes[strike];
                        let activeTrade = null;

                        // Loop through 1-minute timeline
                        for (let i = 0; i < sData.data1m.length; i++) {{
                            const time = sData.times1m[i];
                            const price = sData.data1m[i];
                            
                            // Get 5min index corresponding to this time
                            const idx5m = sData.times5m.findIndex(t => t === time);
                            const m30 = idx5m !== -1 ? calcMetrics(sData.data5m, idx5m, 6) : null;
                            const m60 = idx5m !== -1 ? calcMetrics(sData.data5m, idx5m, 12) : null;

                            if (!activeTrade) {{
                                // Entry logic
                                if (time >= "11:15" && time <= "13:30" && m30 && m60) {{
                                    if (m30.smooth >= config.smooth && m30.speed <= config.entrySpeed && m30.trend === "DOWN" && m60.trend === "DOWN") {{
                                        activeTrade = {{ entry: price, time: time, tsl: price + config.sl }};
                                    }}
                                }}
                            }} else {{
                                // Management logic (TSL)
                                let profit = activeTrade.entry - price;
                                if (profit >= 20) activeTrade.tsl = Math.min(activeTrade.tsl, activeTrade.entry - 12);
                                else if (profit >= 15) activeTrade.tsl = Math.min(activeTrade.tsl, activeTrade.entry - 8);
                                else if (profit >= 10) activeTrade.tsl = Math.min(activeTrade.tsl, activeTrade.entry - 5);
                                else if (profit >= 8) activeTrade.tsl = Math.min(activeTrade.tsl, activeTrade.entry - 3);

                                let exitReason = null;
                                if (price >= activeTrade.tsl) exitReason = "TSL Hit";
                                else if (m30 && m30.speed > config.exitSpeed) exitReason = "Speed Exit";
                                else if (m30 && m60 && (m30.trend === "UP" || m60.trend === "UP")) exitReason = "Reversal";

                                if (exitReason) {{
                                    let pnl = activeTrade.entry - price;
                                    totalPnL += pnl;
                                    html += `<tr><td>${{date}}</td><td>${{strike}}</td><td>${{activeTrade.time}} (@${{activeTrade.entry.toFixed(2)}})</td><td>${{time}} (@${{price.toFixed(2)}})</td><td class="${{pnl >= 0 ? 'profit' : 'loss'}}">${{pnl.toFixed(2)}}</td><td>${{exitReason}}</td></tr>`;
                                    activeTrade = null;
                                }}
                            }}
                        }}
                    }}
                }}
                html += `</tbody></table><h3>Total Points: <span class="${{totalPnL >= 0 ? 'profit' : 'loss'}}">${{totalPnL.toFixed(2)}}</span></h3>`;
                document.getElementById('resultsArea').innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """
    with open("simulator.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    sim_data = prepare_simulator_data()
    generate_interactive_html(sim_data)
