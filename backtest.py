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
DATES_TO_TEST = ["2026-04-10", "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- JSON FAIL-SAFE ENCODER ---
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super(DateTimeEncoder, self).default(obj)

def get_history(symbol, date, res):
    clean_sym = symbol.replace(":", "_")
    filename = f"{clean_sym}_{res}_{date}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    time.sleep(0.6) # Anti-rate limit
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
        
        master_data[date]["spot"] = nifty[['time', 'o', 'c']].to_dict('records')
        
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
    # Using the custom encoder here to fix the Timestamp error
    json_data = json.dumps(data, cls=DateTimeEncoder)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Straddle Simulator</title>
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin-bottom: 15px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
            input {{ background: #0d1117; border: 1px solid #30363d; color: white; padding: 5px; width: 100%; }}
            button {{ background: #238636; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
            th, td {{ padding: 8px; border: 1px solid #30363d; text-align: left; }}
            .profit {{ color: #3fb950; }} .loss {{ color: #f85149; }}
        </style>
    </head>
    <body>
        <h2>📊 Straddle Strategy Simulator</h2>
        <div class="card">
            <div class="grid">
                <div><label>Smooth %</label><input type="number" id="minSmooth" value="70"></div>
                <div><label>Entry Speed</label><input type="number" id="entrySpeed" value="-0.9" step="0.1"></div>
                <div><label>Exit Speed</label><input type="number" id="exitSpeed" value="-0.1" step="0.1"></div>
                <div><label>Initial SL</label><input type="number" id="slPoints" value="10"></div>
                <div style="display:flex; align-items:flex-end;"><button onclick="runSimulation()">RUN</button></div>
            </div>
        </div>
        <div id="summary"></div>
        <div id="results"></div>

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

                let rows = ''; let totalP = 0;
                for (let date in masterData) {{
                    for (let strike in masterData[date].strikes) {{
                        const s = masterData[date].strikes[strike];
                        let active = null;
                        for (let i = 0; i < s.data1m.length; i++) {{
                            const time = s.times1m[i], price = s.data1m[i];
                            const idx5m = s.times5m.indexOf(time);
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
                                else if (m30 && m30.speed > cfg.xSpeed) reason = "Speed";
                                else if (m30 && m60 && (m30.trend === "UP" || m60.trend === "UP")) reason = "Trend UP";

                                if (reason) {{
                                    let pnl = active.entry - price; totalP += pnl;
                                    rows += `<tr><td>${{date}}</td><td>${{strike}}</td><td>${{active.time}} (@${{active.entry.toFixed(2)}})</td><td>${{time}} (@${{price.toFixed(2)}})</td><td class="${{pnl>=0?'profit':'loss'}}">${{pnl.toFixed(2)}}</td><td>${{reason}}</td></tr>`;
                                    active = null;
                                }}
                            }}
                        }}
                    }}
                }}
                document.getElementById('summary').innerHTML = `<h3>Total Points: ${{totalP.toFixed(2)}}</h3>`;
                document.getElementById('results').innerHTML = `<table><thead><tr><th>Date</th><th>Strike</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead><tbody>${{rows}}</tbody></table>`;
            }}
            window.onload = runSimulation;
        </script>
    </body>
    </html>
    """
    with open("simulator.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    sim_data = prepare_simulator_data()
    generate_interactive_html(sim_data)
