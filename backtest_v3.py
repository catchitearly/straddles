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
DATES_TO_TEST = [ "2026-04-07", "2026-04-08", "2026-04-09","2026-04-10", "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17","2026-04-20"]
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

    time.sleep(0.6) 
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
        base_atm = int(round(price_b / 100) * 100) if abs(open_p - price_b) > 200 else int(round(open_p / 100) * 100)

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
    json_data = json.dumps(data, cls=DateTimeEncoder)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Straddle Metrics Dashboard</title>
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 20px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
            .metric-item {{ border: 1px solid #30363d; padding: 15px; border-radius: 6px; background: #0d1117; text-align: center; }}
            .metric-label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; margin-bottom: 5px; }}
            .metric-value {{ font-size: 18px; font-weight: bold; }}
            .config-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; align-items: end; }}
            input {{ background: #0d1117; border: 1px solid #30363d; color: white; padding: 8px; border-radius: 4px; width: 100%; box-sizing: border-box; }}
            button {{ background: #238636; color: white; border: none; padding: 10px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
            th {{ background: #21262d; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #30363d; }}
            .profit {{ color: #3fb950; }} .loss {{ color: #f85149; }}
        </style>
    </head>
    <body>
        <h2>📈 Strategy Metrics & Simulator</h2>
        
        <div class="card">
            <div class="config-grid">
                <div><label>Capital</label><input type="number" id="capital" value="100000"></div>
                <div><label>Smooth %</label><input type="number" id="minSmooth" value="70"></div>
                <div><label>Entry Speed</label><input type="number" id="entrySpeed" value="-0.9" step="0.1"></div>
                <div><label>Exit Speed</label><input type="number" id="exitSpeed" value="-0.1" step="0.05"></div>
                <div><label>Init SL</label><input type="number" id="slPoints" value="10"></div>
                <div><button onclick="runSimulation()">REFRESH</button></div>
            </div>
        </div>

        <div id="metricsDashboard" class="metrics-grid"></div>
        <div id="tradeLog" class="card"></div>

        <script>
            const masterData = {json_data};

            function std(arr) {{
                const mu = arr.reduce((a, b) => a + b, 0) / arr.length;
                return Math.sqrt(arr.map(x => Math.pow(x - mu, 2)).reduce((a, b) => a + b, 0) / arr.length);
            }}

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
                    cap: parseFloat(document.getElementById('capital').value),
                    smooth: parseFloat(document.getElementById('minSmooth').value),
                    eSpeed: parseFloat(document.getElementById('entrySpeed').value),
                    xSpeed: parseFloat(document.getElementById('exitSpeed').value),
                    sl: parseFloat(document.getElementById('slPoints').value)
                }};

                let allTrades = [];
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
                                        active = {{ date, strike, entryTime: time, entryPrice: price, tsl: price + cfg.sl }};
                                    }}
                                }}
                            }} else {{
                                let pft = active.entryPrice - price;
                                if (pft >= 20) active.tsl = Math.min(active.tsl, active.entryPrice - 12);
                                else if (pft >= 15) active.tsl = Math.min(active.tsl, active.entryPrice - 8);
                                else if (pft >= 10) active.tsl = Math.min(active.tsl, active.entryPrice - 5);
                                else if (pft >= 5) active.tsl = Math.min(active.tsl, active.entryPrice - 2);
                                else if (pft >= 8) active.tsl = Math.min(active.tsl, active.entryPrice - 3);
                                else if (pft >= 20) active.tsl = Math.min(active.tsl, active.entryPrice - 10);
                                else if (pft >= 30) active.tsl = Math.min(active.tsl, active.entryPrice - 20);
                                else if (pft >= 40) active.tsl = Math.min(active.tsl, active.entryPrice - 30);
                                else if (pft >= 50) active.tsl = Math.min(active.tsl, active.entryPrice - 40);

                                let reason = null;
                                if (price >= active.tsl) reason = "TSL Hit";
                                else if (m30 && m30.speed > cfg.xSpeed) reason = "Speed";
                                else if (m30 && m60 && (m30.trend === "UP" || m60.trend === "UP")) reason = "Trend UP";

                                if (reason) {{
                                    active.exitTime = time; active.exitPrice = price; active.pnl = active.entryPrice - price; active.reason = reason;
                                    allTrades.push(active); active = null;
                                }}
                            }}
                        }}
                    }}
                }}

                // --- METRICS CALCULATION ---
                allTrades.sort((a,b) => new Date(a.date+' '+a.entryTime) - new Date(b.date+' '+b.entryTime));
                
                let totalPnL = 0, wins = 0, losses = 0, grossWin = 0, grossLoss = 0;
                let equity = 0, peak = 0, maxDD = 0, currentStreak = 0, maxLosingStreak = 0;
                let dailyPnL = {{}};

                allTrades.forEach(t => {{
                    totalPnL += t.pnl;
                    equity += t.pnl;
                    dailyPnL[t.date] = (dailyPnL[t.date] || 0) + t.pnl;
                    
                    if (t.pnl > 0) {{ 
                        wins++; grossWin += t.pnl; 
                        maxLosingStreak = Math.max(maxLosingStreak, currentStreak); 
                        currentStreak = 0; 
                    }} else {{ 
                        losses++; grossLoss += Math.abs(t.pnl); currentStreak++; 
                    }}
                    
                    if (equity > peak) peak = equity;
                    let dd = peak - equity;
                    if (dd > maxDD) maxDD = dd;
                }});
                maxLosingStreak = Math.max(maxLosingStreak, currentStreak);

                const dailyArr = Object.values(dailyPnL);
                const avgDaily = dailyArr.length ? (totalPnL / dailyArr.length) : 0;
                const sharpe = dailyArr.length > 1 ? (avgDaily / std(dailyArr)) : 0;
                const profitFactor = grossLoss > 0 ? (grossWin / grossLoss) : grossWin;
                const winRate = allTrades.length ? (wins / allTrades.length * 100) : 0;

                // --- UI UPDATES ---
                const mHTML = `
                    <div class="metric-item"><div class="metric-label">Total P&L</div><div class="metric-value ${{totalPnL>=0?'profit':'loss'}}">${{totalPnL.toFixed(2)}}</div></div>
                    <div class="metric-item"><div class="metric-label">Win Rate %</div><div class="metric-value">${{winRate.toFixed(1)}}%</div></div>
                    <div class="metric-item"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">${{sharpe.toFixed(2)}}</div></div>
                    <div class="metric-item"><div class="metric-label">Max Drawdown</div><div class="metric-value loss">${{maxDD.toFixed(2)}} (${{(maxDD/cfg.cap*100).toFixed(2)}}%)</div></div>
                    <div class="metric-item"><div class="metric-label">Avg Daily Profit</div><div class="metric-value">${{avgDaily.toFixed(2)}}</div></div>
                    <div class="metric-item"><div class="metric-label">Avg Win/Loss</div><div class="metric-value profit">${{(grossWin/wins || 0).toFixed(2)}}</div><div class="metric-value loss" style="font-size:12px">/ ${{ (grossLoss/losses || 0).toFixed(2)}}</div></div>
                    <div class="metric-item"><div class="metric-label">Profit Factor</div><div class="metric-value">${{profitFactor.toFixed(2)}}</div></div>
                    <div class="metric-item"><div class="metric-label">Max Losing Streak</div><div class="metric-value loss">${{maxLosingStreak}}</div></div>
                    <div class="metric-item"><div class="metric-label">Total / Win / Loss</div><div class="metric-value" style="font-size:14px">${{allTrades.length}} / ${{wins}} / ${{losses}}</div></div>
                `;
                document.getElementById('metricsDashboard').innerHTML = mHTML;

                let logHTML = '<table><thead><tr><th>Date</th><th>Strike</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead><tbody>';
                allTrades.forEach(t => {{
                    logHTML += `<tr><td>${{t.date}}</td><td>${{t.strike}}</td><td>${{t.entryTime}} (@${{t.entryPrice.toFixed(2)}})</td><td>${{t.exitTime}} (@${{t.exitPrice.toFixed(2)}})</td><td class="${{t.pnl>=0?'profit':'loss'}}">${{t.pnl.toFixed(2)}}</td><td>${{t.reason}}</td></tr>`;
                }});
                document.getElementById('tradeLog').innerHTML = logHTML + '</tbody></table>';
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
