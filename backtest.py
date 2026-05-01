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

# Dynamic Expiry Mapping
EXPIRY_MAP = {
    "2026-04-07": "26421", "2026-04-08": "26421", "2026-04-09": "26421",
    "2026-04-13": "26421", "2026-04-15": "26421", "2026-04-16": "26421",
    "2026-04-20": "26421", "2026-04-21": "26421", "2026-04-22": "26APR",
    "2026-04-27": "26505", "2026-04-28": "26505", "2026-04-29": "26505", 
    "2026-04-30": "26505"
}
DATES_TO_TEST = sorted(list(EXPIRY_MAP.keys()))
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
CHECK_HOURS = ["10:00", "11:00", "12:00", "13:00", "14:00"]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_history(symbol, date, res):
    """Checks local cache for NSE_SYMBOL_RES_DATE.csv then API."""
    clean_sym = symbol.replace(":", "_")
    filename = f"{clean_sym}_{res}_{date}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df

    time.sleep(0.7) 
    print(f"Fetching {symbol} ({res}) for {date}...")
    data = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    
    if resp.get("s") == "ok" and resp.get("candles"):
        df = pd.DataFrame(resp["candles"], columns=["epoch","o","h","l","c","v"])
        df["time"] = pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
        # Calculate VWAP
        df['vwap'] = (df['c'] * df['v']).cumsum() / df['v'].cumsum()
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_simulator_data():
    master_data = {}
    for date in DATES_TO_TEST:
        expiry = EXPIRY_MAP[date]
        master_data[date] = {"strikes": {}}
        
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        # Calculate ATMs for the specific check windows
        for t_str in CHECK_HOURS:
            n_row = nifty[nifty['time'].dt.strftime("%H:%M") == t_str]
            if n_row.empty: continue
            base_atm = int(round(n_row.iloc[0]['c'] / 100) * 100)

            for offset in OFFSETS:
                strike = base_atm + offset
                strike_key = f"{expiry}_{strike}"
                if strike_key in master_data[date]["strikes"]: continue

                # Try 1 min first
                df_ce = get_history(f"NSE:NIFTY{expiry}{strike}CE", date, "1")
                df_pe = get_history(f"NSE:NIFTY{expiry}{strike}PE", date, "1")
                
                # Fallback to 5 min if 1 min is missing
                if df_ce.empty or df_pe.empty:
                    print(f"1m missing for {strike}, trying 5m fallback...")
                    df_ce = get_history(f"NSE:NIFTY{expiry}{strike}CE", date, "5")
                    df_pe = get_history(f"NSE:NIFTY{expiry}{strike}PE", date, "5")

                if not (df_ce.empty or df_pe.empty):
                    # Align timestamps
                    m = pd.merge(df_ce[['time','c','vwap']], df_pe[['time','c','vwap']], on='time', how='inner')
                    master_data[date]["strikes"][strike_key] = {
                        "price": (m['c_x'] + m['c_y']).tolist(),
                        "vwap": (m['vwap_x'] + m['vwap_y']).tolist(),
                        "times": m['time'].dt.strftime("%H:%M").tolist()
                    }
    return master_data

def generate_interactive_html(data):
    json_data = json.dumps(data)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Straddle Dashboard - VWAP Ratio</title>
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, sans-serif; padding: 20px; }}
            .nav {{ display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid #30363d; padding-bottom: 15px; }}
            .tab {{ padding: 8px 16px; background: #161b22; border: 1px solid #30363d; cursor: pointer; border-radius: 6px; }}
            .tab.active {{ background: #238636; border-color: #2ea043; color: white; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-val {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
            .profit {{ color: #3fb950; }} .loss {{ color: #f85149; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
            th {{ background: #21262d; }}
        </style>
    </head>
    <body>
        <h2>📈 Strategy Metrics & Entry Time Analysis</h2>
        <div class="nav">
            <div class="tab active" onclick="updateView('ALL')">ALL</div>
            <div class="tab" onclick="updateView('10:00')">10:00 AM</div>
            <div class="tab" onclick="updateView('11:00')">11:00 AM</div>
            <div class="tab" onclick="updateView('12:00')">12:00 PM</div>
            <div class="tab" onclick="updateView('13:00')">01:00 PM</div>
            <div class="tab" onclick="updateView('14:00')">02:00 PM</div>
        </div>

        <div id="metrics" class="grid"></div>
        
        <div class="card" style="text-align: left;">
            <h3>Trade Log</h3>
            <div id="log"></div>
        </div>

        <script>
            const rawData = {json_data};
            let currentTab = 'ALL';

            function runEngine() {{
                const results = [];
                for (let date in rawData) {{
                    const checkTimes = ["10:00", "11:00", "12:00", "13:00", "14:00"];
                    checkTimes.forEach(tHour => {{
                        let below = 0, above = 0, list = [];
                        
                        for (let sId in rawData[date].strikes) {{
                            const s = rawData[date].strikes[sId];
                            const i = s.times.indexOf(tHour);
                            if (i === -1) continue;

                            if (s.price[i] < s.vwap[i]) below++; else above++;
                            list.push({{ id: sId, price: s.price[i], vwap: s.vwap[i], data: s, i }});
                        }}

                        if (list.length > 0) {{
                            // Logic: Entry in strike furthest below VWAP
                            list.sort((a,b) => (a.price - a.vwap) - (b.price - b.vwap));
                            const pick = list[0];
                            
                            // Exit at EOD or SL
                            let exitP = pick.data.price[pick.data.price.length-1];
                            let pnl = pick.price - exitP;

                            results.push({{
                                date, hour: tHour, strike: pick.id,
                                ratio: (below / (above || 1)).toFixed(2),
                                pnl: pnl,
                                wins: pnl > 0
                            }});
                        }}
                    }});
                }}
                return results;
            }}

            function updateView(tab) {{
                currentTab = tab;
                const all = runEngine();
                const filtered = tab === 'ALL' ? all : all.filter(x => x.hour === tab);

                let totalPnl = 0, wins = 0;
                filtered.forEach(x => {{ totalPnl += x.pnl; if(x.wins) wins++; }});

                document.getElementById('metrics').innerHTML = `
                    <div class="card"><div>Total P&L</div><div class="metric-val ${{totalPnl>=0?'profit':'loss'}}">${{totalPnl.toFixed(2)}}</div></div>
                    <div class="card"><div>Win Rate</div><div class="metric-val">${{((wins/filtered.length)*100 || 0).toFixed(1)}}%</div></div>
                    <div class="card"><div>Total Trades</div><div class="metric-val">${{filtered.length}}</div></div>
                `;

                let table = '<table><tr><th>Date</th><th>Time</th><th>Strike</th><th>Ratio (B/A)</th><th>PnL</th></tr>';
                filtered.forEach(x => {{
                    table += `<tr><td>${{x.date}}</td><td>${{x.hour}}</td><td>${{x.strike}}</td><td>${{x.ratio}}</td><td class="${{x.pnl>=0?'profit':'loss'}}">${{x.pnl.toFixed(2)}}</td></tr>`;
                }});
                document.getElementById('log').innerHTML = table + '</table>';

                document.querySelectorAll('.tab').forEach(t => {{
                    t.classList.toggle('active', t.innerText === tab || (tab === 'ALL' && t.innerText === 'ALL'));
                }});
            }}

            window.onload = () => updateView('ALL');
        </script>
    </body>
    </html>
    """
    with open("simulator_final.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    print("Initializing data preparation...")
    sim_data = prepare_simulator_data()
    generate_interactive_html(sim_data)
    print("Complete. Open 'simulator_final.html' to analyze results.")
