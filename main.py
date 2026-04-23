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

# Mapping Trading Dates to their respective Weekly Expiry
EXPIRY_MAP = {
    "2026-04-07": "26409", # April 9 Expiry
    "2026-04-08": "26409",
    "2026-04-09": "26409",
    "2026-04-13": "26416", # April 16 Expiry
    "2026-04-15": "26416",
    "2026-04-16": "26416",
    "2026-04-20": "26423"  # April 23 Expiry
}

DATES_TO_TEST = list(EXPIRY_MAP.keys())
OFFSETS = [-200, -100, 0, 100, 200] # Narrowed for 5L capital efficiency
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [50, 60, 70, 80]
ENTRY_SPEEDS = [round(-0.4 - (i * 0.1), 2) for i in range(6)]
EXIT_SPEEDS  = [-0.1, -0.15]
SL_RANGE     = [8, 6, 4] # Tighter SL for 2-lot 5L capital

ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

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
        expiry = EXPIRY_MAP[date]
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty: continue
        
        morning = nifty[nifty['time'].dt.strftime("%H:%M") <= "10:15"]
        price_b = morning.iloc[-1]['c'] if not morning.empty else nifty.iloc[0]['o']
        base_atm = int(round(price_b / 50) * 50) # Nifty 50-pt strikes
        
        master[date] = {"strikes": {}}
        for off in OFFSETS:
            strike = base_atm + off
            ce_sym, pe_sym = f"NSE:NIFTY{expiry}{strike}CE", f"NSE:NIFTY{expiry}{strike}PE"
            
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
    json_data = json.dumps(data)
    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>5L Elite Strategy Optimizer</title>
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #f0b90b; --profit: #02c076; --loss: #cf304a; --text: #eaecef; }
        body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .histo-bar { background: var(--accent); opacity: 0.6; flex: 1; cursor: pointer; position: relative; min-width: 30px; }
        .histo-bar:hover { opacity: 1; background: #fff; }
        .btn-run { background: var(--accent); color: #000; border: none; padding: 12px 30px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 15px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid var(--border); }
        .elite { color: var(--profit); font-weight: bold; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>Elite Strategy Optimizer (₹5L | 2 Lots)</h1>
        <button class="btn-run" onclick="runOptimizer()">▶ FIND BEST SETUP</button>
    </div>
    <div id="progress" style="height:4px; background:#333; margin:20px 0;"><div id="fill" style="height:100%; background:var(--accent); width:0%;"></div></div>
    
    <div class="panel">
        <h3>Elite Ranking (Win Rate > 70% + Min DD)</h3>
        <table id="rankTable">
            <thead><tr><th>SCORE</th><th>TIME</th><th>SM%</th><th>SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>SHARPE</th><th>DD (₹)</th></tr></thead>
            <tbody id="rankBody"></tbody>
        </table>
    </div>

    <script>
        const masterData = """ + json_data + r""";
        const TIME_VALS = """ + str(ENTRY_TIMES) + r""";
        const SMOOTH_VALS = """ + str(SMOOTH_RANGE) + r""";
        const ESPEED_VALS = """ + str(ENTRY_SPEEDS) + r""";
        const XSPEED_VALS = """ + str(EXIT_SPEEDS) + r""";
        const SL_VALS = """ + str(SL_RANGE) + r""";
        const CAPITAL = 500000;
        const QTY = 130;

        async function runOptimizer() {
            let results = [];
            const total = TIME_VALS.length * SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;

            for (let t of TIME_VALS) {
                for (let sm of SMOOTH_VALS) {
                    for (let es of ESPEED_VALS) {
                        for (let xs of XSPEED_VALS) {
                            for (let sl of SL_VALS) {
                                let dailyPnL = [];
                                for (let d in masterData) {
                                    let dayTotal = 0;
                                    for (let stk in masterData[d].strikes) {
                                        const s = masterData[d].strikes[stk];
                                        const trades = simulate(s, t, sm, es, xs, sl);
                                        dayTotal += trades.reduce((a,b)=>a+b, 0);
                                    }
                                    dailyPnL.push(dayTotal);
                                }
                                
                                const netPnl = dailyPnL.reduce((a,b)=>a+b, 0);
                                const wr = (dailyPnL.filter(x=>x>0).length / dailyPnL.length)*100;
                                let pk=0, cur=0, mdd=0;
                                dailyPnL.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                
                                const avg = netPnl / dailyPnL.length;
                                const std = Math.sqrt(dailyPnL.map(x => Math.pow(x-avg,2)).reduce((a,b)=>a+b)/dailyPnL.length);
                                const sharpe = std === 0 ? 0 : (avg / std) * Math.sqrt(252);

                                if (wr >= 70 && netPnl > 0) {
                                    // Elite Score: Rewards Profit/DD ratio and penalizes low WinRate
                                    const score = (netPnl / (mdd || 5000)) * (wr/100) * (sharpe > 0 ? sharpe : 0.1);
                                    results.push({ t, sm, es, xs, sl, netPnl, wr, sharpe, mdd, score });
                                }
                                count++;
                                if (count % 4000 === 0) {
                                    document.getElementById('fill').style.width = (count/total*100) + "%";
                                    await new Promise(r => setTimeout(r, 0));
                                }
                            }
                        }
                    }
                }
            }
            render(results);
        }

        function simulate(s, entryTime, smooth, eSpeed, xSpeed, sl) {
            let trades = [];
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
                    if (active.ent - price >= 15) active.tsl = Math.min(active.tsl, active.ent - 5);
                    if (price >= active.tsl || (m30 && m30.sp > xSpeed) || time === "15:25") {
                        trades.push(((active.ent - (price + slip)) * QTY) - 100); // Higher brokerage for small qty
                        active = null;
                    }
                }
            }
            return trades;
        }

        function render(res) {
            res.sort((a,b) => b.score - a.score);
            document.getElementById('rankBody').innerHTML = res.slice(0, 30).map(r => `
                <tr class="${r.wr >= 75 ? 'elite' : ''}">
                    <td>${r.score.toFixed(2)}</td><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.sl}</td>
                    <td>₹${Math.round(r.netPnl).toLocaleString()}</td><td>${r.wr.toFixed(1)}%</td>
                    <td>${r.sharpe.toFixed(2)}</td><td>₹${Math.round(r.mdd).toLocaleString()}</td>
                </tr>`).join('');
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    data = prepare_data()
    if data:
        generate_html(data)
        print("Success: Dashboard created for 5L Capital with Expiry Mapping.")
