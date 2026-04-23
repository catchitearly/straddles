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
OFFSETS = [-100, 0, 100] # Precision focus for 5L
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70, 80]
ENTRY_SPEEDS = [round(-0.4 - (i * 0.1), 2) for i in range(5)]
EXIT_SPEEDS  = [-0.1, -0.15,0]
SL_RANGE     = [8, 6, 4]

ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

# --- UTILS (prepare_data and get_history remain same as previous version) ---

def generate_html(data):
    json_data = json.dumps(data)
    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Elite 5L Optimizer v4.0</title>
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #f0b90b; --profit: #02c076; --loss: #cf304a; --text: #eaecef; --muted: #848e9c; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 25px; background: #1e2329; cursor: pointer; border-radius: 4px; font-size: 13px; font-weight: bold; border: 1px solid transparent; }
        .tab.active { background: #2b3139; border-color: var(--accent); color: var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Scatter Plot Styles */
        .chart-area { height: 400px; border-left: 2px solid var(--border); border-bottom: 2px solid var(--border); margin: 40px 0; position: relative; display: flex; align-items: flex-end; }
        .dot { position: absolute; width: 8px; height: 8px; background: var(--accent); border-radius: 50%; cursor: pointer; opacity: 0.6; transition: transform 0.2s; }
        .dot:hover { transform: scale(2); opacity: 1; border: 2px solid #fff; }
        .axis-label { position: absolute; color: var(--muted); font-size: 11px; }

        input[type=range] { width: 100%; margin: 20px 0; accent-color: var(--accent); }
        table { width: 100%; border-collapse: collapse; font-size: 11px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); }
        .btn-run { background: var(--accent); color: #000; border: none; padding: 12px 30px; border-radius: 4px; cursor: pointer; font-weight: bold; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1 style="margin:0;">Elite Strategy Engine v4.0</h1>
            <p style="color:var(--muted); font-size:12px;">₹5L Capital | 2 Lots | Friction-Adjusted Scoring</p>
        </div>
        <button class="btn-run" onclick="runOptimizer()">▶ START BACKTEST</button>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="switchTab('ranking')">TOP SETUPS</div>
        <div class="tab" onclick="switchTab('surface')">RISK/RETURN SURFACE</div>
    </div>

    <div class="panel tab-content active" id="ranking">
        <table id="rankTable">
            <thead><tr><th>SCORE</th><th>TIME</th><th>SM%</th><th>SPD</th><th>SL</th><th>NET P&L</th><th>TRADES</th><th>WIN%</th><th>DD (₹)</th></tr></thead>
            <tbody id="rankBody"></tbody>
        </table>
    </div>

    <div class="panel tab-content" id="surface">
        <div style="display:flex; justify-content:space-between;">
            <label>Filter by Max Drawdown: <span id="ddVal" style="color:var(--accent); font-weight:bold;">₹50,000</span></label>
        </div>
        <input type="range" id="ddSlider" min="5000" max="100000" step="5000" value="50000" oninput="updateSurface()">
        
        <div class="chart-area" id="surfaceChart">
            <div class="axis-label" style="bottom: -25px; left: 50%;">Total Number of Trades</div>
            <div class="axis-label" style="left: -40px; top: 50%; transform: rotate(-90deg);">Return on Capital (%)</div>
        </div>
        
        <div id="surfaceDrill" style="margin-top:20px; display:none; border-top:1px solid var(--border); padding-top:10px;">
            <h4 style="color:var(--accent)">Combination Details</h4>
            <table id="drillTable"><thead><tr><th>TIME</th><th>SM%</th><th>SPD</th><th>SL</th><th>ROC%</th><th>TRADES</th></tr></thead><tbody id="drillBody"></tbody></table>
        </div>
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
        const TAX_PER_TRADE = 200;

        let allResults = [];

        function switchTab(t) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(h => h.classList.remove('active'));
            document.getElementById(t).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        async function runOptimizer() {
            allResults = [];
            for (let t of TIME_VALS) {
                for (let sm of SMOOTH_VALS) {
                    for (let es of ESPEED_VALS) {
                        for (let xs of XSPEED_VALS) {
                            for (let sl of SL_VALS) {
                                let trades = [];
                                for (let d in masterData) {
                                    for (let stk in masterData[d].strikes) {
                                        trades.push(...simulate(masterData[d].strikes[stk], t, sm, es, xs, sl));
                                    }
                                }
                                if (trades.length > 5) {
                                    const totalTrades = trades.length;
                                    const grossPnl = trades.reduce((a,b)=>a+b, 0);
                                    const friction = totalTrades * TAX_PER_TRADE;
                                    const netPnl = grossPnl - friction;
                                    const wr = (trades.filter(x=>x>0).length / totalTrades)*100;
                                    let pk=0, cur=0, mdd=0;
                                    trades.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                    
                                    const score = (netPnl / (mdd || 5000)) * (wr/100) * (1 - (friction/Math.abs(grossPnl)));
                                    allResults.push({ t, sm, es, xs, sl, netPnl, wr, mdd, totalTrades, score, roc: (netPnl/CAPITAL)*100 });
                                }
                            }
                        }
                    }
                }
            }
            renderRanking();
            updateSurface();
        }

        function simulate(s, entryTime, smooth, eSpeed, xSpeed, sl) {
            let trds = []; const slip = Math.abs(s.offset) / 400; let active = null;
            for (let i=0; i < s.data1m.length; i++) {
                const time = s.times1m[i]; const price = s.data1m[i];
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
                        trds.push(((active.ent - (price + slip)) * QTY));
                        active = null;
                    }
                }
            }
            return trds;
        }

        function renderRanking() {
            allResults.sort((a,b) => b.score - a.score);
            document.getElementById('rankBody').innerHTML = allResults.slice(0, 30).map(r => `
                <tr><td>${r.score.toFixed(2)}</td><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.sl}</td>
                <td style="color:${r.netPnl>0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.netPnl).toLocaleString()}</td>
                <td>${r.totalTrades}</td><td>${r.wr.toFixed(1)}%</td><td>₹${Math.round(r.mdd).toLocaleString()}</td></tr>`).join('');
        }

        function updateSurface() {
            const maxDD = document.getElementById('ddSlider').value;
            document.getElementById('ddVal').innerText = '₹' + parseInt(maxDD).toLocaleString();
            const filtered = allResults.filter(r => r.mdd <= maxDD);
            const chart = document.getElementById('surfaceChart');
            chart.innerHTML = '<div class="axis-label" style="bottom: -25px; left: 50%;">Total Number of Trades</div><div class="axis-label" style="left: -40px; top: 50%; transform: rotate(-90deg);">Return on Capital (%)</div>';
            
            const maxTrades = Math.max(...allResults.map(r => r.totalTrades));
            const maxROC = Math.max(...allResults.map(r => r.roc));

            filtered.forEach(r => {
                const dot = document.createElement('div');
                dot.className = 'dot';
                dot.style.left = (r.totalTrades / maxTrades * 95) + '%';
                dot.style.bottom = (r.roc / maxROC * 95) + '%';
                dot.onclick = () => {
                    document.getElementById('surfaceDrill').style.display = 'block';
                    document.getElementById('drillBody').innerHTML = `<tr><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.sl}</td><td>${r.roc.toFixed(2)}%</td><td>${r.totalTrades}</td></tr>`;
                };
                chart.appendChild(dot);
            });
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    generate_html(prepare_data())
