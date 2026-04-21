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
DATES_TO_TEST = ["2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10",
                 "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17","2026-04-20","2026-04-21"]
EXPIRY = "26421" 
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- GRID DEFINITIONS ---
SMOOTH_RANGE = [40,45, 50,55, 60, 65,70,75, 80]
ENTRY_SPEEDS = [round(-0.3 - (i * 0.1), 2) for i in range(8)]
EXIT_SPEEDS  = [-0.1, -0.15, -0.2,0]
SL_RANGE     = [10, 8, 6, 5]

ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

# --- UTILS & DATA PREP (Keep your existing prepare_data and get_history here) ---

def generate_html(data):
    json_data = json.dumps(data)
    html_start = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Interactive Strategy Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --surface: #15191e; --border: #252a31; --accent: #00d4ff; --profit: #00ff88; --loss: #ff4d4d; --text: #e1e8ed; --muted: #8899a6; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 20px; margin: 0; }
        .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .tabs { display: flex; gap: 5px; margin-bottom: 1px; overflow-x: auto; }
        .tab { padding: 10px 20px; background: #1a2026; cursor: pointer; border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0; font-size: 11px; white-space: nowrap; }
        .tab.active { background: var(--surface); color: var(--accent); border-top: 2px solid var(--accent); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .btn-run { background: #ff6b35; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        
        /* Histogram Styling */
        .histogram-container { display: flex; align-items: flex-end; gap: 2px; height: 250px; border-bottom: 2px solid var(--border); margin: 50px 0; padding-bottom: 5px; }
        .histo-bar { background: var(--accent); opacity: 0.6; flex: 1; min-width: 15px; cursor: pointer; position: relative; }
        .histo-bar:hover { opacity: 1; background: #fff; }
        .histo-bar.selected { background: #ff6b35; opacity: 1; }
        .histo-label { position: absolute; bottom: -45px; left: 50%; transform: translateX(-50%) rotate(45deg); font-size: 8px; white-space: nowrap; color: var(--muted); }
        
        table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; margin-top: 20px; }
        th { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); color: var(--muted); }
        td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.02); }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div>
            <h1>Strategy Explorer</h1>
            <p style="color:var(--muted); font-size:12px;">Step Filters: ₹10k P&L | 5% Win% | 5pt DD</p>
        </div>
        <button class="btn-run" onclick="runOptimizer()">▶ ANALYZE DATA</button>
    </div>
    <div id="progressBar" style="height:4px; background:#1a2026; margin-bottom:20px; overflow:hidden;"><div id="progressFill" style="height:100%; background:var(--accent); width:0%;"></div></div>
    
    <div class="tabs" id="tabHeaders">
        <div class="tab active" onclick="switchTab('distribution')">HISTOGRAM & DRILL-DOWN</div>
        <div class="tab" onclick="switchTab('overall')">RANKED SETUPS</div>
    </div>

    <div class="panel">
        <div id="distribution" class="tab-content active">
            <h3 id="histTitle">P&L Frequency (Click a bar to see combinations)</h3>
            <div class="histogram-container" id="pnlHistogram"></div>
            
            <div id="drillDownSection" style="display:none;">
                <h3 id="drillTitle" style="color:var(--accent)">Selected Combinations</h3>
                <table id="drillTable">
                    <thead><tr><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>DD%</th></tr></thead>
                    <tbody id="drillBody"></tbody>
                </table>
            </div>
        </div>

        <div id="overall" class="tab-content">
            <table id="overallTable"><thead><tr><th>TIME</th><th>SM%</th><th>E.SPD</th><th>X.SPD</th><th>SL</th><th>P&L</th><th>WIN%</th><th>SCORE</th></tr></thead><tbody id="overallBody"></tbody></table>
        </div>
    </div>

    <script>
"""
    script_data = f"""
        const masterData = {json_data};
        const TIME_VALS = {ENTRY_TIMES};
        const SMOOTH_VALS = {SMOOTH_RANGE};
        const ESPEED_VALS = {ENTRY_SPEEDS};
        const XSPEED_VALS = {EXIT_SPEEDS};
        const SL_VALS = {SL_RANGE};
        const CAPITAL = 1250000;
        const QTY = 325;
    """

    html_end = r"""
        let allResults = [];
        function switchTab(t) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(h => h.classList.remove('active'));
            document.getElementById(t).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        function simulate(entryTime, smooth, eSpeed, xSpeed, sl) {
            let trades = [];
            for (let d in masterData) {
                for (let stk in masterData[d].strikes) {
                    const s = masterData[d].strikes[stk];
                    const slip = Math.abs(s.offset) / 400;
                    let active = null;
                    for (let i=0; i < s.data1m.length; i++) {
                        const time = s.times1m[i];
                        const price = s.data1m[i];
                        if (time < entryTime) continue;
                        const idx5 = s.times5m.indexOf(time);
                        const m30 = idx5 !== -1 ? (function(p, idx, w) {
                            const slice = p.slice(idx - w + 1, idx + 1);
                            const net = slice[slice.length-1] - slice[0];
                            let total = 0; for(let j=1;j<slice.length;j++) total += Math.abs(slice[j]-slice[j-1]);
                            return { sm: (Math.abs(net)/total)*100, sp: net/(w*5) };
                        })(s.data5m, idx5, 6) : null;
                        if (!active) {
                            if (time >= entryTime && time <= "14:45" && m30 && m30.sm >= smooth && m30.sp <= eSpeed) 
                                active = { ent: price - slip, tsl: (price - slip) + sl };
                        } else {
                            if (active.ent - price >= 20) active.tsl = Math.min(active.tsl, active.ent - 10);
                            if (price >= active.tsl || (m30 && m30.sp > xSpeed) || time === "15:25") {
                                trades.push(((active.ent - (price + slip)) * QTY) - 200);
                                active = null;
                            }
                        }
                    }
                }
            }
            return trades;
        }

        async function runOptimizer() {
            allResults = [];
            const total = TIME_VALS.length * SMOOTH_VALS.length * ESPEED_VALS.length * XSPEED_VALS.length * SL_VALS.length;
            let count = 0;

            for (let t of TIME_VALS) {
                for (let sm of SMOOTH_VALS) {
                    for (let es of ESPEED_VALS) {
                        for (let xs of XSPEED_VALS) {
                            for (let sl of SL_VALS) {
                                const trds = simulate(t, sm, es, xs, sl);
                                if (trds.length > 0) {
                                    const pnl = trds.reduce((a,b)=>a+b, 0);
                                    const wr = (trds.filter(x=>x>0).length / trds.length)*100;
                                    let pk=0, cur=0, mdd=0;
                                    trds.forEach(x=>{ cur+=x; pk=Math.max(pk,cur); mdd=Math.max(mdd,pk-cur); });
                                    // Multi-step binning: P&L (10k), Win% (5%), DD (5pts/5k)
                                    const pnlBin = Math.floor(pnl / 10000) * 10000;
                                    const wrBin = Math.floor(wr / 5) * 5;
                                    const ddBin = Math.floor(mdd / 5000) * 5000;
                                    
                                    allResults.push({ t, sm, es, xs, sl, pnl, wr, mdd, score: (pnl/CAPITAL)*20, pnlBin, wrBin, ddBin });
                                }
                                count++;
                                if (count % 5000 === 0) {
                                    document.getElementById('progressFill').style.width = (count/total*100) + "%";
                                    await new Promise(r => setTimeout(r, 0));
                                }
                            }
                        }
                    }
                }
            }
            renderHistogram();
            renderOverall();
        }

        function renderHistogram() {
            const bins = {};
            allResults.forEach(r => {
                bins[r.pnlBin] = (bins[r.pnlBin] || 0) + 1;
            });
            const sortedBins = Object.keys(bins).sort((a,b) => a-b);
            const maxCount = Math.max(...Object.values(bins));
            
            document.getElementById('pnlHistogram').innerHTML = sortedBins.map(b => `
                <div class="histo-bar" style="height:${(bins[b]/maxCount)*100}%" onclick="showDrillDown(${b})">
                    <div class="histo-label">₹${parseInt(b).toLocaleString()}</div>
                </div>
            `).join('');
        }

        function showDrillDown(binValue) {
            const filtered = allResults.filter(r => r.pnlBin === binValue).sort((a,b) => b.pnl - a.pnl);
            document.getElementById('drillDownSection').style.display = 'block';
            document.getElementById('drillTitle').innerText = `Combinations in ₹${binValue.toLocaleString()} bracket (${filtered.length} found)`;
            document.getElementById('drillBody').innerHTML = filtered.map(r => `
                <tr><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td>
                <td style="color:${r.pnl>=0?'var(--profit)':'var(--loss)'}">₹${Math.round(r.pnl).toLocaleString()}</td>
                <td>${r.wr.toFixed(1)}%</td><td>₹${Math.round(r.mdd).toLocaleString()}</td></tr>
            `).join('');
            window.scrollTo({ top: document.getElementById('drillDownSection').offsetTop, behavior: 'smooth' });
        }

        function renderOverall() {
            allResults.sort((a,b) => b.pnl - a.pnl);
            document.getElementById('overallBody').innerHTML = allResults.slice(0, 50).map(r => `
                <tr><td>${r.t}</td><td>${r.sm}</td><td>${r.es}</td><td>${r.xs}</td><td>${r.sl}</td>
                <td>₹${Math.round(r.pnl).toLocaleString()}</td><td>${r.wr.toFixed(1)}%</td><td>${r.score.toFixed(2)}</td></tr>
            `).join('');
        }
    </script>
</body>
</html>
"""
    with open("simulator_optimizer.html", "w") as f:
        f.write(html_start + script_data + html_end)

if __name__ == "__main__":
    # Add your data loading logic here
    generate_html(prepare_data())
