import os
import math
import json
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
TARGET_DATE = "2026-04-16"
EXPIRY = "26421"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Output directory: {os.path.abspath(OUTPUT_DIR)}")

BG = "#0b0f1a"
CARD = "#111b27"
BORDER = "#1e2d40"
TEXT = "#e2e8f0"
MUTED = "#64748b"
ACCENT = "#00e5b0"
BLUE = "#38bdf8"
RED = "#ff4560"
STRIKE_COLORS = ["#38bdf8", "#a78bfa", "#f97316", "#ff4560", "#fbbf24", "#00e5b0", "#ec4899", "#84cc16", "#64748b"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --- ANALYTICS ---

def _smoothness(prices):
    if len(prices) < 3: return 0.0
    net = abs(prices[-1] - prices[0])
    total = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
    return round((net / total) * 100, 2) if total else 100.0

def _angle(prices):
    n = len(prices)
    if n < 2: return 0.0
    xm, ym = (n-1)/2, sum(prices)/n
    num = sum((i-xm)*(prices[i]-ym) for i in range(n))
    den = sum((i-xm)**2 for i in range(n))
    return round(math.degrees(math.atan(num/den)), 2) if den else 0.0

def _trend(a):
    return "UP" if a > 5 else ("DOWN" if a < -5 else "FLAT")

def compute_rankings(straddle_data):
    rows = []
    for strike, df in straddle_data.items():
        prices = df["straddle"].tolist()
        sm = _smoothness(prices)
        an = _angle(prices)
        tr = _trend(an)
        rows.append({"strike": strike, "smoothness": sm, "angle": an, "trend": tr})
    ranked = sorted(rows, key=lambda x: x["smoothness"], reverse=True)
    for i, r in enumerate(ranked): r["rank"] = i + 1
    return ranked


# --- DASHBOARD BUILDER ---

def build_dashboard_html(straddle_data, atm, rankings):
    strikes = sorted(straddle_data.keys())

    # --- Build main straddle charts ---
    fig_main = make_subplots(
        rows=len(strikes), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Strike {s}{' ◄ ATM' if s == atm else ''}" for s in strikes]
    )
    for idx, strike in enumerate(strikes):
        df = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        row = idx + 1
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"{strike}",
                                      line=dict(color=color, width=2)), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["vwap"], name="VWAP",
                                      line=dict(color=MUTED, width=1, dash="dot")), row=row, col=1)
    fig_main.update_layout(height=2000, template="plotly_dark",
                           paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)

    # --- Ranking chart ---
    fig_rank = go.Figure(go.Bar(
        y=[str(r["strike"]) for r in rankings],
        x=[r["smoothness"] for r in rankings],
        orientation="h",
        marker_color=[ACCENT if r["rank"] == 1 else "#1e3a2f" for r in rankings],
        text=[f"{r['smoothness']}%" for r in rankings],
        textposition="outside", cliponaxis=False
    ))
    fig_rank.update_layout(
        title="Smoothness Ranking (Full Day)",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=400, margin=dict(l=10, r=40, t=40, b=10),
        yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 110])
    )

    # --- Ranking table ---
    table_rows_html = "".join([
        f"""<tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:10px;">{r['rank']}</td>
            <td style="padding:10px;color:{ACCENT};"><b>{r['strike']}</b></td>
            <td style="padding:10px;">{r['smoothness']}%</td>
            <td style="padding:10px;color:{BLUE if r['angle']>0 else RED};">{r['angle']}°</td>
            <td style="padding:10px;">{r['trend']}</td>
        </tr>""" for r in rankings
    ])

    # --- Build speed data for JS (includes full price series for dynamic smoothness) ---
    speed_data = {}
    for strike, df in straddle_data.items():
        speed_data[str(strike)] = [
            {"time": row["time"].strftime("%H:%M"), "price": round(row["straddle"], 2)}
            for _, row in df.iterrows()
        ]

    # All unique time labels (from ATM strike or first available)
    ref_strike = str(atm) if str(atm) in speed_data else list(speed_data.keys())[0]
    all_times = [d["time"] for d in speed_data[ref_strike]]

    main_chart_html = fig_main.to_html(full_html=False, include_plotlyjs='cdn')
    rank_chart_html = fig_rank.to_html(full_html=False, include_plotlyjs=False)

    speed_data_json = json.dumps(speed_data)
    all_times_json = json.dumps(all_times)
    strikes_json = json.dumps([str(s) for s in strikes])
    strike_colors_json = json.dumps(STRIKE_COLORS)

    
    final_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Straddle Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ background-color: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; }}
            .header {{ padding: 20px 40px; background: {CARD}; border-bottom: 1px solid {BORDER}; display: flex; justify-content: space-between; align-items: center; }}
            .content {{ display: flex; padding: 20px; gap: 20px; }}
            .sidebar {{ flex: 1; min-width: 350px; position: sticky; top: 20px; height: fit-content; }}
            .main-charts {{ flex: 3; }}
            .card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }}
            th {{ color: {MUTED}; text-transform: uppercase; font-size: 9px; padding: 10px; border-bottom: 1px solid {BORDER}; }}
            h2 {{ font-size: 14px; color: {ACCENT}; margin-top: 0; letter-spacing: 1px; }}
    
            .speed-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 24px; margin: 20px; }}
            .speed-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }}
            .speed-controls {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
            .slider-group {{ display: flex; flex-direction: column; gap: 6px; }}
            .slider-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
            .slider-value {{ font-size: 13px; color: {ACCENT}; font-weight: bold; min-width: 50px; text-align: center; }}
            input[type=range] {{
                -webkit-appearance: none;
                width: 300px;
                height: 4px;
                background: {BORDER};
                border-radius: 2px;
                outline: none;
            }}
            input[type=range]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 16px; height: 16px;
                border-radius: 50%;
                background: {ACCENT};
                cursor: pointer;
                box-shadow: 0 0 6px {ACCENT}88;
            }}
            .speed-table-wrap {{ overflow-x: auto; }}
            #speedTable th {{ border-bottom: 2px solid {BORDER}; }}
            #speedTable .win-header {{ background: {BORDER}33; text-align: center; font-weight: bold; color: {BLUE}; }}
            #speedTable td {{ padding: 8px 12px; border-bottom: 1px solid {BORDER}22; }}
            .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
            .badge-up {{ background: #00e5b022; color: {ACCENT}; }}
            .badge-down {{ background: #ff456022; color: {RED}; }}
            .badge-flat {{ background: #64748b22; color: {MUTED}; }}
            .strike-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div><span style="color:{ACCENT};font-size:24px;">▣</span> <b style="font-size:20px;letter-spacing:2px;">STRADDLE ANALYSER</b></div>
            <div style="color:{MUTED};">DATE: {TARGET_DATE} | EXPIRY: {EXPIRY}</div>
        </div>
    
        <div class="content">
            <div class="sidebar">
                <div class="card">
                    <h2>SMOOTHNESS RANKING</h2>
                    {rank_chart_html}
                </div>
                <div class="card">
                    <h2>STATISTICS TABLE</h2>
                    <table>
                        <thead><tr><th>Rank</th><th>Strike</th><th>Smooth</th><th>Angle</th><th>Trend</th></tr></thead>
                        <tbody>{table_rows_html}</tbody>
                    </table>
                </div>
            </div>
            <div class="main-charts">
                <div class="card">
                    <h2>HISTORICAL STRADDLE + VWAP</h2>
                    {main_chart_html}
                </div>
            </div>
        </div>
    
        <div class="speed-card">
            <div class="speed-header">
                <div>
                    <h2 style="margin:0;">⚡ MULTI-WINDOW MOMENTUM</h2>
                    <div style="color:{MUTED};font-size:12px;margin-top:4px;">Comparison of 30min vs 60min velocity</div>
                </div>
                <div class="speed-controls">
                    <div class="slider-group">
                        <div class="slider-label">Reference Time (End of Window)</div>
                        <input type="range" id="timeSlider" min="0" max="1" value="0" step="1">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span class="slider-value" id="timeVal">--:--</span>
                            <span style="font-size:12px;color:{MUTED};" id="timeDisplay"></span>
                        </div>
                    </div>
                </div>
            </div>
    
            <div class="speed-table-wrap">
                <table id="speedTable">
                    <thead>
                        <tr>
                            <th rowspan="2">Strike</th>
                            <th colspan="4" class="win-header" style="color:{ACCENT};">30 MINUTE WINDOW</th>
                            <th colspan="4" class="win-header" style="color:{BLUE};">60 MINUTE WINDOW</th>
                        </tr>
                        <tr>
                            <th>Dur</th><th>Speed</th><th>Smooth</th><th>Direction</th>
                            <th>Dur</th><th>Speed</th><th>Smooth</th><th>Direction</th>
                        </tr>
                    </thead>
                    <tbody id="speedTableBody"></tbody>
                </table>
            </div>
        </div>
    
        <script>
        const speedData = {speed_data_json};
        const allTimes = {all_times_json};
        const strikes  = {strikes_json};
        const colors   = {strike_colors_json};
        const ATM      = "{atm}";
    
        const timeSlider = document.getElementById('timeSlider');
        const timeVal    = document.getElementById('timeVal');
        const timeDisplay = document.getElementById('timeDisplay');
        const tbody      = document.getElementById('speedTableBody');
    
        timeSlider.max = allTimes.length - 1;
        timeSlider.value = allTimes.length - 1;
    
        function calcStats(strike, tIdx, durationCandles) {{
            const series = speedData[strike];
            const fromIdx = Math.max(0, tIdx - durationCandles);
            const fromPt = series[fromIdx];
            const toPt = series[tIdx];
            
            if (!fromPt || !toPt) return null;
    
            const mins = (tIdx - fromIdx) * 1;
            const delta = toPt.price - fromPt.price;
            const speed = mins > 0 ? (delta / mins) : 0;
            
            // Smoothness calculation
            const slice = series.slice(fromIdx, tIdx + 1).map(d => d.price);
            let smooth = 100.0;
            if (slice.length >= 3) {{
                const net = Math.abs(slice[slice.length - 1] - slice[0]);
                let total = 0;
                for (let i = 1; i < slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
                smooth = total > 0 ? (net / total) * 100 : 100.0;
            }}
    
            const dir = delta > 2 ? 'UP' : delta < -2 ? 'DOWN' : 'FLAT';
            const badge = dir === 'UP' ? '<span class="badge badge-up">▲ UP</span>' : 
                          dir === 'DOWN' ? '<span class="badge badge-down">▼ DOWN</span>' : 
                          '<span class="badge badge-flat">— FLAT</span>';
    
            return {{
                dur: mins + 'm',
                speed: (speed >= 0 ? '+' : '') + speed.toFixed(2),
                smooth: smooth.toFixed(1) + '%',
                badge: badge,
                speedColor: Math.abs(speed) > 5 ? (delta > 0 ? '{ACCENT}' : '{RED}') : '{TEXT}',
                smoothColor: smooth > 70 ? '{ACCENT}' : smooth > 40 ? '{BLUE}' : '{RED}'
            }};
        }}
    
        function updateTable() {{
            const tIdx = parseInt(timeSlider.value);
            const toTime = allTimes[tIdx];
            timeVal.textContent = toTime;
            timeDisplay.textContent = `Analysis up to ${{toTime}}`;
    
            const rows = strikes.map((strike, idx) => {{
                const s30 = calcStats(strike, tIdx, 30); // 30 * 1m = 30m
                const s60 = calcStats(strike, tIdx, 60); // 60 * 1m = 60m
                const color = colors[idx % colors.length];
                const atmMark = strike === ATM ? ' <small style="color:{ACCENT}">ATM</small>' : '';
    
                if (!s30 || !s60) return '';
    
                return `<tr>
                    <td style="font-weight:bold; border-right: 1px solid {BORDER}44;">
                        <span class="strike-dot" style="background:${{color}};"></span>${{strike}}${{atmMark}}
                    </td>
                    <td style="color:{MUTED}">${{s30.dur}}</td>
                    <td style="color:${{s30.speedColor}}; font-weight:bold;">${{s30.speed}}</td>
                    <td style="color:${{s30.smoothColor}};">${{s30.smooth}}</td>
                    <td style="border-right: 1px solid {BORDER}44;">${{s30.badge}}</td>
                    <td style="color:{MUTED}">${{s60.dur}}</td>
                    <td style="color:${{s60.speedColor}}; font-weight:bold;">${{s60.speed}}</td>
                    <td style="color:${{s60.smoothColor}};">${{s60.smooth}}</td>
                    <td>${{s60.badge}}</td>
                </tr>`;
            }});
            tbody.innerHTML = rows.join('');
        }}
    
        timeSlider.addEventListener('input', updateTable);
        updateTable();
        </script>
    </body>
    </html>"""

    ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
    ist_time_str = ist_now.strftime("%H%M%S")
    filename = f"{TARGET_DATE}_{ist_time_str}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_html)

    logger.info(f"Dashboard saved as: {filepath}")
    return filepath


# --- DATA FETCHING ---

def fetch_candles(fyers, symbol, date):
    data = {
        "symbol": symbol, "resolution": "1", "date_format": "1",
        "range_from": date, "range_to": date, "cont_flag": "1"
    }
    resp = fyers.history(data=data)
    logger.info(f"  Response for {symbol}: {resp.get('s')} | message: {resp.get('message', '-')}")
    if resp.get("s") == "ok" and resp.get("candles"):
        df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s", utc=True)
                      .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
        return df
    return pd.DataFrame()


# --- MAIN ---

def main():
    if not CLIENT_ID:
        logger.error("CLIENT_ID not set. Aborting."); return
    if not TOKEN:
        logger.error("FYERS_ACCESS_TOKEN not set. Aborting."); return

    logger.info(f"CLIENT_ID  : {CLIENT_ID}")
    logger.info(f"TOKEN      : {TOKEN[:10]}...{TOKEN[-5:]}")
    logger.info(f"TARGET_DATE: {TARGET_DATE}")
    logger.info(f"EXPIRY     : {EXPIRY}")

    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")
    profile = fyers.get_profile()
    logger.info(f"Profile check: {profile.get('s')} | {profile.get('message', '-')}")
    if profile.get("s") != "ok":
        logger.error("Fyers auth failed. Token may be expired."); return

    atm = 24400
    results = {}

    for offset in OFFSETS:
        strike = atm + offset
        logger.info(f"Fetching strike {strike}...")
        ce_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}CE", TARGET_DATE)
        pe_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}PE", TARGET_DATE)
        if not ce_df.empty and not pe_df.empty:
            merged = pd.merge(ce_df[['time','close','volume']], pe_df[['time','close','volume']], on='time')
            merged['straddle'] = merged['close_x'] + merged['close_y']
            merged['v']        = merged['volume_x'] + merged['volume_y']
            merged['vwap']     = (merged['straddle'] * merged['v']).cumsum() / merged['v'].cumsum()
            results[strike] = merged
            logger.info(f"  Strike {strike}: {len(merged)} rows OK")
        else:
            logger.warning(f"  Strike {strike}: No data (CE={ce_df.empty}, PE={pe_df.empty})")

    if results:
        rankings = compute_rankings(results)
        filepath = build_dashboard_html(results, atm, rankings)
        logger.info(f"Dashboard created: {filepath}")
    else:
        logger.error("No data retrieved. Check TOKEN, EXPIRY, TARGET_DATE.")

if __name__ == "__main__":
    main()
