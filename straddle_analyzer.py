import os
import math
import json
import logging
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel
import sys

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TARGET_DATE = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
EXPIRY = "26623"  # Update as needed
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

# Dashboard Styling
BG = "#0b0f1a"
CARD = "#111b27"
BORDER = "#1e2d40"
TEXT = "#e2e8f0"
MUTED = "#64748b"
ACCENT = "#00e5b0"
BLUE = "#38bdf8"
RED = "#ff4560"
EMA_COLOR = "#fbbf24"  # Yellow/amber for EMA9 line
STRIKE_COLORS = ["#38bdf8", "#a78bfa", "#f97316", "#ff4560", "#fbbf24", "#00e5b0", "#ec4899", "#84cc16", "#64748b"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUN_TIMESTAMP = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

# --- TELEGRAM ---

def send_telegram_message(text):
    """Send a text message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping message.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("✓ Telegram message sent.")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_document(file_path, caption=""):
    """Send an HTML file as a document via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping document.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                files={"document": (os.path.basename(file_path), f, "text/html")},
                timeout=30
            )
        resp.raise_for_status()
        logger.info(f"✓ Telegram document sent: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram document: {e}")
        return False

def send_telegram_summary(rankings, atm, successful_fetches, total_strikes):
    """Send a formatted summary of the analysis to Telegram."""
    trend_emoji = {"UP": "📈", "DOWN": "📉", "FLAT": "➡️"}
    medal = ["🥇", "🥈", "🥉"]

    lines = [
        f"<b>📊 STRADDLE ANALYSER — {TARGET_DATE}</b>",
        f"<code>ATM: {atm} | Strikes: {successful_fetches}/{total_strikes} | {RUN_TIMESTAMP}</code>",
        "",
        "<b>🏆 SMOOTHNESS RANKINGS</b>",
    ]

    for r in rankings:
        m = medal[r["rank"] - 1] if r["rank"] <= 3 else f"#{r['rank']}"
        atm_tag = " ◄ ATM" if r["strike"] == atm else ""
        emoji = trend_emoji.get(r["trend"], "➡️")
        lines.append(
            f"{m} <b>{r['strike']}</b>{atm_tag}  |  "
            f"Smooth: <code>{r['smoothness']}%</code>  |  "
            f"Angle: <code>{r['angle']}°</code>  |  {emoji} {r['trend']}"
        )

    # Highlight top pick
    top = rankings[0]
    lines += [
        "",
        f"✅ <b>Best Strike:</b> {top['strike']} (Smoothness: {top['smoothness']}%)",
        f"📎 Full dashboard attached below."
    ]

    send_telegram_message("\n".join(lines))

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

def compute_ema(prices, period=9):
    """Compute EMA for a list/series of prices. Returns a list of floats (NaN for initial values)."""
    ema = pd.Series(prices).ewm(span=period, adjust=False).mean()
    return ema.tolist()

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

        # Straddle price line
        fig_main.add_trace(go.Scatter(
            x=df["time"], y=df["straddle"], name=f"{strike}",
            line=dict(color=color, width=2)
        ), row=row, col=1)

        # VWAP line
        fig_main.add_trace(go.Scatter(
            x=df["time"], y=df["vwap"], name="VWAP",
            line=dict(color=MUTED, width=1, dash="dot")
        ), row=row, col=1)

        # EMA9 line
        fig_main.add_trace(go.Scatter(
            x=df["time"], y=df["ema9"], name="EMA9",
            line=dict(color=EMA_COLOR, width=1.5, dash="dash")
        ), row=row, col=1)

    fig_main.update_layout(height=200 * len(strikes), template="plotly_dark",
                            paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)

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

    table_rows_html = "".join([
        f"""<tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:10px;">{r['rank']}</td>
            <td style="padding:10px;color:{ACCENT};"><b>{r['strike']}</b></td>
            <td style="padding:10px;">{r['smoothness']}%</td>
            <td style="padding:10px;color:{BLUE if r['angle']>0 else RED};">{r['angle']}°</td>
            <td style="padding:10px;">{r['trend']}</td>
        </tr>""" for r in rankings
    ])

    speed_data = {str(strike): [{"time": row["time"].strftime("%H:%M"), "price": round(row["straddle"], 2)}
                  for _, row in df.iterrows()] for strike, df in straddle_data.items()}
    ref_strike = list(speed_data.keys())[0]
    all_times = [d["time"] for d in speed_data[ref_strike]]

    final_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="300">
    <title>Straddle Dashboard - {TARGET_DATE}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ background-color: {BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; }}
        .header {{ padding: 20px 40px; background: {CARD}; border-bottom: 1px solid {BORDER}; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
        .timestamp {{ color: {MUTED}; font-size: 12px; background: {BG}; padding: 5px 10px; border-radius: 4px; }}
        .content {{ display: flex; padding: 20px; gap: 20px; flex-wrap: wrap; }}
        .sidebar {{ flex: 1; min-width: 350px; }}
        .main-charts {{ flex: 3; min-width: 500px; }}
        .card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }}
        th {{ color: {MUTED}; text-transform: uppercase; font-size: 9px; padding: 10px; border-bottom: 1px solid {BORDER}; }}
        h2 {{ font-size: 14px; color: {ACCENT}; margin-top: 0; letter-spacing: 1px; }}
        .legend-item {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: 12px; color: {MUTED}; }}
        .legend-line {{ width: 24px; height: 2px; display: inline-block; }}
        .legend-dashed {{ background: repeating-linear-gradient(to right, {MUTED} 0, {MUTED} 4px, transparent 4px, transparent 8px); }}
        .legend-ema {{ background: repeating-linear-gradient(to right, {EMA_COLOR} 0, {EMA_COLOR} 4px, transparent 4px, transparent 8px); }}
        .speed-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 24px; margin: 20px; }}
        .speed-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }}
        .speed-controls {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
        .slider-group {{ display: flex; flex-direction: column; gap: 6px; }}
        .slider-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
        .slider-value {{ font-size: 13px; color: {ACCENT}; font-weight: bold; min-width: 50px; text-align: center; }}
        input[type=range] {{ -webkit-appearance: none; width: 300px; height: 4px; background: {BORDER}; border-radius: 2px; outline: none; }}
        input[type=range]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: {ACCENT}; cursor: pointer; }}
        .speed-table-wrap {{ overflow-x: auto; }}
        #speedTable th {{ border-bottom: 2px solid {BORDER}; }}
        #speedTable .win-header {{ background: {BORDER}33; text-align: center; font-weight: bold; }}
        #speedTable td {{ padding: 8px 12px; border-bottom: 1px solid {BORDER}22; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
        .badge-up {{ background: #00e5b022; color: {ACCENT}; }}
        .badge-down {{ background: #ff456022; color: {RED}; }}
        .badge-flat {{ background: #64748b22; color: {MUTED}; }}
        .strike-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
        @media (max-width: 768px) {{
            .content {{ flex-direction: column; }}
            .sidebar, .main-charts {{ min-width: auto; }}
            .speed-controls {{ flex-direction: column; align-items: flex-start; }}
            input[type=range] {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div><span style="color:{ACCENT};font-size:24px;">▣</span> <b style="font-size:20px;letter-spacing:2px;">STRADDLE ANALYSER</b></div>
        <div class="timestamp">Last Update: {RUN_TIMESTAMP}<br>Auto-refresh: Every 5 min</div>
    </div>
    <div class="content">
        <div class="sidebar">
            <div class="card">
                <h2>SMOOTHNESS RANKING</h2>
                {fig_rank.to_html(full_html=False, include_plotlyjs='cdn')}
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
                <h2>HISTORICAL STRADDLE + VWAP + EMA9</h2>
                <div style="margin-bottom:12px;">
                    <span class="legend-item"><span class="legend-line" style="background:{ACCENT};height:2px;"></span> Straddle</span>
                    <span class="legend-item"><span class="legend-line legend-dashed"></span> VWAP</span>
                    <span class="legend-item"><span class="legend-line legend-ema"></span> EMA 9</span>
                </div>
                {fig_main.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>
    </div>
    <div class="speed-card">
        <div class="speed-header">
            <div>
                <h2 style="margin:0;">⚡ MULTI-WINDOW MOMENTUM</h2>
                <div style="color:{MUTED};font-size:12px;margin-top:4px;">Comparing 30-min vs 60-min volatility speed</div>
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
    const speedData = {json.dumps(speed_data)};
    const allTimes = {json.dumps(all_times)};
    const strikes  = {json.dumps([str(s) for s in strikes])};
    const colors   = {json.dumps(STRIKE_COLORS)};
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
        const mins = (tIdx - fromIdx) * 5;
        const delta = toPt.price - fromPt.price;
        const speed = mins > 0 ? (delta / mins) : 0;
        const slice = series.slice(fromIdx, tIdx + 1).map(d => d.price);
        let net = Math.abs(slice[slice.length - 1] - slice[0]);
        let total = 0;
        for (let i = 1; i < slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
        let smooth = total > 0 ? (net / total) * 100 : 100.0;
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
        timeVal.textContent = allTimes[tIdx];
        timeDisplay.textContent = `Analysis Window End: ${{allTimes[tIdx]}}`;
        const rows = strikes.map((strike, idx) => {{
            const s30 = calcStats(strike, tIdx, 6);
            const s60 = calcStats(strike, tIdx, 12);
            if (!s30 || !s60) return '';
            const atmMark = strike === ATM ? ' <small style="color:{ACCENT}">ATM</small>' : '';
            return `<tr>
                <td style="font-weight:bold; border-right: 1px solid {BORDER}44;">
                    <span class="strike-dot" style="background:${{colors[idx % colors.length]}};"></span>${{strike}}${{atmMark}}
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
    setTimeout(function() {{ location.reload(); }}, 300000);
    </script>
</body>
</html>"""

    docs_path = "docs/index.html"
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(OUTPUT_DIR, f"dashboard_{timestamp}.html")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return docs_path, backup_path

# --- DATA FETCHING ---

def fetch_candles(fyers, symbol, date):
    data = {"symbol": symbol, "resolution": "5", "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}
    try:
        resp = fyers.history(data=data)
        if resp.get("s") == "ok" and resp.get("candles"):
            df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
    return pd.DataFrame()

# --- MAIN ---

def main():
    if not CLIENT_ID or not TOKEN:
        logger.error("API Credentials missing. Please set CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.")
        return

    try:
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

        profile = fyers.get_profile()
        if profile.get("s") != "ok":
            logger.error(f"Authentication failed: {profile}")
            return

        logger.info(f"Successfully authenticated. Running analysis for {TARGET_DATE}")

        atm = 24100  # Update this dynamically if needed

        results = {}
        successful_fetches = 0

        for offset in OFFSETS:
            strike = atm + offset
            logger.info(f"Fetching data for strike {strike}")

            ce_symbol = f"NSE:NIFTY{EXPIRY}{strike}CE"
            pe_symbol = f"NSE:NIFTY{EXPIRY}{strike}PE"

            ce_df = fetch_candles(fyers, ce_symbol, TARGET_DATE)
            pe_df = fetch_candles(fyers, pe_symbol, TARGET_DATE)

            if not ce_df.empty and not pe_df.empty:
                merged = pd.merge(ce_df[['time','close','volume']], pe_df[['time','close','volume']], on='time')
                merged['straddle'] = merged['close_x'] + merged['close_y']
                merged['v'] = merged['volume_x'] + merged['volume_y']
                merged['vwap'] = (merged['straddle'] * merged['v']).cumsum() / merged['v'].cumsum()
                # Compute EMA9 of straddle price
                merged['ema9'] = merged['straddle'].ewm(span=9, adjust=False).mean()
                results[strike] = merged
                successful_fetches += 1
                logger.info(f"✓ Successfully processed strike {strike}")
            else:
                logger.warning(f"✗ Failed to fetch data for strike {strike}")

        if results:
            rankings = compute_rankings(results)
            docs_path, backup_path = build_dashboard_html(results, atm, rankings)
            logger.info(f"✓ Dashboard generated: {docs_path}")

            # Send to Telegram
            send_telegram_summary(rankings, atm, successful_fetches, len(OFFSETS))
            send_telegram_document(
                docs_path,
                caption=f"📊 <b>Straddle Dashboard</b> — {TARGET_DATE}\nOpen in browser to view interactive charts."
            )

            logger.info(f"✓ Successfully processed {successful_fetches}/{len(OFFSETS)} strikes")
        else:
            logger.error("No data successfully fetched for any strike")
            send_telegram_message(f"❌ <b>Straddle Analyser Failed</b>\nDate: {TARGET_DATE}\nNo data fetched for any strike.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        send_telegram_message(f"❌ <b>Straddle Analyser Error</b>\nDate: {TARGET_DATE}\n<code>{str(e)}</code>")

if __name__ == "__main__":
    main()
