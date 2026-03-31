import os
import math
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID") 
TOKEN = os.getenv("FYERS_ACCESS_TOKEN") 
TARGET_DATE = "2026-03-30" 
EXPIRY = "26407"  
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

# UI Colors (Matching your Dash theme)
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

# --- ANALYTICS ENGINE ---

def _smoothness(prices: list) -> float:
    if len(prices) < 3: return 0.0
    net = abs(prices[-1] - prices[0])
    total = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
    return round((net / total) * 100, 2) if total else 100.0

def _angle(prices: list) -> float:
    n = len(prices)
    if n < 2: return 0.0
    xm, ym = (n - 1) / 2, sum(prices) / n
    num = sum((i - xm) * (prices[i] - ym) for i in range(n))
    den = sum((i - xm) ** 2 for i in range(n))
    return round(math.degrees(math.atan(num / den)), 2) if den else 0.0

def compute_rankings(straddle_data: dict) -> list:
    rows = []
    for strike, df in straddle_data.items():
        prices = df["straddle"].tolist()
        sm = _smoothness(prices)
        an = _angle(prices)
        rows.append({"strike": strike, "smoothness": sm, "angle": an})
    
    # Sort and rank
    rows = sorted(rows, key=lambda x: x['smoothness'], reverse=True)
    for i, r in enumerate(rows): r["smoothness_rank"] = i + 1
    return rows

# --- PLOTTING ---

def build_dashboard(straddle_data, atm, rankings):
    # 1. Main Straddle Charts
    strikes = sorted(straddle_data.keys())
    fig_main = make_subplots(rows=len(strikes), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for idx, strike in enumerate(strikes):
        df = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"S:{strike}", line=dict(color=color, width=2)), row=idx+1, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["vwap"], name="VWAP", line=dict(color=MUTED, width=1, dash="dot")), row=idx+1, col=1)

    fig_main.update_layout(height=1800, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False, title=f"Nifty Analysis - {TARGET_DATE}")

    # 2. Ranking Chart (Smoothness)
    by_smooth = sorted(rankings, key=lambda r: r["smoothness_rank"])
    fig_rank = go.Figure(go.Bar(
        y=[str(r["strike"]) for r in by_smooth],
        x=[r["smoothness"] for r in by_smooth],
        orientation="h",
        marker_color=[ACCENT if r["smoothness_rank"] == 1 else "#1e3a2f" for r in by_smooth],
        text=[f"{r['smoothness']}%" for r in by_smooth],
        textposition="outside"
    ))
    fig_rank.update_layout(title="Smoothness Ranking", template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD, height=400, yaxis=dict(autorange="reversed"))

    # 3. Combine into HTML Dashboard
    # Convert charts to HTML Divs
    main_div = fig_main.to_html(full_html=False, include_plotlyjs='cdn')
    rank_div = fig_rank.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ background-color: {BG}; color: {TEXT}; font-family: 'IBM Plex Mono', monospace; margin: 0; padding: 20px; }}
            .header {{ display: flex; justify-content: space-between; padding: 20px; border-bottom: 1px solid {BORDER}; background: {CARD}; }}
            .container {{ display: flex; gap: 20px; margin-top: 20px; }}
            .sidebar {{ flex: 1; background: {CARD}; padding: 15px; border-radius: 8px; border: 1px solid {BORDER}; height: fit-content; }}
            .main-content {{ flex: 3; }}
            .stat-card {{ background: #0d1520; padding: 10px; border-radius: 4px; margin-bottom: 10px; border-left: 4px solid {ACCENT}; }}
            h2 {{ color: {ACCENT}; font-size: 14px; letter-spacing: 2px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div><b style="color:{ACCENT}; font-size:20px;">▣ STRADDLE DASHBOARD</b> | {TARGET_DATE}</div>
            <div style="color:{MUTED}">ATM: {atm} | Expiry: {EXPIRY}</div>
        </div>
        <div class="container">
            <div class="sidebar">
                <h2>RANKING ANALYTICS</h2>
                {rank_div}
                <div style="margin-top:20px;">
                    <h2>KEY INSIGHTS</h2>
                    <div class="stat-card">Best Strike: {by_smooth[0]['strike']}</div>
                    <div class="stat-card">Avg Smoothness: {round(sum(r['smoothness'] for r in rankings)/len(rankings), 2)}%</div>
                </div>
            </div>
            <div class="main-content">
                {main_div}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# --- DATA FETCHING & MAIN ---

def fetch_candles(fyers, symbol, date):
    data = {{"symbol": symbol, "resolution": "5", "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}}
    resp = fyers.history(data=data)
    if resp.get("s") != "ok" or not resp.get("candles"): return pd.DataFrame()
    df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df

def main():
    try:
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")
        # Simple manual ATM for testing or use your get_current_atm()
        atm = 24500 
        
        results = {}
        for offset in OFFSETS:
            strike = atm + offset
            ce_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}CE", TARGET_DATE)
            pe_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}PE", TARGET_DATE)
            
            if not ce_df.empty and not pe_df.empty:
                merged = pd.merge(ce_df[['time', 'close', 'volume']], pe_df[['time', 'close', 'volume']], on='time', suffixes=('_ce', '_pe'))
                merged['straddle'] = merged['close_ce'] + merged['close_pe']
                merged['vol'] = merged['volume_ce'] + merged['volume_pe']
                merged['vwap'] = (merged['straddle'] * merged['vol']).cumsum() / merged['vol'].cumsum()
                results[strike] = merged

        if results:
            rankings = compute_rankings(results)
            build_dashboard(results, atm, rankings)
            print("Dashboard Generated Successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
