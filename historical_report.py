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
TARGET_DATE = "2026-04-01" # Change to desired date
EXPIRY = "26407"  
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

# UI Theme Colors
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

# --- ANALYTICS FUNCTIONS ---

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

def _trend(a: float) -> str:
    return "UP" if a > 5 else ("DOWN" if a < -5 else "FLAT")

def compute_rankings(straddle_data: dict) -> list:
    rows = []
    for strike, df in straddle_data.items():
        prices = df["straddle"].tolist()
        sm = _smoothness(prices)
        an = _angle(prices)
        tr = _trend(an)
        rows.append({
            "strike": strike, 
            "smoothness": sm, 
            "angle": an, 
            "trend": tr
        })
    
    # Sort by Smoothness for ranking
    ranked_rows = sorted(rows, key=lambda x: x['smoothness'], reverse=True)
    for i, r in enumerate(ranked_rows):
        r["rank"] = i + 1
    return ranked_rows

# --- DASHBOARD BUILDER ---

def build_dashboard_html(straddle_data, atm, rankings):
    # 1. Main Straddle Charts
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
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"{strike}", line=dict(color=color, width=2)), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["vwap"], name="VWAP", line=dict(color=MUTED, width=1, dash="dot")), row=row, col=1)

    fig_main.update_layout(height=2000, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)

    # 2. Ranking Chart (Horizontal Bar)
    fig_rank = go.Figure(go.Bar(
        y=[str(r["strike"]) for r in rankings],
        x=[r["smoothness"] for r in rankings],
        orientation="h",
        marker_color=[ACCENT if r["rank"] == 1 else "#1e3a2f" for r in rankings],
        text=[f"{r['smoothness']}%" for r in rankings],
        textposition="outside",
        cliponaxis=False
    ))
    fig_rank.update_layout(
        title="Smoothness Ranking (Full Day)",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD, 
        height=400, margin=dict(l=10, r=40, t=40, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 110])
    )

    # 3. Ranking Table HTML
    table_rows = "".join([
        f"""<tr style="border-bottom: 1px solid {BORDER};">
            <td style="padding:10px;">{r['rank']}</td>
            <td style="padding:10px; color:{ACCENT};"><b>{r['strike']}</b></td>
            <td style="padding:10px;">{r['smoothness']}%</td>
            <td style="padding:10px; color:{BLUE if r['angle'] > 0 else RED};">{r['angle']}°</td>
            <td style="padding:10px;">{r['trend']}</td>
        </tr>""" for r in rankings
    ])

    # Convert figures to HTML components
    main_chart_html = fig_main.to_html(full_html=False, include_plotlyjs='cdn')
    rank_chart_html = fig_rank.to_html(full_html=False, include_plotlyjs=False)

    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Straddle Dashboard</title>
        <style>
            body {{ background-color: {BG}; color: {TEXT}; font-family: sans-serif; margin: 0; padding: 0; }}
            .header {{ padding: 20px 40px; background: {CARD}; border-bottom: 1px solid {BORDER}; display: flex; justify-content: space-between; align-items: center; }}
            .content {{ display: flex; padding: 20px; gap: 20px; }}
            .sidebar {{ flex: 1; min-width: 350px; position: sticky; top: 20px; height: fit-content; }}
            .main-charts {{ flex: 3; }}
            .card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
            th {{ color: {MUTED}; text-transform: uppercase; font-size: 10px; padding: 10px; }}
            h2 {{ font-size: 14px; color: {ACCENT}; margin-top: 0; letter-spacing: 1px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div><span style="color:{ACCENT}; font-size:24px;">▣</span> <b style="font-size:20px; letter-spacing:2px;">STRADDLE ANALYSER</b></div>
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
                        <thead>
                            <tr><th>Rank</th><th>Strike</th><th>Smooth</th><th>Angle</th><th>Trend</th></tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
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
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)

# --- MAIN EXECUTION ---

def fetch_candles(fyers, symbol, date):
    data = {"symbol": symbol, "resolution": "5", "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    if resp.get("s") == "ok" and resp.get("candles"):
        df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df
    return pd.DataFrame()

def main():
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")
    # For testing, we use a fixed ATM. In live, you can call get_current_atm(fyers)
    atm = 22900 
    
    results = {}
    for offset in OFFSETS:
        strike = atm + offset
        ce_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}CE", TARGET_DATE)
        pe_df = fetch_candles(fyers, f"NSE:NIFTY{EXPIRY}{strike}PE", TARGET_DATE)
        
        if not ce_df.empty and not pe_df.empty:
            merged = pd.merge(ce_df[['time', 'close', 'volume']], pe_df[['time', 'close', 'volume']], on='time')
            merged['straddle'] = merged['close_x'] + merged['close_y']
            merged['v'] = merged['volume_x'] + merged['volume_y']
            merged['vwap'] = (merged['straddle'] * merged['v']).cumsum() / merged['v'].cumsum()
            results[strike] = merged

    if results:
        rankings = compute_rankings(results)
        build_dashboard_html(results, atm, rankings)
        logger.info("Dashboard with Rankings successfully created!")
    else:
        logger.error("No data retrieved. Check Expiry/Token.")

if __name__ == "__main__":
    main()
