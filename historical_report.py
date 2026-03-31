import os
import math
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from fyers_apiv3 import fyersModel

# ==========================================
# ─── CONFIGURATION ────────────────────────
# ==========================================

# These are pulled from your GitHub Repo Secrets
CLIENT_ID = os.getenv("CLIENT_ID") 
TOKEN = os.getenv("FYERS_ACCESS_TOKEN") 

# Target Date for Analysis (Format: YYYY-MM-DD)
# Change this to any past trading day to test, e.g., "2026-03-30"
TARGET_DATE = "2026-03-30" 

# Symbol & Expiry Settings
# Note: For April 7th 2026, Fyers format is usually "26407" or "26APR07"
EXPIRY = "26407"  
RESOLUTION = "5"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# UI / Styling
BG_COLOR = "#0b0f1a"
BORDER_COLOR = "#1e2d40"
ACCENT_COLOR = "#00e5b0"
TEXT_COLOR = "#e2e8f0"
STRIKE_COLORS = ["#38bdf8", "#a78bfa", "#f97316", "#ff4560", "#fbbf24", "#00e5b0", "#ec4899", "#84cc16", "#64748b"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# ─── DATA LAYER ───────────────────────────
# ==========================================

def get_fyers():
    if not TOKEN:
        raise ValueError("FYERS_ACCESS_TOKEN is missing in Secrets!")
    return fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_current_atm(fyers):
    """Fetches Nifty 50 Price and rounds to nearest 100."""
    quotes = {"symbols": "NSE:NIFTY50-INDEX"}
    resp = fyers.depth(data=quotes)
    if resp.get("s") == "ok" and "NSE:NIFTY50-INDEX" in resp["d"]:
        cmp = resp["d"]["NSE:NIFTY50-INDEX"]["last_price"]
        if cmp == 0:
            logger.warning("CMP is 0 (Market Closed). Using fallback ATM 24500.")
            return 24500
        atm = round(cmp / 100) * 100
        logger.info(f"Nifty CMP: {cmp} | Calculated ATM: {atm}")
        return atm
    logger.warning("Could not fetch CMP from API. Using fallback ATM 24500.")
    return 24500

def fetch_candles(fyers, symbol, target_date):
    """Fetches 5-min candles and filters for market hours."""
    data = {
        "symbol": symbol,
        "resolution": RESOLUTION,
        "date_format": "1",
        "range_from": target_date,
        "range_to": target_date,
        "cont_flag": "1",
    }
    resp = fyers.history(data=data)
    
    if resp.get("s") != "ok":
        logger.error(f"API Error for {symbol}: {resp.get('message')}")
        return pd.DataFrame()
    
    candles = resp.get("candles", [])
    if not candles:
        logger.warning(f"No data found for {symbol} on {target_date}")
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["epoch","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    
    # Filter for NSE Market Hours
    t = df["time"].dt.time
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return df[(t >= open_t) & (t <= close_t)].reset_index(drop=True)

def compute_straddle(ce_df, pe_df):
    """Combines CE and PE data into a single Straddle DataFrame with VWAP."""
    if ce_df.empty or pe_df.empty:
        return pd.DataFrame()
        
    merged = pd.merge(
        ce_df[["time","close","volume"]].rename(columns={"close":"ce_close","volume":"ce_vol"}),
        pe_df[["time","close","volume"]].rename(columns={"close":"pe_close","volume":"pe_vol"}),
        on="time", how="inner"
    )
    
    merged["straddle"] = merged["ce_close"] + merged["pe_close"]
    merged["combined_vol"] = merged["ce_vol"] + merged["pe_vol"]
    
    # Calculate Cumulative VWAP
    merged["vwap"] = (merged["straddle"] * merged["combined_vol"]).cumsum() / merged["combined_vol"].cumsum()
    return merged

# ==========================================
# ─── CHARTING & EXPORT ────────────────────
# ==========================================

def build_report(straddle_data, atm, report_date):
    """Generates a multi-row interactive Plotly chart and saves to HTML."""
    strikes = sorted(straddle_data.keys())
    n = len(strikes)
    
    fig = make_subplots(
        rows=n, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        subplot_titles=[f"Strike {s}{' (ATM)' if s == atm else ''}" for s in strikes]
    )

    for idx, strike in enumerate(strikes):
        df = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        row = idx + 1
        
        # Straddle Price Line
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["straddle"], 
            name=f"Straddle {strike}",
            line=dict(color=color, width=2)
        ), row=row, col=1)
        
        # VWAP Dotted Line
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["vwap"], 
            name=f"VWAP {strike}",
            line=dict(color="#64748b", width=1, dash="dot")
        ), row=row, col=1)

    fig.update_layout(
        height=300 * n, 
        template="plotly_dark", 
        paper_bgcolor=BG_COLOR, 
        plot_bgcolor=BG_COLOR,
        title=dict(
            text=f"Nifty Straddle Analysis - {report_date}",
            font=dict(size=20, color=ACCENT_COLOR)
        ),
        showlegend=False,
        margin=dict(l=50, r=20, t=80, b=50)
    )
    
    fig.write_html("index.html")
    logger.info(f"✅ Successfully generated index.html for {report_date}")

# ==========================================
# ─── MAIN ─────────────────────────────────
# ==========================================

def main():
    try:
        fyers = get_fyers()
        atm = get_current_atm(fyers)
        
        results = {}
        for offset in OFFSETS:
            strike = atm + offset
            ce_sym = f"NSE:NIFTY{EXPIRY}{strike}CE"
            pe_sym = f"NSE:NIFTY{EXPIRY}{strike}PE"
            
            logger.info(f"Fetching data for Strike: {strike}...")
            ce_df = fetch_candles(fyers, ce_sym, TARGET_DATE)
            pe_df = fetch_candles(fyers, pe_sym, TARGET_DATE)
            st_df = compute_straddle(ce_df, pe_df)
            
            if not st_df.empty:
                results[strike] = st_df

        if results:
            build_report(results, atm, TARGET_DATE)
        else:
            logger.error(f"❌ No data found for any strikes on {TARGET_DATE}. Check your Expiry/Date settings.")

    except Exception as e:
        logger.error(f"❌ Critical Script Error: {e}")

if __name__ == "__main__":
    main()
