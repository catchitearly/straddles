import os
import math
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
# These should be set in your GitHub Repo Secrets
CLIENT_ID = os.getenv("CLIENT_ID", "YOUR_CLIENT_ID") 
# Your manually updated token from auth.py
TOKEN = os.getenv("FYERS_ACCESS_TOKEN") 

RESOLUTION = "5"
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
EXPIRY = "26407" # Update this weekly/monthly as needed
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# Styling Constants
BG_COLOR = "#0b0f1a"
BORDER_COLOR = "#1e2d40"
ACCENT_COLOR = "#00e5b0"
TEXT_COLOR = "#e2e8f0"
STRIKE_COLORS = ["#38bdf8", "#a78bfa", "#f97316", "#ff4560", "#fbbf24", "#00e5b0", "#ec4899", "#84cc16", "#64748b"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- DATA LAYER ---

def get_fyers():
    if not TOKEN:
        raise ValueError("FYERS_ACCESS_TOKEN environment variable is missing!")
    return fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_current_atm(fyers):
    """Fetches CMP of Nifty 50 and rounds to nearest 100."""
    quotes = {"symbols": "NSE:NIFTY50-INDEX"}
    resp = fyers.depth(data=quotes)
    if resp.get("s") == "ok":
        cmp = resp["d"]["NSE:NIFTY50-INDEX"]["last_price"]
        atm = round(cmp / 100) * 100
        logger.info(f"Nifty CMP: {cmp} | Calculated ATM: {atm}")
        return atm
    return 22300  # Fallback

def fetch_candles(fyers, symbol, from_date, to_date):
    data = {
        "symbol": symbol,
        "resolution": RESOLUTION,
        "date_format": "1",
        "range_from": from_date,
        "range_to": to_date,
        "cont_flag": "1",
    }
    resp = fyers.history(data=data)
    candles = resp.get("candles", [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["epoch","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    
    # Filter Market Hours
    t = df["time"].dt.time
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return df[(t >= open_t) & (t <= close_t)].reset_index(drop=True)

def compute_straddle(ce_df, pe_df):
    if ce_df.empty or pe_df.empty: return pd.DataFrame()
    merged = pd.merge(
        ce_df[["time","close","volume"]].rename(columns={"close":"ce_close","volume":"ce_vol"}),
        pe_df[["time","close","volume"]].rename(columns={"close":"pe_close","volume":"pe_vol"}),
        on="time", how="inner"
    )
    merged["straddle"] = merged["ce_close"] + merged["pe_close"]
    merged["combined_vol"] = merged["ce_vol"] + merged["pe_vol"]
    merged["vwap"] = (merged["straddle"] * merged["combined_vol"]).cumsum() / merged["combined_vol"].cumsum()
    return merged

# --- CHARTING ---

def build_report(straddle_data, atm):
    strikes = sorted(straddle_data.keys())
    n = len(strikes)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=[f"Strike {s}{' (ATM)' if s == atm else ''}" for s in strikes])

    for idx, strike in enumerate(strikes):
        df = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        
        fig.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"{strike}",
                                 line=dict(color=color, width=2)), row=idx+1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["vwap"], name="VWAP",
                                 line=dict(color="#64748b", width=1, dash="dot")), row=idx+1, col=1)

    fig.update_layout(height=300*n, template="plotly_dark", paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
                      title=f"Nifty Straddle Analysis - {date.today()}", showlegend=False)
    
    # Save to file
    fig.write_html("index.html")
    logger.info("Report saved to index.html")

# --- MAIN EXECUTION ---

def main():
    try:
        fyers = get_fyers()
        atm = get_current_atm(fyers)
        today = date.today().isoformat()
        
        results = {}
        for offset in OFFSETS:
            strike = atm + offset
            ce_sym = f"NSE:NIFTY{EXPIRY}{strike}CE"
            pe_sym = f"NSE:NIFTY{EXPIRY}{strike}PE"
            
            logger.info(f"Processing Strike: {strike}")
            ce_df = fetch_candles(fyers, ce_sym, today, today)
            pe_df = fetch_candles(fyers, pe_sym, today, today)
            st_df = compute_straddle(ce_df, pe_df)
            
            if not st_df.empty:
                results[strike] = st_df

        if results:
            build_report(results, atm)
        else:
            logger.error("No data found for any strikes.")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
