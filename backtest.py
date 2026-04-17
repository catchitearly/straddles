import os
import math
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
DATES_TO_TEST = ["2026-04-10", "2026-04-13", "2026-04-09", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421" 
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
IST = ZoneInfo("Asia/Kolkata")

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

def get_history(symbol, date, res="5"):
    data = {"symbol": symbol, "resolution": res, "date_format": "1", 
            "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=data)
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch","o","h","l","c","v"])
        df["time"] = pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
        return df
    return pd.DataFrame()

def calc_metrics(prices, idx, window):
    if idx < window: return None
    slice_data = prices[idx-window+1 : idx+1]
    net = abs(slice_data[-1] - slice_data[0])
    total = sum(abs(slice_data[i] - slice_data[i-1]) for i in range(1, len(slice_data)))
    smooth = (net / total * 100) if total > 0 else 0
    speed = (slice_data[-1] - slice_data[0]) / (window * 5)
    
    x = list(range(len(slice_data)))
    xm, ym = sum(x)/len(x), sum(slice_data)/len(slice_data)
    num = sum((x[i]-xm)*(slice_data[i]-ym) for i in range(len(x)))
    den = sum((x[i]-xm)**2 for i in range(len(x)))
    angle = math.degrees(math.atan(num/den)) if den != 0 else 0
    trend = "UP" if angle > 5 else ("DOWN" if angle < -5 else "FLAT")
    return {"smooth": smooth, "speed": speed, "trend": trend}

def simulate_backtest(dates):
    all_trades = []

    for date in dates:
        # 1. Fetch Nifty Spot for ATM Base
        nifty_df = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty_df.empty: continue
        open_p = nifty_df.iloc[0]['o']
        try:
            price_b = nifty_df[nifty_df['time'].dt.hour < 11].iloc[-1]['c']
        except: continue
        
        base_atm = int(round(price_b / 50) * 50) if abs(open_p - price_b) > 200 else int(round(open_p / 50) * 50)
        
        # 2. Pre-fetch all Straddle Data for the offsets
        strike_data = {}
        for offset in OFFSETS:
            strike = base_atm + offset
            ce = get_history(f"NSE:NIFTY{EXPIRY}{strike}CE", date)
            pe = get_history(f"NSE:NIFTY{EXPIRY}{strike}PE", date)
            if not ce.empty and not pe.empty:
                merged = pd.merge(ce[['time','c']], pe[['time','c']], on='time')
                merged['straddle'] = merged['c_x'] + merged['c_y']
                strike_data[strike] = merged

        # 3. Parallel Simulation
        # active_trades format: { strike_price: {trade_details} }
        active_trades = {}
        
        # Use a reference timeline (from any strike data)
        if not strike_data: continue
        timeline = list(strike_data.values())[0]

        for i in range(len(timeline)):
            curr_t = timeline.iloc[i]['time']
            if curr_t.hour < 11 or (curr_t.hour == 11 and curr_t.minute < 5): continue

            for strike, df in strike_data.items():
                if i >= len(df): continue
                curr_p = df.iloc[i]['straddle']
                
                # If we have a live trade for this strike, manage it
                if strike in active_trades:
                    trade = active_trades[strike]
                    profit = trade["Entry Price"] - curr_p
                    
                    # Trailing logic
                    if profit >= 20: trade["TSL"] = min(trade["TSL"], trade["Entry Price"] - 12)
                    elif profit >= 15: trade["TSL"] = min(trade["TSL"], trade["Entry Price"] - 8)
                    elif profit >= 10: trade["TSL"] = min(trade["TSL"], trade["Entry Price"] - 5)
                    elif profit >= 8: trade["TSL"] = min(trade["TSL"], trade["Entry Price"] - 3)
                    
                    # Exit check
                    m30 = calc_metrics(df['straddle'].tolist(), i, 6)
                    m60 = calc_metrics(df['straddle'].tolist(), i, 12)
                    
                    exit_reason = None
                    if curr_p >= trade["TSL"]: exit_reason = "TSL Hit"
                    elif curr_p >= trade["Entry Price"] + 10: exit_reason = "Initial SL Hit"
                    elif m30 and m30['speed'] > -0.10: exit_reason = "Speed Slowdown"
                    elif m30 and m60 and (m30['trend'] == "UP" or m60['trend'] == "UP"): exit_reason = "Trend Reversal"
                    
                    if exit_reason:
                        trade.update({
                            "Exit Time": curr_t.strftime("%H:%M"), "Exit Price": round(curr_p, 2),
                            "P&L": round(trade["Entry Price"] - curr_p, 2), "Reason": exit_reason
                        })
                        all_trades.append(trade)
                        del active_trades[strike]

                # Else, check for Entry
                else:
                    m30 = calc_metrics(df['straddle'].tolist(), i, 6)
                    m60 = calc_metrics(df['straddle'].tolist(), i, 12)
                    if m30 and m60:
                        if m30['smooth'] > 70 and m30['speed'] < -0.9 and m30['trend'] == "DOWN" and m60['trend'] == "DOWN":
                            active_trades[strike] = {
                                "Date": date, "Strike": strike, "Entry Time": curr_t.strftime("%H:%M"),
                                "Entry Price": round(curr_p, 2), "TSL": round(curr_p + 10, 2)
                            }
    return all_trades

# --- HTML REPORT GENERATION ---
results = simulate_backtest(DATES_TO_TEST)
res_df = pd.DataFrame(results)

def color_pnl(val):
    color = "#00e5b0" if val > 0 else "#ff4560"
    return f'color: {color}; font-weight: bold'

if not res_df.empty:
    # Use .map instead of .applymap for better compatibility with newer Pandas
    html_table = res_df.style.map(color_pnl, subset=['P&L']).to_html(index=False)
else:
    html_table = "No trades found."

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ background: #0b0f1a; color: #e2e8f0; font-family: sans-serif; padding: 20px; }}
        h2 {{ color: #00e5b0; border-bottom: 1px solid #1e2d40; padding-bottom: 10px; }}
        .dataframe {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #111b27; }}
        th {{ background: #1e2d40; color: #64748b; padding: 12px; text-align: left; font-size: 11px; text-transform: uppercase; }}
        td {{ padding: 12px; border: 1px solid #1e2d40; }}
        tr:hover {{ background: #1a2635; }}
    </style>
</head>
<body>
    <h2>Parallel Straddle Backtest Results</h2>
    {html_table}
</body>
</html>
"""

with open("backtest_report.html", "w") as f:
    f.write(html_content)
