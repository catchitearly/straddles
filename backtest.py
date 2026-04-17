import os
import math
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
# Add your list of dates here for backtesting
DATES_TO_TEST = ["2026-04-10", "2026-04-13", "2026-04-15", "2026-04-16", "2026-04-17"]
EXPIRY = "26421" 
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
    slice = prices[idx-window+1 : idx+1]
    net = abs(slice[-1] - slice[0])
    total = sum(abs(slice[i] - slice[i-1]) for i in range(1, len(slice)))
    smooth = (net / total * 100) if total > 0 else 0
    speed = (slice[-1] - slice[0]) / (window * 5)
    
    x = list(range(len(slice)))
    xm, ym = sum(x)/len(x), sum(slice)/len(slice)
    num = sum((x[i]-xm)*(slice[i]-ym) for i in range(len(x)))
    den = sum((x[i]-xm)**2 for i in range(len(x)))
    angle = math.degrees(math.atan(num/den)) if den != 0 else 0
    trend = "UP" if angle > 5 else ("DOWN" if angle < -5 else "FLAT")
    return {"smooth": smooth, "speed": speed, "trend": trend}

def simulate_day(date):
    # 1. Fetch Nifty Spot for ATM Logic
    nifty_df = get_history("NSE:NIFTY50-INDEX", date, "1")
    if nifty_df.empty: return {"date": date, "status": "No Spot Data"}
    
    open_p = nifty_df.iloc[0]['o']
    try:
        eleven_am = nifty_df[nifty_df['time'].dt.hour < 11].iloc[-1]['c']
    except: return {"date": date, "status": "Market closed before 11"}
    
    atm = int(round(eleven_am / 50) * 50) if abs(open_p - eleven_am) > 200 else int(round(open_p / 50) * 50)
    
    # 2. Fetch Straddle Data
    ce = get_history(f"NSE:NIFTY{EXPIRY}{atm}CE", date)
    pe = get_history(f"NSE:NIFTY{EXPIRY}{atm}PE", date)
    if ce.empty or pe.empty: return {"date": date, "status": "No Option Data"}
    
    df = pd.merge(ce[['time','c']], pe[['time','c']], on='time')
    df['straddle'] = df['c_x'] + df['c_y']
    prices = df['straddle'].tolist()
    
    # 3. Trade Simulation
    in_trade = False
    entry_p, tsl = 0, 0
    entry_time = ""
    
    for i in range(len(df)):
        curr_t = df.iloc[i]['time']
        if curr_t.hour < 11 or (curr_t.hour == 11 and curr_t.minute < 5): continue
        
        curr_p = prices[i]
        m30 = calc_metrics(prices, i, 6)
        m60 = calc_metrics(prices, i, 12)
        if not m30 or not m60: continue

        if not in_trade:
            # Entry Criteria
            if m30['smooth'] > 70 and m30['speed'] < -0.9 and m30['trend'] == "DOWN" and m60['trend'] == "DOWN":
                in_trade, entry_p, entry_time = True, curr_p, curr_t.strftime("%H:%M")
                tsl = entry_p + 10
        else:
            # Trailing & Exit
            profit = entry_p - curr_p
            if profit >= 20: tsl = min(tsl, entry_p - 12)
            elif profit >= 15: tsl = min(tsl, entry_p - 8)
            elif profit >= 10: tsl = min(tsl, entry_p - 5)
            elif profit >= 8: tsl = min(tsl, entry_p - 3)
            
            if curr_p >= tsl or curr_p >= entry_p + 10 or m30['speed'] > -0.10 or m30['trend'] == "UP" or m60['trend'] == "UP":
                return {"date": date, "atm": atm, "entry": entry_p, "exit": curr_p, 
                        "time": entry_time, "pnl": round(entry_p - curr_p, 2), "status": "Success"}
                
    return {"date": date, "atm": atm, "status": "No Trade"}

# --- EXECUTION & HTML GENERATION ---
results = [simulate_day(d) for d in DATES_TO_TEST]
res_df = pd.DataFrame(results)

html_template = f"""
<html>
<head><style>
    body {{ font-family: sans-serif; background: #1a1a1a; color: white; padding: 40px; }}
    table {{ width: 100%; border-collapse: collapse; background: #2d2d2d; }}
    th, td {{ padding: 12px; border: 1px solid #444; text-align: left; }}
    th {{ background: #00e5b0; color: black; }}
    .profit {{ color: #00e5b0; font-weight: bold; }}
    .loss {{ color: #ff4560; font-weight: bold; }}
</style></head>
<body>
    <h2>Straddle Backtest Summary (Expiry: {EXPIRY})</h2>
    {res_df.to_html(index=False, classes='table').replace('style="text-align: right;"', '')}
</body>
</html>
"""

with open("backtest_report.html", "w") as f:
    f.write(html_template.replace('<td>-', '<td class="loss">-').replace('<td>+', '<td class="profit">+'))

print("Backtest complete. Report saved to backtest_report.html")
