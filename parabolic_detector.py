import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel

# Try to import scipy, fall back to numpy polyfit if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not installed. Using NumPy for calculations (reduced accuracy)")

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

# Verify credentials
if not CLIENT_ID or not TOKEN:
    print("❌ ERROR: Missing API credentials. Set CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.")
    exit(1)

# Dynamic Expiry Mapping
EXPIRY_MAP = {
    "2026-04-07": "26421",
    "2026-04-08": "26421",
    "2026-04-09": "26421",
    "2026-04-13": "26421",
    "2026-04-15": "26421",
    "2026-04-16": "26421",
    "2026-04-20": "26421",  
    "2026-04-21": "26421",
    "2026-04-22": "26APR",
    "2026-04-23": "26APR",
    "2026-04-24": "26APR"
}

DATES_TO_TEST = list(EXPIRY_MAP.keys())
OFFSETS = [-300, -200, -100, 0, 100, 200, 300] 
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

try:
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")
    print("✅ Fyers API initialized")
except Exception as e:
    print(f"❌ Failed to initialize Fyers API: {e}")
    exit(1)

# --- ENHANCED GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70, 80]
ENTRY_SPEEDS = [round(-0.4 - (i * 0.1), 2) for i in range(5)]
EXIT_SPEEDS = [-0.1, -0.15, 0]
SL_RANGE = [10, 8, 6, 5]

# Parabolic detection parameters (reduced for faster testing)
ACCELERATION_THRESHOLD = [-0.3, -0.2]  # Reduced from 3 to 2 values
MIN_DOWN_CANDLES = [4, 5]
MAX_HIGH_SPIKE = [15, 20]
MOMENTUM_BREADTH = [2, 3]
SPEED_DECELERATION = [0.4, 0.5]
RSI_OVERSOLD = [25, 30]
PVR_RATIO = [0.2, 0.3]
BODY_RATIO = [0.3, 0.4]

ENTRY_TIMES = []
curr = datetime.strptime("10:15", "%H:%M")
end = datetime.strptime("14:45", "%H:%M")
while curr <= end:
    ENTRY_TIMES.append(curr.strftime("%H:%M"))
    curr += timedelta(minutes=15)

def get_history(symbol, date, res):
    """Fetch historical data with caching"""
    filepath = os.path.join(DATA_DIR, f"{symbol.replace(':', '_')}_{res}_{date}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["time"] = pd.to_datetime(df["time"])
        return df
    
    print(f"  Fetching {symbol} ({res}) for {date}...")
    time.sleep(0.7)  # Rate limiting
    arg = {
        "symbol": symbol, 
        "resolution": res, 
        "date_format": "1", 
        "range_from": date, 
        "range_to": date, 
        "cont_flag": "1"
    }
    
    try:
        resp = fyers.history(data=arg)
        if resp.get("s") == "ok" and resp.get("candles"):
            df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
            df["time"] = (pd.to_datetime(df["epoch"], unit="s")
                         .dt.tz_localize("UTC")
                         .dt.tz_convert(IST)
                         .dt.tz_localize(None))
            df.to_csv(filepath, index=False)
            return df
        else:
            print(f"  ⚠️ No data for {symbol}: {resp.get('message', 'Unknown error')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

def prepare_data():
    """Prepare all data for backtesting"""
    master = {}
    total_strikes = 0
    
    for date in DATES_TO_TEST:
        print(f"\n📅 Processing {date}...")
        expiry = EXPIRY_MAP.get(date)
        if not expiry:
            print(f"  ⚠️ No expiry mapping for {date}")
            continue
            
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty:
            print(f"  ❌ No NIFTY data for {date}")
            continue
        
        # Get ATM price from 10:15 AM
        morning = nifty[nifty['time'].dt.strftime("%H:%M") <= "10:15"]
        if morning.empty:
            price_b = nifty.iloc[0]['c']
        else:
            price_b = morning.iloc[-1]['c']
        
        base_atm = int(round(price_b / 100) * 100)
        print(f"  ATM: {base_atm} (based on {price_b:.2f})")
        
        master[date] = {"strikes": {}, "atm": base_atm}
        
        for off in OFFSETS:
            strike = base_atm + off
            ce_sym = f"NSE:NIFTY{expiry}{strike}CE"
            pe_sym = f"NSE:NIFTY{expiry}{strike}PE"
            
            d5ce = get_history(ce_sym, date, "5")
            d5pe = get_history(pe_sym, date, "5")
            d1ce = get_history(ce_sym, date, "1")
            d1pe = get_history(pe_sym, date, "1")
            
            if not (d5ce.empty or d5pe.empty or d1ce.empty or d1pe.empty):
                # Merge 5min data
                m5 = pd.merge(d5ce[['time', 'c', 'v']], 
                             d5pe[['time', 'c', 'v']], 
                             on='time', suffixes=('_ce', '_pe'))
                
                # Merge 1min data with OHLC
                m1 = pd.merge(d1ce[['time', 'o', 'h', 'l', 'c', 'v']], 
                             d1pe[['time', 'o', 'h', 'l', 'c', 'v']], 
                             on='time', suffixes=('_ce', '_pe'))
                
                if len(m5) > 0 and len(m1) > 0:
                    master[date]["strikes"][str(strike)] = {
                        "premium5m": (m5['c_ce'] + m5['c_pe']).tolist(),
                        "volume5m": (m5['v_ce'] + m5['v_pe']).tolist(),
                        "times5m": m5['time'].dt.strftime("%H:%M").tolist(),
                        "premium1m": (m1['c_ce'] + m1['c_pe']).tolist(),
                        "open1m": (m1['o_ce'] + m1['o_pe']).tolist(),
                        "high1m": (m1['h_ce'] + m1['h_pe']).tolist(),
                        "low1m": (m1['l_ce'] + m1['l_pe']).tolist(),
                        "close1m": (m1['c_ce'] + m1['c_pe']).tolist(),
                        "volume1m": (m1['v_ce'] + m1['v_pe']).tolist(),
                        "times1m": m1['time'].dt.strftime("%H:%M").tolist(),
                        "offset": off
                    }
                    total_strikes += 1
        
        print(f"  ✅ Loaded {len(master[date]['strikes'])} strikes for {date}")
    
    print(f"\n📊 Total strikes loaded: {total_strikes}")
    return master

def calculate_quadratic_convexity(premiums, window=6):
    """Calculate quadratic coefficient using numpy polyfit"""
    if len(premiums) < window:
        return 0
    x = np.arange(window)
    y = premiums[-window:]
    coeffs = np.polyfit(x, y, 2)
    return coeffs[0]  # Negative = concave down (parabolic)

def calculate_rsi(premiums, window=14):
    """Calculate RSI for premium series"""
    if len(premiums) < window + 1:
        return 50
    
    deltas = np.diff(premiums[-window-1:])
    gains = deltas[deltas > 0].sum() / window
    losses = -deltas[deltas < 0].sum() / window
    
    if losses == 0:
        return 100
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_body_ratio(open_prices, close_prices, high_prices, low_prices):
    """Calculate candle body ratio for exhaustion detection"""
    body = np.abs(close_prices - open_prices)
    range_ = high_prices - low_prices
    range_ = np.where(range_ == 0, 1, range_)
    return body / range_

def enhanced_simulate(strike_data, entry_time, smooth_threshold, entry_speed_threshold, sl_points,
                     accel_threshold, min_down_candles, max_spike_pct, 
                     breadth_threshold, exit_deceleration, rsi_oversold, pvr_ratio, body_ratio_exhaust):
    """Enhanced simulation with parabolic move detection"""
    
    # Extract data
    premium_1m = strike_data['premium1m']
    times_1m = strike_data['times1m']
    open_1m = strike_data['open1m']
    close_1m = strike_data['close1m']
    high_1m = strike_data['high1m']
    low_1m = strike_data['low1m']
    premium_5m = strike_data['premium5m']
    times_5m = strike_data['times5m']
    volume_5m = strike_data['volume5m']
    offset = strike_data['offset']
    
    slip = abs(offset) / 400
    trades = []
    active = None
    entry_bar_speed = None
    
    for i in range(len(premium_1m)):
        current_time = times_1m[i]
        if current_time < entry_time:
            continue
            
        # Find corresponding 5min bar
        try:
            idx_5m = times_5m.index(current_time) if current_time in times_5m else -1
        except ValueError:
            idx_5m = -1
        
        if idx_5m != -1 and idx_5m >= 5:
            # Calculate metrics on 5min data
            window_premiums = premium_5m[max(0, idx_5m-5):idx_5m+1]
            
            # Smoothness (linearity)
            net_change = window_premiums[-1] - window_premiums[0]
            total_volatility = sum(abs(window_premiums[j] - window_premiums[j-1]) 
                                  for j in range(1, len(window_premiums)))
            smoothness = (abs(net_change) / total_volatility) * 100 if total_volatility > 0 else 0
            
            # Speed (rate of change per minute)
            speed = net_change / (len(window_premiums) * 5)
            
            # Delta-speed (acceleration)
            if idx_5m >= 6:
                prev_window = premium_5m[max(0, idx_5m-6):idx_5m]
                prev_net = prev_window[-1] - prev_window[0]
                prev_speed = prev_net / (len(prev_window) * 5) if len(prev_window) > 0 else 0
                delta_speed = speed - prev_speed
            else:
                delta_speed = 0
            
            # Quadratic convexity
            convexity = calculate_quadratic_convexity(window_premiums, min(6, len(window_premiums)))
            
            # Volume confirmation
            if len(volume_5m) >= 3:
                avg_volume = np.mean(volume_5m[max(0, idx_5m-3):idx_5m+1])
                volume_trend = volume_5m[idx_5m] > avg_volume * 1.2 if idx_5m >= 0 else False
            else:
                volume_trend = False
            
            # Count down candles (1min)
            if i >= 5:
                down_candles = sum(1 for j in range(i-5, i+1) if close_1m[j] < open_1m[j])
            else:
                down_candles = 0
            
            # Recent spike check
            if i >= 6:
                max_premium = max(premium_1m[max(0, i-6):i+1])
                spike_pct = ((max_premium - premium_1m[i]) / premium_1m[i]) * 100 if premium_1m[i] > 0 else 0
            else:
                spike_pct = 0
            
            # RSI
            rsi = calculate_rsi(premium_1m[:i+1], 14)
            
            # Premium Velocity Ratio
            if i >= 6:
                recent_speed = (premium_1m[i] - premium_1m[i-1]) if i > 0 else 0
                avg_speed_6 = (premium_1m[i] - premium_1m[max(0, i-6)]) / 6
                pvr = abs(recent_speed / avg_speed_6) if avg_speed_6 != 0 else 1
            else:
                pvr = 1
            
            # Body ratio
            if i >= 2:
                body_ratios = calculate_body_ratio(
                    np.array(open_1m[max(0, i-2):i+1]),
                    np.array(close_1m[max(0, i-2):i+1]),
                    np.array(high_1m[max(0, i-2):i+1]),
                    np.array(low_1m[max(0, i-2):i+1])
                )
                avg_body_ratio = np.mean(body_ratios)
            else:
                avg_body_ratio = 0.5
            
            # ENTRY CONDITIONS
            if not active:
                entry_conditions = (
                    smoothness >= smooth_threshold and
                    speed <= entry_speed_threshold and
                    delta_speed <= accel_threshold and
                    convexity < 0 and
                    volume_trend and
                    down_candles >= min_down_candles and
                    spike_pct <= max_spike_pct and
                    rsi > rsi_oversold and
                    pvr > pvr_ratio
                )
                
                if entry_conditions and current_time <= "14:45":
                    active = {
                        'entry_price': premium_1m[i] - slip,
                        'entry_time': current_time,
                        'entry_bar': i,
                        'tsl': (premium_1m[i] - slip) + sl_points,
                        'entry_speed': speed,
                        'entry_rsi': rsi
                    }
                    entry_bar_speed = speed
            
            # EXIT CONDITIONS
            elif active:
                # Update trailing stop
                if active['entry_price'] - premium_1m[i] >= 15:
                    active['tsl'] = min(active['tsl'], active['entry_price'] - 5)
                
                exit_signal = False
                exit_reason = ""
                
                # Check various exit conditions
                if speed > (entry_bar_speed + exit_deceleration):
                    exit_signal = True
                    exit_reason = "Speed deceleration"
                elif rsi > rsi_oversold and active['entry_rsi'] <= rsi_oversold:
                    exit_signal = True
                    exit_reason = "RSI exhaustion"
                elif pvr < pvr_ratio:
                    exit_signal = True
                    exit_reason = "Velocity decay"
                elif avg_body_ratio < body_ratio_exhaust:
                    exit_signal = True
                    exit_reason = "Body exhaustion"
                elif speed > 0:
                    exit_signal = True
                    exit_reason = "Speed reversal"
                elif premium_1m[i] + slip >= active['tsl']:
                    exit_signal = True
                    exit_reason = "Stop loss"
                elif current_time == "15:25":
                    exit_signal = True
                    exit_reason = "End of day"
                
                if exit_signal:
                    pnl = (active['entry_price'] - (premium_1m[i] + slip)) * 130
                    trades.append({
                        'pnl': pnl,
                        'entry_time': active['entry_time'],
                        'exit_time': current_time,
                        'exit_reason': exit_reason,
                        'bars_held': i - active['entry_bar']
                    })
                    active = None
    
    return trades

def run_grid_search(master_data):
    """Run grid search with parameter combinations"""
    results = []
    
    # Calculate total combinations for progress tracking
    total_combinations = (len(ENTRY_TIMES) * len(SMOOTH_RANGE) * len(ENTRY_SPEEDS) * len(SL_RANGE) *
                         len(ACCELERATION_THRESHOLD) * len(MIN_DOWN_CANDLES) * len(MAX_HIGH_SPIKE) *
                         len(MOMENTUM_BREADTH) * len(SPEED_DECELERATION) * len(RSI_OVERSOLD) *
                         len(PVR_RATIO) * len(BODY_RATIO))
    
    print(f"\n🔍 Grid search combinations: {total_combinations}")
    combination_count = 0
    
    for entry_time in ENTRY_TIMES:
        for smooth in SMOOTH_RANGE:
            for entry_speed in ENTRY_SPEEDS:
                for sl in SL_RANGE:
                    for accel in ACCELERATION_THRESHOLD:
                        for down_candles in MIN_DOWN_CANDLES:
                            for max_spike in MAX_HIGH_SPIKE:
                                for breadth in MOMENTUM_BREADTH:
                                    for decel in SPEED_DECELERATION:
                                        for rsi_os in RSI_OVERSOLD:
                                            for pvr in PVR_RATIO:
                                                for body_ratio in BODY_RATIO:
                                                    combination_count += 1
                                                    
                                                    if combination_count % 500 == 0:
                                                        print(f"  Progress: {combination_count}/{total_combinations} ({combination_count/total_combinations*100:.1f}%)")
                                                    
                                                    all_trades = []
                                                    for date, date_data in master_data.items():
                                                        strike_trades = []
                                                        for strike, strike_info in date_data['strikes'].items():
                                                            trades = enhanced_simulate(
                                                                strike_info, entry_time, smooth, entry_speed, sl,
                                                                accel, down_candles, max_spike, breadth,
                                                                decel, rsi_os, pvr, body_ratio
                                                            )
                                                            if trades:
                                                                strike_trades.extend(trades)
                                                        
                                                        # Breadth check
                                                        if len(strike_trades) >= breadth:
                                                            all_trades.extend(strike_trades)
                                                        else:
                                                            all_trades = []
                                                            break
                                                    
                                                    if len(all_trades) >= 3:
                                                        pnls = [t['pnl'] for t in all_trades]
                                                        gross_pnl = sum(pnls)
                                                        total_trades = len(pnls)
                                                        net_pnl = gross_pnl - (total_trades * 200)
                                                        win_rate = (sum(1 for p in pnls if p > 0) / total_trades) * 100
                                                        
                                                        cumulative = np.cumsum(pnls)
                                                        peak = np.maximum.accumulate(cumulative)
                                                        drawdown = peak - cumulative
                                                        max_drawdown = np.max(drawdown)
                                                        
                                                        gross_profit = sum(p for p in pnls if p > 0)
                                                        gross_loss = abs(sum(p for p in pnls if p < 0))
                                                        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                                                        
                                                        score = (net_pnl / max_drawdown) * (win_rate / 100) * profit_factor if max_drawdown > 0 else 0
                                                        
                                                        results.append({
                                                            'params': {
                                                                'entry_time': entry_time,
                                                                'smoothness': smooth,
                                                                'entry_speed': entry_speed,
                                                                'sl_points': sl,
                                                                'acceleration': accel,
                                                                'down_candles': down_candles,
                                                                'max_spike': max_spike,
                                                                'breadth': breadth,
                                                                'exit_deceleration': decel,
                                                                'rsi_oversold': rsi_os,
                                                                'pvr_ratio': pvr,
                                                                'body_ratio': body_ratio
                                                            },
                                                            'trades': all_trades,
                                                            'metrics': {
                                                                'net_pnl': net_pnl,
                                                                'gross_pnl': gross_pnl,
                                                                'total_trades': total_trades,
                                                                'win_rate': win_rate,
                                                                'max_drawdown': max_drawdown,
                                                                'profit_factor': profit_factor,
                                                                'score': score
                                                            }
                                                        })
    
    results.sort(key=lambda x: x['metrics']['score'], reverse=True)
    return results

def generate_html_dashboard(results, master_data):
    """Generate simplified HTML dashboard"""
    top_results = results[:20]
    
    # Prepare data for JSON
    dashboard_data = {
        'top_results': [
            {
                'rank': i+1,
                'score': round(r['metrics']['score'], 2),
                'params': r['params'],
                'metrics': {
                    'net_pnl': round(r['metrics']['net_pnl'], 2),
                    'total_trades': r['metrics']['total_trades'],
                    'win_rate': round(r['metrics']['win_rate'], 2),
                    'max_drawdown': round(r['metrics']['max_drawdown'], 2),
                    'profit_factor': round(r['metrics']['profit_factor'], 2)
                },
                'sample_trades': [
                    {
                        'entry_time': t['entry_time'],
                        'exit_time': t['exit_time'],
                        'pnl': round(t['pnl'], 2),
                        'exit_reason': t['exit_reason']
                    } for t in r['trades'][:10]
                ]
            } for i, r in enumerate(top_results)
        ]
    }
    
    json_data = json.dumps(dashboard_data)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Parabolic Move Backtest Results</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .badge {{ display: inline-block; padding: 3px 6px; border-radius: 3px; font-size: 11px; background: #e0e0e0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Parabolic Move Detection - Backtest Results</h1>
        <div class="card">
            <h2>🏆 Top 20 Strategies</h2>
            <table id="resultsTable">
                <thead>
                    <tr><th>Rank</th><th>Score</th><th>Entry</th><th>Win%</th><th>Net P&L</th><th>Trades</th><th>Max DD</th><th>P Factor</th></tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    <script>
        const data = {json_data};
        const tbody = document.querySelector('#resultsTable tbody');
        data.top_results.forEach(r => {{
            tbody.innerHTML += `
                <tr>
                    <td><b>#${{r.rank}}</b></td>
                    <td>${{r.score}}</td>
                    <td>${{r.params.entry_time}}</td>
                    <td class="positive">${{r.metrics.win_rate}}%</td>
                    <td class="${{r.metrics.net_pnl > 0 ? 'positive' : 'negative'}}">₹${{r.metrics.net_pnl.toLocaleString()}}</td>
                    <td>${{r.metrics.total_trades}}</td>
                    <td class="negative">₹${{r.metrics.max_drawdown.toLocaleString()}}</td>
                    <td>${{r.metrics.profit_factor}}</td>
                </tr>
            `;
        }});
    </script>
</body>
</html>"""
    
    with open("parabolic_dashboard.html", "w") as f:
        f.write(html)
    print("✅ Dashboard generated: parabolic_dashboard.html")

if __name__ == "__main__":
    print("🚀 Starting Parabolic Move Detection Backtest")
    print("=" * 60)
    
    print("\n📊 Preparing data...")
    data = prepare_data()
    
    if data:
        print(f"\n✅ Data prepared for {len(data)} dates")
        results = run_grid_search(data)
        
        if results:
            print(f"\n✅ Found {len(results)} valid configurations")
            generate_html_dashboard(results, data)
            
            best = results[0]
            print("\n🏆 BEST CONFIGURATION:")
            print(f"   Score: {best['metrics']['score']:.2f}")
            print(f"   Net P&L: ₹{best['metrics']['net_pnl']:,.2f}")
            print(f"   Win Rate: {best['metrics']['win_rate']:.1f}%")
            print(f"   Total Trades: {best['metrics']['total_trades']}")
            print(f"   Max Drawdown: ₹{best['metrics']['max_drawdown']:,.2f}")
            print("\n📋 Best Parameters:")
            for key, value in best['params'].items():
                print(f"   {key}: {value}")
        else:
            print("❌ No valid configurations found")
    else:
        print("❌ No data available")
