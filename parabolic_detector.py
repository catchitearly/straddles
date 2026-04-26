import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel
from scipy import stats

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

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
    "2026-04-24": "26APR"
}

DATES_TO_TEST = list(EXPIRY_MAP.keys())
OFFSETS = [-300, -200, -100, 0, 100, 200, 300] 
IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")

# --- ENHANCED GRID DEFINITIONS ---
SMOOTH_RANGE = [40, 50, 60, 70, 80]
ENTRY_SPEEDS = [round(-0.4 - (i * 0.1), 2) for i in range(5)]
EXIT_SPEEDS = [-0.1, -0.15, 0]
SL_RANGE = [10, 8, 6, 5]

# New parameters for parabolic detection
ACCELERATION_THRESHOLD = [-0.3, -0.2, -0.1]  # Delta-speed negative thresholds
MIN_DOWN_CANDLES = [4, 5]  # Minimum down candles in last N bars
MAX_HIGH_SPIKE = [15, 20, 25]  # Max % spike from highest premium
MOMENTUM_BREADTH = [2, 3]  # Minimum strikes showing same signal
SPEED_DECELERATION = [0.4, 0.5, 0.6]  # Speed improvement threshold for exit
RSI_OVERSOLD = [25, 30, 35]  # RSI threshold for oversold
PVR_RATIO = [0.2, 0.3, 0.4]  # Premium velocity ratio threshold
BODY_RATIO = [0.3, 0.4, 0.5]  # Body ratio exhaustion threshold

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
    time.sleep(0.7)
    arg = {"symbol": symbol, "resolution": res, "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}
    resp = fyers.history(data=arg)
    if resp.get("s") == "ok":
        df = pd.DataFrame(resp["candles"], columns=["epoch", "o", "h", "l", "c", "v"])
        df["time"] = (pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None))
        df.to_csv(filepath, index=False)
        return df
    return pd.DataFrame()

def prepare_data():
    """Prepare all data for backtesting"""
    master = {}
    for date in DATES_TO_TEST:
        expiry = EXPIRY_MAP.get(date)
        nifty = get_history("NSE:NIFTY50-INDEX", date, "1")
        if nifty.empty or not expiry: continue
        
        morning = nifty[nifty['time'].dt.strftime("%H:%M") <= "10:15"]
        price_b = morning.iloc[-1]['c'] if not morning.empty else nifty.iloc[0]['o']
        base_atm = int(round(price_b / 100) * 100)
        
        master[date] = {"strikes": {}}
        for off in OFFSETS:
            strike = base_atm + off
            ce_sym, pe_sym = f"NSE:NIFTY{expiry}{strike}CE", f"NSE:NIFTY{expiry}{strike}PE"
            d5ce, d5pe = get_history(ce_sym, date, "5"), get_history(pe_sym, date, "5")
            d1ce, d1pe = get_history(ce_sym, date, "1"), get_history(pe_sym, date, "1")
            
            if not (d5ce.empty or d5pe.empty or d1ce.empty or d1pe.empty):
                m5 = pd.merge(d5ce[['time', 'c', 'v']], d5pe[['time', 'c', 'v']], on='time', suffixes=('_ce', '_pe'))
                m1 = pd.merge(d1ce[['time', 'o', 'h', 'l', 'c', 'v']], d1pe[['time', 'o', 'h', 'l', 'c', 'v']], on='time', suffixes=('_ce', '_pe'))
                
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
    return master

def calculate_quadratic_convexity(premiums, window=6):
    """Calculate quadratic coefficient for convexity detection"""
    if len(premiums) < window:
        return 0
    x = np.arange(window)
    y = premiums[-window:]
    coeffs = np.polyfit(x, y, 2)
    return coeffs[0]  # Quadratic coefficient (negative = concave down)

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
    return 100 - (100 / (1 + rs))

def calculate_body_ratio(open_prices, close_prices, high_prices, low_prices):
    """Calculate candle body ratio for exhaustion detection"""
    body = abs(close_prices - open_prices)
    range_ = high_prices - low_prices
    range_ = np.where(range_ == 0, 1, range_)  # Avoid division by zero
    return body / range_

def enhanced_simulate(strike_data, entry_time, smooth_threshold, entry_speed_threshold, sl_points,
                     accel_threshold, min_down_candles, max_spike_pct, 
                     breadth_threshold, exit_deceleration, rsi_oversold, pvr_ratio, body_ratio_exhaust):
    """Enhanced simulation with parabolic move detection"""
    trades = []
    premium_1m = strike_data['premium1m']
    times_1m = strike_data['times1m']
    open_1m = strike_data['open1m']
    close_1m = strike_data['close1m']
    high_1m = strike_data['high1m']
    low_1m = strike_data['low1m']
    volume_1m = strike_data['volume1m']
    premium_5m = strike_data['premium5m']
    times_5m = strike_data['times5m']
    volume_5m = strike_data['volume5m']
    offset = strike_data['offset']
    
    slip = abs(offset) / 400
    active = None
    entry_bar_speed = None
    entry_volume_profile = None
    
    # Track signals across strikes for breadth
    signal_bars = []
    
    for i in range(len(premium_1m)):
        current_time = times_1m[i]
        if current_time < entry_time:
            continue
            
        # Find corresponding 5min bar index
        idx_5m = times_5m.index(current_time) if current_time in times_5m else -1
        
        if idx_5m != -1 and idx_5m >= 5:
            # Calculate metrics on 5min data
            window_premiums = premium_5m[max(0, idx_5m-5):idx_5m+1]
            window_volumes = volume_5m[max(0, idx_5m-5):idx_5m+1]
            
            # Smoothness (linearity of move)
            net_change = window_premiums[-1] - window_premiums[0]
            total_volatility = sum(abs(window_premiums[j] - window_premiums[j-1]) for j in range(1, len(window_premiums)))
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
            
            # Volume confirmation (volume-weighted price movement)
            if len(window_volumes) > 0:
                avg_volume = np.mean(window_volumes[-3:])
                volume_trend = window_volumes[-1] > avg_volume * 1.2  # Volume spike confirmation
            else:
                volume_trend = False
            
            # Count down candles in last 6 bars (1min)
            if i >= 5:
                down_candles = sum(1 for j in range(i-5, i+1) if close_1m[j] < open_1m[j])
            else:
                down_candles = 0
            
            # Check for recent spike
            if i >= 6:
                max_premium = max(premium_1m[max(0, i-6):i+1])
                spike_pct = ((max_premium - premium_1m[i]) / premium_1m[i]) * 100 if premium_1m[i] > 0 else 0
            else:
                spike_pct = 0
            
            # RSI on 1min premium
            rsi = calculate_rsi(premium_1m[:i+1], 14)
            
            # Premium velocity ratio
            if i >= 6:
                recent_speed = (premium_1m[i] - premium_1m[i-1]) if i > 0 else 0
                avg_speed_6 = (premium_1m[i] - premium_1m[max(0, i-6)]) / 6 if i >= 6 else recent_speed
                pvr = abs(recent_speed / avg_speed_6) if avg_speed_6 != 0 else 1
            else:
                pvr = 1
            
            # Candle body analysis
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
            
            # Entry conditions - PARABOLIC MOVE DETECTION
            if not active:
                # All conditions must be met for entry
                entry_conditions = (
                    smoothness >= smooth_threshold and
                    speed <= entry_speed_threshold and  # Negative speed
                    delta_speed <= accel_threshold and  # Accelerating downward
                    convexity < 0 and  # Concave down (parabolic shape)
                    volume_trend and  # Volume confirmation
                    down_candles >= min_down_candles and  # Minimum down candles
                    spike_pct <= max_spike_pct and  # No recent spike
                    rsi > rsi_oversold and  # Not extremely oversold yet
                    pvr > pvr_ratio  # Momentum still strong
                )
                
                if entry_conditions and current_time <= "14:45":
                    active = {
                        'entry_price': premium_1m[i] - slip,
                        'entry_time': current_time,
                        'entry_bar': i,
                        'tsl': (premium_1m[i] - slip) + sl_points,
                        'entry_speed': speed,
                        'entry_rsi': rsi,
                        'convexity': convexity
                    }
                    entry_bar_speed = speed
                    entry_volume_profile = volume_trend
                    signal_bars.append({'time': current_time, 'strike': offset})
            
            # Exit conditions - MOMENTUM FADING
            elif active:
                # Update trailing stop loss (max 15 points from entry)
                if active['entry_price'] - premium_1m[i] >= 15:
                    active['tsl'] = min(active['tsl'], active['entry_price'] - 5)
                
                # Check exit conditions
                exit_signal = False
                exit_reason = ""
                
                # 1. Speed deceleration (momentum fading)
                if speed > (entry_bar_speed + exit_deceleration):
                    exit_signal = True
                    exit_reason = "Speed deceleration"
                
                # 2. RSI crossing above oversold
                elif rsi > rsi_oversold and active['entry_rsi'] <= rsi_oversold:
                    exit_signal = True
                    exit_reason = "RSI exhaustion"
                
                # 3. Premium velocity ratio dropping
                elif pvr < pvr_ratio:
                    exit_signal = True
                    exit_reason = "Velocity decay"
                
                # 4. Candle body exhaustion
                elif avg_body_ratio < body_ratio_exhaust:
                    exit_signal = True
                    exit_reason = "Body exhaustion"
                
                # 5. Speed turns positive (original exit)
                elif speed > 0:
                    exit_signal = True
                    exit_reason = "Speed reversal"
                
                # 6. Stop loss hit
                elif premium_1m[i] + slip >= active['tsl']:
                    exit_signal = True
                    exit_reason = "Stop loss"
                
                # 7. End of day
                elif current_time == "15:25":
                    exit_signal = True
                    exit_reason = "End of day"
                
                if exit_signal:
                    pnl = (active['entry_price'] - (premium_1m[i] + slip)) * 130  # 2 lots = 130 quantity
                    trades.append({
                        'pnl': pnl,
                        'entry_time': active['entry_time'],
                        'exit_time': current_time,
                        'entry_price': active['entry_price'],
                        'exit_price': premium_1m[i] + slip,
                        'exit_reason': exit_reason,
                        'bars_held': i - active['entry_bar'],
                        'convexity': active['convexity']
                    })
                    active = None
    
    return trades

def run_grid_search(master_data):
    """Run comprehensive grid search with all parameter combinations"""
    results = []
    total_combinations = (len(ENTRY_TIMES) * len(SMOOTH_RANGE) * len(ENTRY_SPEEDS) * len(SL_RANGE) *
                         len(ACCELERATION_THRESHOLD) * len(MIN_DOWN_CANDLES) * len(MAX_HIGH_SPIKE) *
                         len(MOMENTUM_BREADTH) * len(SPEED_DECELERATION) * len(RSI_OVERSOLD) *
                         len(PVR_RATIO) * len(BODY_RATIO))
    
    print(f"Total combinations to test: {total_combinations}")
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
                                                    if combination_count % 1000 == 0:
                                                        print(f"Processed {combination_count}/{total_combinations} combinations")
                                                    
                                                    all_trades = []
                                                    for date, date_data in master_data.items():
                                                        # Track signals across strikes for breadth confirmation
                                                        strike_signals = []
                                                        
                                                        for strike, strike_info in date_data['strikes'].items():
                                                            trades = enhanced_simulate(
                                                                strike_info, entry_time, smooth, entry_speed, sl,
                                                                accel, down_candles, max_spike, breadth,
                                                                decel, rsi_os, pvr, body_ratio
                                                            )
                                                            if trades:
                                                                strike_signals.append(len(trades))
                                                                all_trades.extend(trades)
                                                        
                                                        # Breadth check: require minimum strikes showing signals
                                                        if len(strike_signals) < breadth:
                                                            all_trades = []  # Invalid setup, clear trades
                                                            break
                                                    
                                                    if len(all_trades) >= 3:  # Minimum trades required
                                                        # Calculate metrics
                                                        pnls = [t['pnl'] for t in all_trades]
                                                        gross_pnl = sum(pnls)
                                                        total_trades = len(pnls)
                                                        tax_impact = total_trades * 200  # ₹200 per trade
                                                        net_pnl = gross_pnl - tax_impact
                                                        win_rate = (sum(1 for p in pnls if p > 0) / total_trades) * 100
                                                        
                                                        # Drawdown calculation
                                                        cumulative = np.cumsum(pnls)
                                                        peak = np.maximum.accumulate(cumulative)
                                                        drawdown = peak - cumulative
                                                        max_drawdown = np.max(drawdown)
                                                        
                                                        # Sharpe ratio (assuming 0% risk-free rate)
                                                        returns = np.array(pnls) / 500000  # Return per trade relative to capital
                                                        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
                                                        
                                                        # Profit factor
                                                        gross_profit = sum(p for p in pnls if p > 0)
                                                        gross_loss = abs(sum(p for p in pnls if p < 0))
                                                        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                                                        
                                                        # Average trade metrics
                                                        avg_win = gross_profit / sum(1 for p in pnls if p > 0) if sum(1 for p in pnls if p > 0) > 0 else 0
                                                        avg_loss = gross_loss / sum(1 for p in pnls if p < 0) if sum(1 for p in pnls if p < 0) > 0 else 0
                                                        
                                                        # Exit reason distribution
                                                        exit_reasons = {}
                                                        for t in all_trades:
                                                            reason = t['exit_reason']
                                                            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                                                        
                                                        # Composite score
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
                                                                'sharpe_ratio': sharpe,
                                                                'profit_factor': profit_factor,
                                                                'avg_win': avg_win,
                                                                'avg_loss': avg_loss,
                                                                'exit_reasons': exit_reasons,
                                                                'score': score
                                                            }
                                                        })
    
    # Sort by score
    results.sort(key=lambda x: x['metrics']['score'], reverse=True)
    return results

def generate_html_dashboard(results, master_data):
    """Generate comprehensive HTML dashboard with all visualizations"""
    top_results = results[:50]  # Top 50 configurations
    
    # Prepare data for JSON serialization
    dashboard_data = {
        'top_results': [
            {
                'rank': i+1,
                'score': round(r['metrics']['score'], 2),
                'params': r['params'],
                'metrics': {
                    'net_pnl': round(r['metrics']['net_pnl'], 2),
                    'gross_pnl': round(r['metrics']['gross_pnl'], 2),
                    'total_trades': r['metrics']['total_trades'],
                    'win_rate': round(r['metrics']['win_rate'], 2),
                    'max_drawdown': round(r['metrics']['max_drawdown'], 2),
                    'sharpe_ratio': round(r['metrics']['sharpe_ratio'], 2),
                    'profit_factor': round(r['metrics']['profit_factor'], 2),
                    'avg_win': round(r['metrics']['avg_win'], 2),
                    'avg_loss': round(r['metrics']['avg_loss'], 2),
                    'exit_reasons': r['metrics']['exit_reasons']
                },
                'sample_trades': [
                    {
                        'entry_time': t['entry_time'],
                        'exit_time': t['exit_time'],
                        'pnl': round(t['pnl'], 2),
                        'exit_reason': t['exit_reason'],
                        'bars_held': t['bars_held']
                    } for t in r['trades'][:20]  # First 20 trades as sample
                ]
            } for i, r in enumerate(top_results)
        ],
        'best_config': {
            'params': top_results[0]['params'] if top_results else {},
            'metrics': top_results[0]['metrics'] if top_results else {}
        }
    }
    
    json_data = json.dumps(dashboard_data)
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parabolic Move Detector - Backtest Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 { color: #667eea; margin-bottom: 10px; }
        .header p { color: #666; font-size: 14px; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-card h3 { font-size: 12px; color: #888; margin-bottom: 5px; }
        .metric-card .value { font-size: 24px; font-weight: bold; color: #667eea; }
        .metric-card .unit { font-size: 12px; color: #888; margin-left: 5px; }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .table-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #e5e7eb;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #f0f0f0;
        }
        tr:hover { background: #f8f9fa; }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-success { background: #d1fae5; color: #065f46; }
        .badge-warning { background: #fed7aa; color: #92400e; }
        .badge-info { background: #dbeafe; color: #1e40af; }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            background: white;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
        }
        .tab:hover { transform: translateY(-1px); }
        .tab.active { background: #667eea; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .parameter-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
        }
        .parameter-item {
            background: #f8f9fa;
            padding: 8px;
            border-radius: 5px;
            font-size: 12px;
        }
        .parameter-item strong { color: #667eea; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeIn 0.5s ease-out; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header fade-in">
            <h1>📊 Parabolic Move Detector v5.0</h1>
            <p>Advanced mean reversion strategy with acceleration, convexity, and momentum confirmation</p>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('best')">🏆 Best Configuration</div>
            <div class="tab" onclick="switchTab('ranking')">📈 Strategy Ranking</div>
            <div class="tab" onclick="switchTab('analysis')">📉 Performance Analysis</div>
            <div class="tab" onclick="switchTab('trades')">💼 Trade Explorer</div>
        </div>

        <div id="best" class="tab-content active">
            <div class="metrics-grid" id="bestMetrics"></div>
            <div class="chart-container">
                <h3>📊 Exit Reason Distribution</h3>
                <div id="exitReasonChart"></div>
            </div>
            <div class="parameter-list" id="bestParams"></div>
        </div>

        <div id="ranking" class="tab-content">
            <div class="table-container">
                <table id="rankingTable">
                    <thead>
                        <tr><th>Rank</th><th>Score</th><th>Entry Time</th><th>Smooth%</th><th>Entry Spd</th><th>SL</th>
                            <th>Net P&L</th><th>Trades</th><th>Win%</th><th>Sharpe</th><th>Max DD</th><th>Profit Factor</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>

        <div id="analysis" class="tab-content">
            <div class="chart-container">
                <h3>📈 Risk-Return Profile</h3>
                <div id="riskReturnChart"></div>
            </div>
            <div class="chart-container">
                <h3>🎯 Parameter Sensitivity</h3>
                <div id="sensitivityChart"></div>
            </div>
        </div>

        <div id="trades" class="tab-content">
            <div class="table-container">
                <h3>💼 Sample Trades - Best Strategy</h3>
                <table id="tradesTable">
                    <thead><tr><th>Entry Time</th><th>Exit Time</th><th>P&L (₹)</th><th>Exit Reason</th><th>Bars Held</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const data = """ + json_data + """;
        
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
        
        function renderBestConfig() {
            const best = data.best_config;
            const metrics = best.metrics;
            if (!metrics) return;
            
            document.getElementById('bestMetrics').innerHTML = `
                <div class="metric-card"><h3>Net P&L</h3><div class="value ${metrics.net_pnl > 0 ? 'positive' : 'negative'}">₹${metrics.net_pnl.toLocaleString()}</div></div>
                <div class="metric-card"><h3>Win Rate</h3><div class="value positive">${metrics.win_rate}%</div></div>
                <div class="metric-card"><h3>Total Trades</h3><div class="value">${metrics.total_trades}</div></div>
                <div class="metric-card"><h3>Sharpe Ratio</h3><div class="value">${metrics.sharpe_ratio}</div></div>
                <div class="metric-card"><h3>Profit Factor</h3><div class="value">${metrics.profit_factor}</div></div>
                <div class="metric-card"><h3>Max Drawdown</h3><div class="value negative">₹${metrics.max_drawdown.toLocaleString()}</div></div>
                <div class="metric-card"><h3>Avg Win</h3><div class="value positive">₹${metrics.avg_win.toLocaleString()}</div></div>
                <div class="metric-card"><h3>Avg Loss</h3><div class="value negative">₹${Math.abs(metrics.avg_loss).toLocaleString()}</div></div>
            `;
            
            const params = best.params;
            document.getElementById('bestParams').innerHTML = `
                <div class="parameter-item"><strong>Entry Time:</strong> ${params.entry_time}</div>
                <div class="parameter-item"><strong>Smoothness:</strong> ${params.smoothness}%</div>
                <div class="parameter-item"><strong>Entry Speed:</strong> ${params.entry_speed}</div>
                <div class="parameter-item"><strong>Stop Loss:</strong> ${params.sl_points} pts</div>
                <div class="parameter-item"><strong>Acceleration:</strong> ${params.acceleration}</div>
                <div class="parameter-item"><strong>Min Down Candles:</strong> ${params.down_candles}</div>
                <div class="parameter-item"><strong>Max Spike:</strong> ${params.max_spike}%</div>
                <div class="parameter-item"><strong>Momentum Breadth:</strong> ${params.breadth}</div>
                <div class="parameter-item"><strong>Exit Deceleration:</strong> ${params.exit_deceleration}</div>
                <div class="parameter-item"><strong>RSI Oversold:</strong> ${params.rsi_oversold}</div>
                <div class="parameter-item"><strong>PVR Ratio:</strong> ${params.pvr_ratio}</div>
                <div class="parameter-item"><strong>Body Ratio:</strong> ${params.body_ratio}</div>
            `;
            
            // Exit reason pie chart
            const exitReasons = metrics.exit_reasons || {};
            const exitData = [{
                values: Object.values(exitReasons),
                labels: Object.keys(exitReasons),
                type: 'pie',
                hole: 0.4,
                marker: { colors: ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'] }
            }];
            
            Plotly.newPlot('exitReasonChart', exitData, {
                title: 'Exit Reason Distribution',
                height: 400,
                showlegend: true
            });
        }
        
        function renderRanking() {
            const tbody = document.querySelector('#rankingTable tbody');
            tbody.innerHTML = data.top_results.map(r => `
                <tr>
                    <td><strong>#${r.rank}</strong></td>
                    <td><span class="badge badge-success">${r.score}</span></td>
                    <td>${r.params.entry_time}</td>
                    <td>${r.params.smoothness}%</td>
                    <td>${r.params.entry_speed}</td>
                    <td>${r.params.sl_points}</td>
                    <td class="${r.metrics.net_pnl > 0 ? 'positive' : 'negative'}">₹${r.metrics.net_pnl.toLocaleString()}</td>
                    <td>${r.metrics.total_trades}</td>
                    <td class="positive">${r.metrics.win_rate}%</td>
                    <td>${r.metrics.sharpe_ratio}</td>
                    <td class="negative">₹${r.metrics.max_drawdown.toLocaleString()}</td>
                    <td>${r.metrics.profit_factor}</td>
                </tr>
            `).join('');
        }
        
        function renderAnalysis() {
            // Risk-Return Scatter Plot
            const riskReturn = data.top_results.map(r => ({
                x: r.metrics.max_drawdown,
                y: r.metrics.net_pnl / 500000 * 100,  // ROC %
                text: `Score: ${r.score}<br>Win Rate: ${r.metrics.win_rate}%`,
                mode: 'markers',
                marker: { size: 10, color: r.metrics.sharpe_ratio, colorscale: 'Viridis', showscale: true }
            }));
            
            Plotly.newPlot('riskReturnChart', riskReturn, {
                title: 'Risk vs Return - Each Point is a Strategy Configuration',
                xaxis: { title: 'Max Drawdown (₹)' },
                yaxis: { title: 'Return on Capital (%)' },
                hovermode: 'closest',
                height: 500
            });
            
            // Parameter Sensitivity (average performance by entry time and smoothness)
            const paramData = {};
            data.top_results.forEach(r => {
                const key = `${r.params.entry_time}|${r.params.smoothness}`;
                if (!paramData[key]) paramData[key] = { scores: [], pnls: [] };
                paramData[key].scores.push(r.score);
                paramData[key].pnls.push(r.metrics.net_pnl);
            });
            
            const sensitivityData = [{
                z: Object.values(paramData).map(v => v.scores.reduce((a,b) => a+b, 0) / v.scores.length),
                x: [...new Set(data.top_results.map(r => r.params.entry_time))],
                y: [...new Set(data.top_results.map(r => r.params.smoothness))],
                type: 'heatmap',
                colorscale: 'RdYlGn',
                reversescale: true
            }];
            
            Plotly.newPlot('sensitivityChart', sensitivityData, {
                title: 'Strategy Performance Heatmap (Score by Entry Time & Smoothness)',
                xaxis: { title: 'Entry Time' },
                yaxis: { title: 'Smoothness (%)' },
                height: 400
            });
        }
        
        function renderTrades() {
            if (data.top_results.length === 0) return;
            const trades = data.top_results[0].sample_trades || [];
            const tbody = document.querySelector('#tradesTable tbody');
            tbody.innerHTML = trades.map(t => `
                <tr>
                    <td>${t.entry_time}</td>
                    <td>${t.exit_time}</td>
                    <td class="${t.pnl > 0 ? 'positive' : 'negative'}">₹${t.pnl.toLocaleString()}</td>
                    <td><span class="badge badge-info">${t.exit_reason}</span></td>
                    <td>${t.bars_held} mins</td>
                </tr>
            `).join('');
        }
        
        // Initialize all visualizations
        renderBestConfig();
        renderRanking();
        renderAnalysis();
        renderTrades();
    </script>
</body>
</html>"""
    
    with open("parabolic_dashboard.html", "w") as f:
        f.write(html_template)
    print("Dashboard generated: parabolic_dashboard.html")

if __name__ == "__main__":
    print("🚀 Starting Parabolic Move Detection Backtest")
    print("=" * 60)
    
    print("📊 Preparing data...")
    data = prepare_data()
    
    if data:
        print(f"✅ Data prepared for {len(data)} dates")
        print("🔍 Running grid search...")
        results = run_grid_search(data)
        
        if results:
            print(f"✅ Found {len(results)} valid configurations")
            print("📈 Generating dashboard...")
            generate_html_dashboard(results, data)
            print("✨ Backtest complete! Open parabolic_dashboard.html to view results")
            
            # Print top result summary
            best = results[0]
            print("\n🏆 BEST CONFIGURATION:")
            print(f"   Score: {best['metrics']['score']:.2f}")
            print(f"   Net P&L: ₹{best['metrics']['net_pnl']:,.2f}")
            print(f"   Win Rate: {best['metrics']['win_rate']:.1f}%")
            print(f"   Total Trades: {best['metrics']['total_trades']}")
            print(f"   Sharpe Ratio: {best['metrics']['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: ₹{best['metrics']['max_drawdown']:,.2f}")
            print("\n📋 Best Parameters:")
            for key, value in best['params'].items():
                print(f"   {key}: {value}")
        else:
            print("❌ No valid configurations found. Try adjusting parameters.")
    else:
        print("❌ No data found. Please check data_cache folder and API connection.")
