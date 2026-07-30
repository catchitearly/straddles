import os
import math
import json
import logging
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from fyers_apiv3 import fyersModel
from scipy.stats import norm
from scipy.optimize import brentq
import sys

# --- CONFIGURATION ---
CLIENT_ID = os.getenv("CLIENT_ID")
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TARGET_DATE = os.getenv("TARGET_DATE") or datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
EXPIRY = os.getenv("OPTION_EXPIRY_CODE", "26804")
EXPIRY_DATE = os.getenv("OPTION_EXPIRY_DATE", "2026-08-04")
EXPIRY_TIME = "15:30"
RISK_FREE_RATE = 0.1
SPOT_SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_STEP = 100
FALLBACK_ATM = 24200
CANDLE_INTERVAL_MINUTES = 5
THETA_WINDOW_MINUTES = 15
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

# Straddle price / VWAP are computed on a rolling multi-day window instead of
# resetting fresh each session — this gives the rolling VWAP more volume
# history to average over, which smooths out the noisy first minutes of each
# session and produces a more stable, less choppy VWAP line for today.
VWAP_LOOKBACK_TRADING_DAYS = 3       # trading days INCLUDING today used for the rolling straddle VWAP
VWAP_LOOKBACK_CALENDAR_BUFFER = 12   # calendar days to request from the API to safely cover N trading days (weekends/holidays)

LOT_SIZE = 65
GEX_STRIKE_COUNT = 40
GEX_SCALE = 1e7
GEX_SPOT_RANGE_POINTS = 1000
GEX_SPOT_STEP = 25

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

GEX_HISTORY_FILE = os.path.join(OUTPUT_DIR, f"gex_history_{TARGET_DATE}.json")
GEX_HISTORY_DOCS_FILE = os.path.join("docs", f"gex_history_{TARGET_DATE}.json")

# Weekly expiry codes to combine for the "Combined GEX" tab.
# Format: Fyers weekly symbol codes (YYMDD) or monthly codes (YYMMM).
# Update each week as expiries roll.
GEX_MULTI_EXPIRY_CODES = ["26JUL", "26804", "26AUG"]
GEX_COMBINED_HISTORY_FILE = os.path.join(OUTPUT_DIR, f"gex_combined_history_{TARGET_DATE}.json")
GEX_COMBINED_HISTORY_DOCS_FILE = os.path.join("docs", f"gex_combined_history_{TARGET_DATE}.json")

# =============================================================================
# BASKET STRATEGY: short a 3-strike straddle basket (N-100, N, N+100) when its
# combined premium is below its combined VWAP, cover on VWAP reclaim, a
# trailing ATR(14) stop, or forced EOD square-off.
# =============================================================================
BASKET_OFFSETS              = [-100, 0, 100]   # strikes relative to N
BASKET_STRIKE_SAMPLE_TIME   = "09:40"          # NIFTY close sampled at/after this time -> rounded to N
BASKET_ENTRY_START_TIME     = "09:45"          # earliest an entry can be taken
BASKET_ENTRY_CUTOFF_TIME    = "14:30"          # no new entries after this time
BASKET_EOD_EXIT_TIME        = "15:15"          # force-close any open trade at this time
BASKET_ATR_PERIOD           = 14               # ATR lookback, in candles (5-min candles -> ~70 min lookback)
BASKET_ATR_MULTIPLIER       = 2.0              # trailing stop = trailing extreme +/- multiplier * ATR
BASKET_MAX_TRADES_PER_DAY   = 3

BASKET_STATE_FILE      = os.path.join(OUTPUT_DIR, f"basket_state_{TARGET_DATE}.json")
BASKET_STATE_DOCS_FILE = os.path.join("docs", f"basket_state_{TARGET_DATE}.json")

# Dashboard Styling
BG = "#0b0f1a"
CARD = "#111b27"
BORDER = "#1e2d40"
TEXT = "#e2e8f0"
MUTED = "#64748b"
ACCENT = "#00e5b0"
BLUE = "#38bdf8"
RED = "#ff4560"
EMA_COLOR = "#fbbf24"
STRIKE_COLORS = ["#38bdf8", "#a78bfa", "#f97316", "#ff4560", "#fbbf24", "#00e5b0", "#ec4899", "#84cc16", "#64748b"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUN_TIMESTAMP = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

EXPIRY_DT = datetime.combine(
    datetime.strptime(EXPIRY_DATE, "%Y-%m-%d").date(),
    dtime.fromisoformat(EXPIRY_TIME),
    tzinfo=ZoneInfo("Asia/Kolkata"),
)

# --- TELEGRAM ---

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping message.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("✓ Telegram message sent.")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_document(file_path, caption=""):
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
    trend_emoji = {"UP": "📈", "DOWN": "📉", "FLAT": "➡️"}
    medal = ["🥇", "🥈", "🥉"]
    lines = [
        f"<b>📊 STRADDLE ANALYSER — {TARGET_DATE}</b>",
        f"<code>ATM: {atm} | Strikes: {successful_fetches}/{total_strikes} | {RUN_TIMESTAMP}</code>",
        "", "<b>🏆 SMOOTHNESS RANKINGS</b>",
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
    top = rankings[0]
    lines += ["", f"✅ <b>Best Strike:</b> {top['strike']} (Smoothness: {top['smoothness']}%)", "📎 Full dashboard attached below."]
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

# --- OPTIONS PRICING: IV & GREEKS (Black-Scholes) ---

def _time_to_expiry_years(candle_time, expiry_dt=None):
    """candle_time is a tz-naive Asia/Kolkata timestamp. Returns years to expiry (>=0).
    expiry_dt defaults to the main EXPIRY_DT; multi-expiry GEX passes each expiry's own datetime.
    """
    expiry_dt = expiry_dt or EXPIRY_DT
    aware = candle_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    seconds = (expiry_dt - aware).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.0 * 24.0 * 3600.0)

def _bs_d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, opt_type):
    intrinsic = max(S - K, 0.0) if opt_type == "CE" else max(K - S, 0.0)
    if T <= 0 or sigma <= 0:
        return intrinsic
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, opt_type):
    if T <= 0 or price is None or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S - K, 0.0) if opt_type == "CE" else max(K - S, 0.0)
    if price < intrinsic - 0.05:
        return float("nan")
    try:
        lo, hi = 1e-4, 5.0
        f_lo = bs_price(S, K, T, r, lo, opt_type) - price
        f_hi = bs_price(S, K, T, r, hi, opt_type) - price
        if f_lo * f_hi > 0:
            return float("nan")
        return brentq(lambda sig: bs_price(S, K, T, r, sig, opt_type) - price, lo, hi, maxiter=100)
    except Exception:
        return float("nan")

def bs_gamma_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or sigma != sigma:
        return 0.0, 0.0
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T) / 100.0
    return gamma, vega

def bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or sigma != sigma:
        return 0.0
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    if opt_type == "CE":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0

def bs_theta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or sigma != sigma:
        return 0.0
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    if opt_type == "CE":
        annual = -(S * pdf * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        annual = -(S * pdf * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return annual / 365.0

def compute_greeks_row(row, strike):
    S = row.get("spot")
    t_years = _time_to_expiry_years(row["time"])
    if S is None or S != S or S <= 0:
        return pd.Series({
            "iv_ce": float("nan"), "iv_pe": float("nan"), "iv_pct": float("nan"),
            "gamma_total": 0.0, "vega_total": 0.0,
            "theta_total": 0.0, "theta_ce": 0.0, "theta_pe": 0.0,
            "delta_ce": 0.0, "delta_pe": 0.0, "delta_total": 0.0,
        })
    iv_ce = implied_vol(row["close_x"], S, strike, t_years, RISK_FREE_RATE, "CE")
    iv_pe = implied_vol(row["close_y"], S, strike, t_years, RISK_FREE_RATE, "PE")
    gamma_ce, vega_ce = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_ce)
    gamma_pe, vega_pe = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_pe)
    theta_ce = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_ce, "CE")
    theta_pe = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_pe, "PE")
    delta_ce = bs_delta(S, strike, t_years, RISK_FREE_RATE, iv_ce, "CE")
    delta_pe = bs_delta(S, strike, t_years, RISK_FREE_RATE, iv_pe, "PE")
    ivs = [v for v in (iv_ce, iv_pe) if v == v]
    iv_pct = (sum(ivs) / len(ivs) * 100.0) if ivs else float("nan")
    _n0 = lambda v: v if v == v else 0.0
    return pd.Series({
        "iv_ce": iv_ce, "iv_pe": iv_pe, "iv_pct": iv_pct,
        "gamma_total": _n0(gamma_ce) + _n0(gamma_pe),
        "vega_total":  _n0(vega_ce)  + _n0(vega_pe),
        "theta_total": _n0(theta_ce) + _n0(theta_pe),
        "theta_ce":    _n0(theta_ce),   # CE leg theta per calendar day
        "theta_pe":    _n0(theta_pe),   # PE leg theta per calendar day
        "delta_ce":    _n0(delta_ce),   # CE leg delta
        "delta_pe":    _n0(delta_pe),   # PE leg delta
        "delta_total": _n0(delta_ce) + _n0(delta_pe),  # Straddle net delta (CE + PE)
    })

# =============================================================================
# GAMMA EXPOSURE (GEX) & GAMMA FLIP
# =============================================================================
#
# GEX convention: calls → POSITIVE, puts → NEGATIVE.
# Net GEX > 0 = dealers long gamma → dampen moves (pinned).
# Net GEX < 0 = dealers short gamma → amplify moves (trending).
#
# Formula (institutional / SpotGamma standard):
#   GEX_Call(K) = +OI_CE × Γ_CE × LotSize × S² × 0.01   (₹ value of 1% move)
#   GEX_Put(K)  = −OI_PE × Γ_PE × LotSize × S² × 0.01
#
# OI source: fyers.optionchain() live snapshot — Fyers history API has no OI.
# GEX is therefore a per-run snapshot; the time series is built by appending
# one point per GitHub Actions run into a JSON file committed back to the repo.
#
# MULTI-EXPIRY: GEX_MULTI_EXPIRY_CODES lists Fyers expiry codes to aggregate.
# Each expiry uses its own time-to-expiry (T) when recomputing gamma on the
# flip-level grid, giving a true combined dealer gamma picture.

def fetch_option_chain(fyers, strike_count=GEX_STRIKE_COUNT, timestamp="", _quiet=False):
    """Fetch live option-chain snapshot (OI + LTP per strike) from Fyers.
    Returns (chain_df, spot_price, expiry_data) or (None, None, None) on failure.
    chain_df columns: strike, ltp_ce, ltp_pe, oi_ce, oi_pe.
    """
    try:
        resp = fyers.optionchain(data={"symbol": SPOT_SYMBOL, "strikecount": strike_count, "timestamp": timestamp})
    except Exception as e:
        logger.error(f"Option chain fetch failed (timestamp={timestamp!r}): {e}")
        return None, None, None

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.error(f"Option chain error response (timestamp={timestamp!r}): {resp}")
        return None, None, None

    data = resp.get("data") or {}
    chain = data.get("optionsChain") or []
    expiry_data = data.get("expiryData") or []
    if not chain:
        if not _quiet:
            logger.warning(f"Option chain returned no rows. Raw response (truncated): {json.dumps(resp)[:800]}")
        return None, None, expiry_data

    def _get(rec, *keys):
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
        return None

    spot_price = None
    rows = []
    for rec in chain:
        opt_type = _get(rec, "option_type", "optionType", "opt_type")
        if opt_type not in ("CE", "PE"):
            maybe_spot = _get(rec, "ltp", "lp")
            if maybe_spot:
                spot_price = maybe_spot
            continue
        strike = _get(rec, "strike_price", "strikePrice", "strike")
        ltp = _get(rec, "ltp", "lp") or 0.0
        oi  = _get(rec, "oi", "openInterest") or 0
        if strike is None:
            continue
        rows.append({"strike": strike, "option_type": opt_type, "ltp": ltp, "oi": oi})

    if not rows:
        if not _quiet:
            logger.warning(f"Could not parse any option chain rows. Sample: {json.dumps(chain[:2])}")
        return None, None, expiry_data

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="strike", values=["ltp", "oi"], columns="option_type", aggfunc="first")
    pivot.columns = [f"{val}_{opt.lower()}" for val, opt in pivot.columns]
    pivot = pivot.reset_index()
    for col in ["ltp_ce", "ltp_pe", "oi_ce", "oi_pe"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot, spot_price, expiry_data

def fetch_available_expiries(fyers):
    """Discover available expiries from Fyers (date + epoch timestamp).
    Returns list of dicts: {"date": <raw date string>, "epoch": <str/int>}.
    """
    _, _, expiry_data = fetch_option_chain(fyers, strike_count=1, timestamp="", _quiet=True)
    out = []
    for rec in (expiry_data or []):
        date_val  = rec.get("date")  if isinstance(rec, dict) else None
        epoch_val = rec.get("expiry") if isinstance(rec, dict) else None
        if date_val is not None and epoch_val is not None:
            out.append({"date": date_val, "epoch": epoch_val})
    return out

MONTH_ABBREVS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def parse_expiry_code(code):
    """Parse a Fyers expiry code into (kind, ...).
    Weekly  YYMDD  → ("weekly", date)        e.g. "26714" → 2026-07-14
    Monthly YYMMM  → ("monthly", year, month) e.g. "26JUL" → July 2026
    """
    code = code.strip().upper()
    if len(code) == 5 and code[2:].isalpha() and code[2:] in MONTH_ABBREVS:
        yy = int(code[:2])
        return ("monthly", 2000 + yy, MONTH_ABBREVS[code[2:]])
    month_map = {**{str(m): m for m in range(1, 10)}, "O": 10, "N": 11, "D": 12}
    yy    = int(code[:2])
    m_char = code[2]
    dd    = int(code[3:])
    month = month_map.get(m_char)
    if month is None:
        raise ValueError(f"Could not parse expiry code {code!r}")
    return ("weekly", datetime(2000 + yy, month, dd).date())

def _match_expiry_epoch(parsed, available_expiries):
    """Match a parsed expiry code to Fyers' own expiry list.
    Returns (resolved_date, epoch) or (None, None).
    Monthly codes match the LATEST available date in that calendar month.
    """
    parsed_dates = []
    for rec in available_expiries:
        raw = str(rec["date"])
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
            try:
                parsed_dates.append((datetime.strptime(raw, fmt).date(), rec["epoch"]))
                break
            except ValueError:
                continue

    kind = parsed[0]
    if kind == "weekly":
        target_date = parsed[1]
        for d, epoch in parsed_dates:
            if d == target_date:
                return d, epoch
        return None, None
    else:
        _, year, month = parsed
        candidates = sorted((d, e) for d, e in parsed_dates if d.year == year and d.month == month)
        if not candidates:
            return None, None
        return candidates[-1]

def compute_gex_snapshot(chain_df, spot, expiry_dt=None, as_of=None):
    """Per-strike GEX using BS gamma solved from each leg's live LTP.
    GEX_Call = +OI_CE × Γ_CE × LotSize × S² × 0.01
    GEX_Put  = −OI_PE × Γ_PE × LotSize × S² × 0.01
    Returns chain_df with iv/gamma/gex columns added.
    """
    as_of   = as_of or datetime.now(ZoneInfo("Asia/Kolkata"))
    t_years = _time_to_expiry_years(as_of.replace(tzinfo=None), expiry_dt)
    records = []
    for _, row in chain_df.iterrows():
        strike  = row["strike"]
        iv_ce   = implied_vol(row["ltp_ce"], spot, strike, t_years, RISK_FREE_RATE, "CE")
        iv_pe   = implied_vol(row["ltp_pe"], spot, strike, t_years, RISK_FREE_RATE, "PE")
        gamma_ce, _ = bs_gamma_vega(spot, strike, t_years, RISK_FREE_RATE, iv_ce)
        gamma_pe, _ = bs_gamma_vega(spot, strike, t_years, RISK_FREE_RATE, iv_pe)
        oi_ce = row["oi_ce"] if row["oi_ce"] == row["oi_ce"] else 0.0
        oi_pe = row["oi_pe"] if row["oi_pe"] == row["oi_pe"] else 0.0
        # GEX formula (per document): OI × Γ × LotSize × S × 0.01
        # S × 0.01 = ₹ value of a 1% move in the underlying
        gex_ce =  gamma_ce * oi_ce * LOT_SIZE * spot * 0.01
        gex_pe = -gamma_pe * oi_pe * LOT_SIZE * spot * 0.01
        records.append({
            "strike": strike, "oi_ce": oi_ce, "oi_pe": oi_pe,
            "iv_ce": iv_ce, "iv_pe": iv_pe,
            "gamma_ce": gamma_ce, "gamma_pe": gamma_pe,
            "gex_ce": gex_ce, "gex_pe": gex_pe, "gex_net": gex_ce + gex_pe,
        })
    return pd.DataFrame(records)

def compute_gamma_flip(gex_df, spot, expiry_dt=None, as_of=None):
    """Scan a hypothetical spot grid, recomputing gamma at each level (IV fixed,
    OI fixed), and find where total GEX crosses zero.
    Returns (flip_level_or_None, [(spot, total_gex), ...]).
    """
    as_of   = as_of or datetime.now(ZoneInfo("Asia/Kolkata"))
    t_years = _time_to_expiry_years(as_of.replace(tzinfo=None), expiry_dt)
    grid    = np.arange(
        spot - GEX_SPOT_RANGE_POINTS,
        spot + GEX_SPOT_RANGE_POINTS + GEX_SPOT_STEP,
        GEX_SPOT_STEP
    )
    totals = []
    for s_hyp in grid:
        total = 0.0
        for _, row in gex_df.iterrows():
            strike = row["strike"]
            if row["iv_ce"] == row["iv_ce"]:
                g_ce, _ = bs_gamma_vega(s_hyp, strike, t_years, RISK_FREE_RATE, row["iv_ce"])
                total +=  g_ce * row["oi_ce"] * LOT_SIZE * s_hyp * 0.01
            if row["iv_pe"] == row["iv_pe"]:
                g_pe, _ = bs_gamma_vega(s_hyp, strike, t_years, RISK_FREE_RATE, row["iv_pe"])
                total -= g_pe * row["oi_pe"] * LOT_SIZE * s_hyp * 0.01
        totals.append(total)

    totals = np.array(totals)
    flip   = None
    for i in range(len(totals) - 1):
        if totals[i] == 0:
            flip = float(grid[i])
            break
        if totals[i] * totals[i + 1] < 0:
            frac = totals[i] / (totals[i] - totals[i + 1])
            flip = float(grid[i] + frac * (grid[i + 1] - grid[i]))
            break
    return flip, list(zip(grid.tolist(), totals.tolist()))

def compute_gamma_flip_multi(gex_records, spot, as_of=None):
    """Multi-expiry gamma flip: each record carries its own expiry_dt so that
    T is correct per expiry when recomputing gamma on the grid.
    gex_records: list of dicts with keys strike, expiry_dt, iv_ce, iv_pe, oi_ce, oi_pe.
    Returns (flip_level_or_None, [(spot, total_gex), ...]).
    """
    as_of       = as_of or datetime.now(ZoneInfo("Asia/Kolkata"))
    candle_time = as_of.replace(tzinfo=None)
    grid        = np.arange(
        spot - GEX_SPOT_RANGE_POINTS,
        spot + GEX_SPOT_RANGE_POINTS + GEX_SPOT_STEP,
        GEX_SPOT_STEP
    )
    totals = []
    for s_hyp in grid:
        total = 0.0
        for rec in gex_records:
            t_years = _time_to_expiry_years(candle_time, rec["expiry_dt"])
            if t_years <= 0:
                continue
            if rec["iv_ce"] == rec["iv_ce"]:
                g_ce, _ = bs_gamma_vega(s_hyp, rec["strike"], t_years, RISK_FREE_RATE, rec["iv_ce"])
                total +=  g_ce * rec["oi_ce"] * LOT_SIZE * s_hyp * 0.01
            if rec["iv_pe"] == rec["iv_pe"]:
                g_pe, _ = bs_gamma_vega(s_hyp, rec["strike"], t_years, RISK_FREE_RATE, rec["iv_pe"])
                total -= g_pe * rec["oi_pe"] * LOT_SIZE * s_hyp * 0.01
        totals.append(total)

    totals = np.array(totals)
    flip   = None
    for i in range(len(totals) - 1):
        if totals[i] == 0:
            flip = float(grid[i])
            break
        if totals[i] * totals[i + 1] < 0:
            frac = totals[i] / (totals[i] - totals[i + 1])
            flip = float(grid[i] + frac * (grid[i + 1] - grid[i]))
            break
    return flip, list(zip(grid.tolist(), totals.tolist()))

# --- GEX History persistence ---

def load_gex_history():
    path = GEX_HISTORY_DOCS_FILE if os.path.exists(GEX_HISTORY_DOCS_FILE) else GEX_HISTORY_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read GEX history ({path}): {e}")
    return []

def save_gex_history(history):
    for path in (GEX_HISTORY_FILE, GEX_HISTORY_DOCS_FILE):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f)
        except Exception as e:
            logger.warning(f"Could not write GEX history ({path}): {e}")

def load_combined_gex_history():
    path = GEX_COMBINED_HISTORY_DOCS_FILE if os.path.exists(GEX_COMBINED_HISTORY_DOCS_FILE) else GEX_COMBINED_HISTORY_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read combined GEX history ({path}): {e}")
    return []

def save_combined_gex_history(history):
    for path in (GEX_COMBINED_HISTORY_FILE, GEX_COMBINED_HISTORY_DOCS_FILE):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f)
        except Exception as e:
            logger.warning(f"Could not write combined GEX history ({path}): {e}")

def update_gex_and_get_history(fyers, fallback_spot):
    """Fetch a fresh option-chain snapshot (nearest expiry), compute GEX + flip
    + total CE/PE OI, append to persisted history log.
    Returns (history, latest_status) where latest_status is a dict describing
    this run's spot-vs-gamma-flip position (or None if unavailable).
    """
    history = load_gex_history()
    latest_status = None
    try:
        chain_df, chain_spot, _expiry_data = fetch_option_chain(fyers)
        spot = chain_spot or fallback_spot
        if chain_df is not None and spot:
            gex_df  = compute_gex_snapshot(chain_df, spot)
            net_gex = float(gex_df["gex_net"].sum())
            flip, _ = compute_gamma_flip(gex_df, spot)

            total_oi_ce = float(chain_df["oi_ce"].fillna(0).sum())
            total_oi_pe = float(chain_df["oi_pe"].fillna(0).sum())
            oi_diff     = total_oi_pe - total_oi_ce

            history.append({
                "time": RUN_TIMESTAMP, "spot": float(spot), "net_gex": net_gex, "flip": flip,
                "oi_ce": total_oi_ce, "oi_pe": total_oi_pe, "oi_diff": oi_diff,
            })
            save_gex_history(history)

            if flip is not None:
                position = "ABOVE" if spot > flip else ("BELOW" if spot < flip else "AT")
                latest_status = {
                    "spot": float(spot), "flip": flip, "position": position, "net_gex": net_gex,
                }

            logger.info(
                f"✓ GEX snapshot: spot={spot:.1f}  net_gex={net_gex/GEX_SCALE:.2f} Cr  flip={flip}  "
                f"OI(CE/PE)={total_oi_ce:.0f}/{total_oi_pe:.0f}  diff={oi_diff:.0f}"
            )
        else:
            logger.warning("Skipping GEX snapshot this run (no option chain / spot).")
    except Exception as e:
        logger.error(f"GEX computation failed: {e}", exc_info=True)
    return history, latest_status

def update_combined_gex_and_get_history(fyers, fallback_spot, expiry_codes=None):
    """Fetch option chains for multiple expiries, compute combined net GEX and
    flip level (true multi-expiry grid search), append to combined history log.
    Returns full day's combined history so far.
    """
    expiry_codes = expiry_codes or GEX_MULTI_EXPIRY_CODES
    history      = load_combined_gex_history()
    try:
        available = fetch_available_expiries(fyers)
        if not available:
            logger.warning("Could not discover available expiries — skipping combined GEX this run.")
            return history

        combined_records   = []
        per_expiry_net_gex = {}
        spot               = fallback_spot

        for code in expiry_codes:
            try:
                parsed = parse_expiry_code(code)
            except ValueError as e:
                logger.warning(f"Skipping expiry code {code!r}: {e}")
                continue

            resolved_date, epoch = _match_expiry_epoch(parsed, available)
            if epoch is None:
                logger.warning(
                    f"No matching Fyers expiry for code {code!r} (parsed={parsed}). "
                    f"Available: {[r['date'] for r in available]}"
                )
                continue

            chain_df, chain_spot, _ = fetch_option_chain(fyers, timestamp=epoch)
            if chain_df is None:
                logger.warning(f"No chain data for expiry {code} ({resolved_date}) — skipping.")
                continue
            spot = chain_spot or spot
            if not spot:
                continue

            expiry_dt = datetime.combine(
                resolved_date, dtime.fromisoformat(EXPIRY_TIME), tzinfo=ZoneInfo("Asia/Kolkata")
            )
            gex_df = compute_gex_snapshot(chain_df, spot, expiry_dt=expiry_dt)
            per_expiry_net_gex[code] = float(gex_df["gex_net"].sum())

            for _, row in gex_df.iterrows():
                combined_records.append({
                    "strike": row["strike"], "expiry_dt": expiry_dt,
                    "iv_ce": row["iv_ce"],  "iv_pe": row["iv_pe"],
                    "oi_ce": row["oi_ce"],  "oi_pe": row["oi_pe"],
                })
            logger.info(
                f"✓ Loaded expiry {code} ({resolved_date})  net={per_expiry_net_gex[code]/GEX_SCALE:.2f} Cr"
            )

        if not combined_records or not spot:
            logger.warning("Skipping combined GEX snapshot (no expiries loaded).")
            return history

        combined_net_gex   = sum(per_expiry_net_gex.values())
        combined_flip, _   = compute_gamma_flip_multi(combined_records, spot)

        history.append({
            "time": RUN_TIMESTAMP, "spot": float(spot),
            "net_gex": combined_net_gex, "flip": combined_flip,
            "per_expiry": per_expiry_net_gex,
        })
        save_combined_gex_history(history)
        logger.info(
            f"✓ Combined GEX: spot={spot:.1f}  net_gex={combined_net_gex/GEX_SCALE:.2f} Cr  "
            f"flip={combined_flip}  expiries={list(per_expiry_net_gex.keys())}"
        )
    except Exception as e:
        logger.error(f"Combined GEX computation failed: {e}", exc_info=True)
    return history

# =============================================================================
# BASKET STRATEGY ENGINE
# =============================================================================
# Basket = 3 straddles (N-100, N, N+100), N = NIFTY close at/after 09:40
# rounded to the nearest 100. Combined premium = sum of all 6 option legs'
# closes. Combined VWAP is computed as ONE synthetic-instrument VWAP on that
# summed price series, weighted by the summed volume of all 6 legs (resets
# each session, same convention as the per-strike VWAP elsewhere).
#
# SHORT entry: combined premium < combined VWAP — either the first check at
# 09:45 if already true, or any later fresh cross from above to below VWAP,
# up to 14:30, capped at BASKET_MAX_TRADES_PER_DAY, one position open at a time.
#
# Exit (short, so favourable move is DOWN):
#   - premium crosses back ABOVE VWAP  -> exit ("VWAP_CROSS")
#   - premium crosses back ABOVE the trailing ATR(BASKET_ATR_PERIOD) stop -> exit ("TRAIL_ATR_STOP")
#     stop = (lowest premium since entry) + BASKET_ATR_MULTIPLIER * ATR(BASKET_ATR_PERIOD)
#     ATR here uses a simplified True Range (abs close-to-close change only).
#   - 15:15 forced square-off -> exit ("EOD_SQUAREOFF")

def determine_basket_center_strike(spot_df, sample_time_str=BASKET_STRIKE_SAMPLE_TIME, step=STRIKE_STEP):
    """NIFTY close at/after sample_time_str, rounded to nearest `step`. Returns None if unavailable."""
    if spot_df.empty:
        return None
    sample_time = dtime.fromisoformat(sample_time_str)
    candidates = spot_df[spot_df["time"].dt.time >= sample_time]
    row = candidates.iloc[0] if not candidates.empty else spot_df.iloc[-1]
    price = row.get("spot", row.get("close"))
    if price is None or price != price:
        return None
    return int(round(price / step) * step)

def fetch_basket_leg_candles(fyers, strike, target_date):
    """Fetch CE+PE candles (time, close, high, low, volume) for one basket strike."""
    ce_symbol = f"NSE:NIFTY{EXPIRY}{strike}CE"
    pe_symbol = f"NSE:NIFTY{EXPIRY}{strike}PE"
    ce_df = fetch_candles(fyers, ce_symbol, target_date)
    pe_df = fetch_candles(fyers, pe_symbol, target_date)
    if ce_df.empty or pe_df.empty:
        return None
    cols = ["time", "close", "high", "low", "volume"]
    merged = pd.merge(ce_df[cols], pe_df[cols], on="time", suffixes=("_ce", "_pe"))
    return merged

def build_basket_dataframe(fyers, target_date, spot_df, offsets=BASKET_OFFSETS):
    """Determine N, fetch all 3 basket strikes' legs, and build the combined
    (synthetic single-instrument) basket price/VWAP/ATR series.
    Returns (basket_df, N) or (None, None) if data is unavailable.
    """
    N = determine_basket_center_strike(spot_df)
    if N is None:
        logger.warning("Basket: could not determine center strike N (no spot data).")
        return None, None

    leg_frames = []
    for offset in offsets:
        strike = N + offset
        leg_df = fetch_basket_leg_candles(fyers, strike, target_date)
        if leg_df is None:
            logger.warning(f"Basket: no data for strike {strike} — skipping basket build.")
            return None, N
        leg_frames.append(leg_df.set_index("time"))

    # Align all 3 strikes on common timestamps, then sum across the 6 legs.
    common_index = leg_frames[0].index
    for lf in leg_frames[1:]:
        common_index = common_index.intersection(lf.index)
    if len(common_index) == 0:
        logger.warning("Basket: no overlapping timestamps across the 3 strikes.")
        return None, N
    common_index = common_index.sort_values()

    basket_close = sum(lf.loc[common_index, "close_ce"] + lf.loc[common_index, "close_pe"] for lf in leg_frames)
    basket_high  = sum(lf.loc[common_index, "high_ce"]  + lf.loc[common_index, "high_pe"]  for lf in leg_frames)
    basket_low   = sum(lf.loc[common_index, "low_ce"]   + lf.loc[common_index, "low_pe"]   for lf in leg_frames)
    basket_vol   = sum(lf.loc[common_index, "volume_ce"] + lf.loc[common_index, "volume_pe"] for lf in leg_frames)

    basket_df = pd.DataFrame({
        "time": common_index, "close": basket_close.values, "high": basket_high.values,
        "low": basket_low.values, "volume": basket_vol.values,
    }).reset_index(drop=True)

    # Session VWAP on the combined basket (single synthetic instrument).
    pv = basket_df["close"] * basket_df["volume"]
    basket_df["vwap"] = pv.cumsum() / basket_df["volume"].cumsum().replace(0, np.nan)

    # ATR(BASKET_ATR_PERIOD) on the combined basket, using a simplified True
    # Range (abs close-to-close change only, not full High-Low range). Warm-up
    # bars before `period` candles of history exist just average whatever's
    # available so far (rolling min_periods=1).
    tr = basket_df["close"].diff().abs()
    basket_df["atr"] = tr.rolling(window=BASKET_ATR_PERIOD, min_periods=1).mean()

    return basket_df, N

def run_basket_strategy(basket_df):
    """Simulate the short-the-dip-below-VWAP strategy over basket_df.
    Returns (trades, open_position, basket_df_with_pnl).
    trades: list of completed-trade dicts.
    open_position: dict if a trade is still open at the end of the data, else None.
    """
    if basket_df is None or basket_df.empty:
        return [], None, basket_df

    entry_start  = dtime.fromisoformat(BASKET_ENTRY_START_TIME)
    entry_cutoff = dtime.fromisoformat(BASKET_ENTRY_CUTOFF_TIME)
    eod_time     = dtime.fromisoformat(BASKET_EOD_EXIT_TIME)

    trades = []
    position = None
    total_entries = 0
    prev_below = None
    entry_window_started = False

    realized_cum = 0.0
    pnl_points_series = []
    realized_points_series = []

    for _, row in basket_df.iterrows():
        t = row["time"].time()
        price = row["close"]
        vwap = row["vwap"]
        atr = row["atr"] if row["atr"] == row["atr"] else 0.0
        below = (price < vwap) if (price == price and vwap == vwap) else None

        # ---- manage an open position first ----
        if position is not None:
            position["trailing_low"] = min(position["trailing_low"], price)
            candidate_stop = position["trailing_low"] + BASKET_ATR_MULTIPLIER * atr
            # Ratchet only: the stop must never loosen (move up) once tightened,
            # even if ATR itself later expands on a volatile candle.
            position["stop_level"] = candidate_stop if position["stop_level"] is None \
                else min(position["stop_level"], candidate_stop)
            stop_level = position["stop_level"]

            exit_reason = None
            if t >= eod_time:
                exit_reason = "EOD_SQUAREOFF"
            elif vwap == vwap and price > vwap:
                exit_reason = "VWAP_CROSS"
            elif price > stop_level:
                exit_reason = "TRAIL_ATR_STOP"

            if exit_reason:
                pnl_pts = position["entry_price"] - price   # SHORT: profit when price falls
                trade = dict(position)
                trade.update({
                    "exit_time": row["time"], "exit_price": price, "exit_reason": exit_reason,
                    "pnl_points": pnl_pts, "pnl_rupees": pnl_pts * LOT_SIZE,
                })
                trades.append(trade)
                realized_cum += pnl_pts
                position = None

        # ---- check for a new entry (only if flat) ----
        if (position is None and below and t >= entry_start and t <= entry_cutoff
                and total_entries < BASKET_MAX_TRADES_PER_DAY):
            fresh_cross = (not entry_window_started) or (prev_below is False)
            if fresh_cross:
                total_entries += 1
                position = {
                    "trade_no": total_entries, "entry_time": row["time"], "entry_price": price,
                    "entry_vwap": vwap, "trailing_low": price,
                    "stop_level": price + BASKET_ATR_MULTIPLIER * atr,
                }

        if t >= entry_start:
            entry_window_started = True
        prev_below = below

        unrealized = (position["entry_price"] - price) if position is not None else 0.0
        pnl_points_series.append(realized_cum + unrealized)
        realized_points_series.append(realized_cum)

    basket_df = basket_df.copy()
    basket_df["pnl_points"] = pnl_points_series
    basket_df["realized_pnl_points"] = realized_points_series
    basket_df["pnl_rupees"] = basket_df["pnl_points"] * LOT_SIZE

    return trades, position, basket_df

# --- Basket state persistence (avoids duplicate Telegram alerts across reruns) ---

def load_basket_state():
    path = BASKET_STATE_DOCS_FILE if os.path.exists(BASKET_STATE_DOCS_FILE) else BASKET_STATE_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read basket state ({path}): {e}")
    return {"notified_entries": [], "notified_exits": []}

def save_basket_state(state):
    for path in (BASKET_STATE_FILE, BASKET_STATE_DOCS_FILE):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Could not write basket state ({path}): {e}")

def _fmt_time(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)

def notify_basket_events(trades, open_position, N):
    """Send Telegram alerts for any entry/exit not already notified in a prior run."""
    state = load_basket_state()
    notified_entries = set(state.get("notified_entries", []))
    notified_exits   = set(state.get("notified_exits", []))

    strikes_label = f"{N-100}/{N}/{N+100}"
    all_positions = trades + ([open_position] if open_position else [])

    for pos in sorted(all_positions, key=lambda p: p["entry_time"]):
        key = _fmt_time(pos["entry_time"])
        if key in notified_entries:
            continue
        send_telegram_message(
            f"🔴 <b>BASKET SHORT ENTRY #{pos['trade_no']}</b>\n"
            f"Strikes (CE+PE): <code>{strikes_label}</code>\n"
            f"Time: {key}\n"
            f"Premium: ₹{pos['entry_price']:.2f}  &lt;  VWAP: ₹{pos['entry_vwap']:.2f}"
        )
        notified_entries.add(key)

    reason_labels = {
        "VWAP_CROSS": "VWAP reclaim (cover)",
        "TRAIL_ATR_STOP": f"Trailing ATR({BASKET_ATR_PERIOD}) stop x{BASKET_ATR_MULTIPLIER}",
        "EOD_SQUAREOFF": "EOD square-off (15:15)",
    }
    for tr in sorted(trades, key=lambda t: t["exit_time"]):
        key = _fmt_time(tr["exit_time"])
        if key in notified_exits:
            continue
        emoji = "✅" if tr["pnl_points"] >= 0 else "❌"
        reason = reason_labels.get(tr["exit_reason"], tr["exit_reason"])
        send_telegram_message(
            f"{emoji} <b>BASKET EXIT #{tr['trade_no']}</b> — {reason}\n"
            f"Strikes (CE+PE): <code>{strikes_label}</code>\n"
            f"Entry: ₹{tr['entry_price']:.2f} @ {_fmt_time(tr['entry_time'])}\n"
            f"Exit:  ₹{tr['exit_price']:.2f} @ {key}\n"
            f"PnL: {tr['pnl_points']:+.2f} pts  (₹{tr['pnl_rupees']:+,.0f})"
        )
        notified_exits.add(key)

    save_basket_state({
        "notified_entries": sorted(notified_entries),
        "notified_exits": sorted(notified_exits),
    })

# --- DASHBOARD BUILDER ---

def _metric_figure(straddle_data, strikes, atm, column, title, yaxis_title):
    fig = go.Figure()
    for idx, strike in enumerate(strikes):
        df    = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        label = f"{strike}" + (" (ATM)" if strike == atm else "")
        fig.add_trace(go.Scatter(x=df["time"], y=df[column], name=label, visible=True,
                                  line=dict(color=color, width=2)))
    fig.update_layout(
        title=title, template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, yaxis_title=yaxis_title, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def build_dashboard_html(straddle_data, atm, rankings, gex_history=None, combined_gex_history=None,
                          basket_df=None, basket_trades=None, basket_open_position=None, basket_N=None):
    strikes               = sorted(straddle_data.keys())
    gex_history           = gex_history or []
    combined_gex_history  = combined_gex_history or []
    basket_trades         = basket_trades or []

    # ── Overview straddle subplots ────────────────────────────────────────────
    fig_main = make_subplots(
        rows=len(strikes), cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=[f"Strike {s}{' ◄ ATM' if s == atm else ''}" for s in strikes]
    )
    for idx, strike in enumerate(strikes):
        df    = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        row   = idx + 1
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"{strike}",
                                       line=dict(color=color, width=2)), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["vwap"], name="VWAP",
                                       line=dict(color=MUTED, width=1, dash="dot")), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["ema9"], name="EMA9",
                                       line=dict(color=EMA_COLOR, width=1.5, dash="dash")), row=row, col=1)
    fig_main.update_layout(height=200 * len(strikes), template="plotly_dark",
                            paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False)

    # ── Smoothness ranking bar ────────────────────────────────────────────────
    fig_rank = go.Figure(go.Bar(
        y=[str(r["strike"]) for r in rankings], x=[r["smoothness"] for r in rankings],
        orientation="h",
        marker_color=[ACCENT if r["rank"] == 1 else "#1e3a2f" for r in rankings],
        text=[f"{r['smoothness']}%" for r in rankings],
        textposition="outside", cliponaxis=False
    ))
    fig_rank.update_layout(
        title="Smoothness Ranking (Full Day)", template="plotly_dark",
        paper_bgcolor=CARD, plot_bgcolor=CARD, height=400,
        margin=dict(l=10, r=40, t=40, b=10),
        yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 110])
    )

    # ── Greek figures ─────────────────────────────────────────────────────────
    fig_iv      = _metric_figure(straddle_data, strikes, atm, "iv_pct",      "Implied Volatility (Straddle Avg)",            "IV (%)")
    fig_theta   = _metric_figure(straddle_data, strikes, atm, "theta_total", "Theta (Straddle Total, ₹/day)",                "Theta")
    fig_delta   = _metric_figure(straddle_data, strikes, atm, "delta_total", "Delta (Straddle Net, CE + PE)",                "Delta")
    fig_gamma   = _metric_figure(straddle_data, strikes, atm, "gamma_total", "Gamma (Straddle Total)",                       "Gamma")
    fig_theta15 = _metric_figure(straddle_data, strikes, atm, "theta_15min", f"Theta Decay (Trailing {THETA_WINDOW_MINUTES} min, ₹)", "Theta (₹ / 15 min)")

    # ── Single-expiry GEX chart ───────────────────────────────────────────────
    fig_gex = make_subplots(specs=[[{"secondary_y": True}]])
    if gex_history:
        gex_times = [h["time"]    for h in gex_history]
        gex_spots = [h["spot"]    for h in gex_history]
        gex_flips = [h["flip"]    for h in gex_history]
        gex_nets  = [h["net_gex"] / GEX_SCALE if h["net_gex"] is not None else None for h in gex_history]
        fig_gex.add_trace(go.Scatter(x=gex_times, y=gex_spots, name="NIFTY Spot",
                                      line=dict(color=BLUE, width=2)), secondary_y=False)
        fig_gex.add_trace(go.Scatter(x=gex_times, y=gex_flips, name="Gamma Flip Level",
                                      line=dict(color=ACCENT, width=2, dash="dash")), secondary_y=False)
        fig_gex.add_trace(go.Scatter(x=gex_times, y=gex_nets, name="Net GEX (₹ Cr)",
                                      line=dict(color=RED, width=2),
                                      fill="tozeroy", fillcolor="rgba(255,69,96,0.08)"), secondary_y=True)
        fig_gex.add_trace(go.Scatter(x=gex_times, y=[0]*len(gex_times),
                                      line=dict(color=MUTED, width=1, dash="dot"), showlegend=False), secondary_y=True)
    fig_gex.update_layout(
        title="NIFTY Spot vs Gamma Flip Level  |  Net GEX (₹ Cr)",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    fig_gex.update_yaxes(title_text="NIFTY Price", secondary_y=False)
    fig_gex.update_yaxes(title_text="Net GEX (₹ Cr)", secondary_y=True)

    # ── Combined multi-expiry GEX chart ───────────────────────────────────────
    fig_gex_combined = make_subplots(specs=[[{"secondary_y": True}]])
    if combined_gex_history:
        c_times = [h["time"]    for h in combined_gex_history]
        c_spots = [h["spot"]    for h in combined_gex_history]
        c_flips = [h["flip"]    for h in combined_gex_history]
        c_nets  = [h["net_gex"] / GEX_SCALE if h["net_gex"] is not None else None for h in combined_gex_history]
        fig_gex_combined.add_trace(go.Scatter(x=c_times, y=c_spots, name="NIFTY Spot",
                                               line=dict(color=BLUE, width=2)), secondary_y=False)
        fig_gex_combined.add_trace(go.Scatter(x=c_times, y=c_flips, name="Combined Gamma Flip",
                                               line=dict(color=ACCENT, width=2, dash="dash")), secondary_y=False)
        fig_gex_combined.add_trace(go.Scatter(x=c_times, y=c_nets, name="Combined Net GEX (₹ Cr)",
                                               line=dict(color=RED, width=2.5),
                                               fill="tozeroy", fillcolor="rgba(255,69,96,0.08)"), secondary_y=True)
        fig_gex_combined.add_trace(go.Scatter(x=c_times, y=[0]*len(c_times),
                                               line=dict(color=MUTED, width=1, dash="dot"), showlegend=False), secondary_y=True)
        # Per-expiry breakdown lines
        expiry_codes_seen = sorted({code for h in combined_gex_history for code in (h.get("per_expiry") or {}).keys()})
        for idx, code in enumerate(expiry_codes_seen):
            per_series = [(h.get("per_expiry") or {}).get(code) for h in combined_gex_history]
            per_series = [v / GEX_SCALE if v is not None else None for v in per_series]
            fig_gex_combined.add_trace(go.Scatter(
                x=c_times, y=per_series, name=f"Expiry {code} (₹ Cr)",
                line=dict(color=STRIKE_COLORS[idx % len(STRIKE_COLORS)], width=1, dash="dot")
            ), secondary_y=True)
    fig_gex_combined.update_layout(
        title=f"Combined GEX Across Expiries ({', '.join(GEX_MULTI_EXPIRY_CODES)})",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=560, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    fig_gex_combined.update_yaxes(title_text="NIFTY Price", secondary_y=False)
    fig_gex_combined.update_yaxes(title_text="Net GEX (₹ Cr)", secondary_y=True)

    # ── OI Difference chart (current expiry: total PE OI − total CE OI) ──────
    fig_oi_diff = make_subplots(specs=[[{"secondary_y": True}]])
    if gex_history:
        oi_times = [h["time"] for h in gex_history]
        oi_spots = [h["spot"] for h in gex_history]
        oi_ce    = [h.get("oi_ce")   for h in gex_history]
        oi_pe    = [h.get("oi_pe")   for h in gex_history]
        oi_diff  = [h.get("oi_diff") for h in gex_history]
        fig_oi_diff.add_trace(go.Scatter(x=oi_times, y=oi_spots, name="NIFTY Spot",
                                          line=dict(color=BLUE, width=2)), secondary_y=False)
        fig_oi_diff.add_trace(go.Scatter(x=oi_times, y=oi_ce, name="Total CE OI",
                                          line=dict(color=RED, width=1.5, dash="dot")), secondary_y=True)
        fig_oi_diff.add_trace(go.Scatter(x=oi_times, y=oi_pe, name="Total PE OI",
                                          line=dict(color=ACCENT, width=1.5, dash="dot")), secondary_y=True)
        fig_oi_diff.add_trace(go.Scatter(x=oi_times, y=oi_diff, name="OI Diff (PE − CE)",
                                          line=dict(color="#a78bfa", width=2.5),
                                          fill="tozeroy", fillcolor="rgba(167,139,250,0.10)"), secondary_y=True)
        fig_oi_diff.add_trace(go.Scatter(x=oi_times, y=[0]*len(oi_times),
                                          line=dict(color=MUTED, width=1, dash="dot"), showlegend=False), secondary_y=True)
    fig_oi_diff.update_layout(
        title="NIFTY Spot vs Open Interest (PE − CE), Current Expiry",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    fig_oi_diff.update_yaxes(title_text="NIFTY Price", secondary_y=False)
    fig_oi_diff.update_yaxes(title_text="Open Interest (contracts)", secondary_y=True)

    # ── Basket strategy: price / VWAP / ATR chart with entry-exit markers ─────
    fig_basket = go.Figure()
    if basket_df is not None and not basket_df.empty:
        fig_basket.add_trace(go.Scatter(x=basket_df["time"], y=basket_df["close"], name="Basket Premium",
                                         line=dict(color=BLUE, width=2)))
        fig_basket.add_trace(go.Scatter(x=basket_df["time"], y=basket_df["vwap"], name="Basket VWAP",
                                         line=dict(color=ACCENT, width=1.5, dash="dot")))
        fig_basket.add_trace(go.Scatter(x=basket_df["time"], y=basket_df["atr"], name=f"ATR({BASKET_ATR_PERIOD})",
                                         line=dict(color=MUTED, width=1, dash="dash"), yaxis="y2"))

        all_trades_for_markers = list(basket_trades) + ([basket_open_position] if basket_open_position else [])
        if all_trades_for_markers:
            entry_x = [p["entry_time"] for p in all_trades_for_markers]
            entry_y = [p["entry_price"] for p in all_trades_for_markers]
            fig_basket.add_trace(go.Scatter(
                x=entry_x, y=entry_y, mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=13, color=RED, line=dict(width=1, color=TEXT)),
                hovertemplate="Entry #%{text}<br>%{x}<br>₹%{y:.2f}<extra></extra>",
                text=[p["trade_no"] for p in all_trades_for_markers],
            ))
        if basket_trades:
            exit_x = [t["exit_time"] for t in basket_trades]
            exit_y = [t["exit_price"] for t in basket_trades]
            exit_colors = [ACCENT if t["pnl_points"] >= 0 else RED for t in basket_trades]
            fig_basket.add_trace(go.Scatter(
                x=exit_x, y=exit_y, mode="markers", name="Exit / Cover",
                marker=dict(symbol="triangle-up", size=13, color=exit_colors, line=dict(width=1, color=TEXT)),
                hovertemplate="Exit #%{text}<br>%{x}<br>₹%{y:.2f}<extra></extra>",
                text=[t["trade_no"] for t in basket_trades],
            ))
    fig_basket.update_layout(
        title=f"Basket Premium vs VWAP  |  Strikes {basket_N-100 if basket_N else '-'}/{basket_N or '-'}/{basket_N+100 if basket_N else '-'}",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15),
        yaxis=dict(title="Basket Premium (₹)"),
        yaxis2=dict(title=f"ATR({BASKET_ATR_PERIOD})", overlaying="y", side="right", showgrid=False),
    )

    # ── Basket strategy: intraday PnL curve ───────────────────────────────────
    fig_basket_pnl = go.Figure()
    if basket_df is not None and not basket_df.empty and "pnl_rupees" in basket_df.columns:
        pnl_colors = ["rgba(0,229,176,0.15)" if v >= 0 else "rgba(255,69,96,0.15)" for v in basket_df["pnl_rupees"]]
        fig_basket_pnl.add_trace(go.Scatter(
            x=basket_df["time"], y=basket_df["pnl_rupees"], name="Running PnL (₹)",
            line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(0,229,176,0.10)"
        ))
        fig_basket_pnl.add_trace(go.Scatter(
            x=basket_df["time"], y=[0]*len(basket_df), line=dict(color=MUTED, width=1, dash="dot"), showlegend=False
        ))
    fig_basket_pnl.update_layout(
        title="Basket Strategy — Running PnL (₹, mark-to-market)",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15), yaxis=dict(title="PnL (₹)"),
    )

    # ── Basket strategy: trade log table ───────────────────────────────────────
    if basket_trades:
        reason_labels = {
            "VWAP_CROSS": "VWAP reclaim", "TRAIL_ATR_STOP": f"ATR({BASKET_ATR_PERIOD}) stop", "EOD_SQUAREOFF": "EOD",
        }
        basket_trade_rows_html = "".join([
            f"""<tr style="border-bottom:1px solid {BORDER};">
                <td style="padding:8px;">{t['trade_no']}</td>
                <td style="padding:8px;">{t['entry_time'].strftime('%H:%M:%S')}</td>
                <td style="padding:8px;">₹{t['entry_price']:.2f}</td>
                <td style="padding:8px;">{t['exit_time'].strftime('%H:%M:%S')}</td>
                <td style="padding:8px;">₹{t['exit_price']:.2f}</td>
                <td style="padding:8px;">{reason_labels.get(t['exit_reason'], t['exit_reason'])}</td>
                <td style="padding:8px;color:{ACCENT if t['pnl_points']>=0 else RED};font-weight:bold;">{t['pnl_points']:+.2f}</td>
                <td style="padding:8px;color:{ACCENT if t['pnl_points']>=0 else RED};font-weight:bold;">₹{t['pnl_rupees']:+,.0f}</td>
            </tr>""" for t in basket_trades
        ])
        total_pnl_rupees = sum(t["pnl_rupees"] for t in basket_trades)
    else:
        basket_trade_rows_html = f'<tr><td colspan="8" style="padding:12px;color:{MUTED};">No completed trades yet today.</td></tr>'
        total_pnl_rupees = 0.0
    if basket_open_position:
        basket_trade_rows_html += f"""<tr style="border-bottom:1px solid {BORDER};background:{BORDER}22;">
            <td style="padding:8px;">{basket_open_position['trade_no']}</td>
            <td style="padding:8px;">{basket_open_position['entry_time'].strftime('%H:%M:%S')}</td>
            <td style="padding:8px;">₹{basket_open_position['entry_price']:.2f}</td>
            <td style="padding:8px;" colspan="5">🟡 OPEN</td>
        </tr>"""

    # ── HTML helpers ──────────────────────────────────────────────────────────
    table_rows_html = "".join([
        f"""<tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:10px;">{r['rank']}</td>
            <td style="padding:10px;color:{ACCENT};"><b>{r['strike']}</b></td>
            <td style="padding:10px;">{r['smoothness']}%</td>
            <td style="padding:10px;color:{BLUE if r['angle']>0 else RED};">{r['angle']}°</td>
            <td style="padding:10px;">{r['trend']}</td>
        </tr>""" for r in rankings
    ])

    toggle_items_html = "".join([
        f"""<label class="toggle-item">
            <input type="checkbox" checked onchange="toggleStrike({idx}, this.checked)">
            <span class="strike-dot" style="background:{STRIKE_COLORS[idx % len(STRIKE_COLORS)]};"></span>
            {strike}{' <small style="color:' + ACCENT + '">ATM</small>' if strike == atm else ''}
        </label>""" for idx, strike in enumerate(strikes)
    ])

    # ── OTM Analysis figures ──────────────────────────────────────────────────
    # CE OTM = strikes ABOVE ATM  → show CE leg: price, IV, Theta, 15-min Theta
    # PE OTM = strikes BELOW ATM  → show PE leg: price, IV, Theta, 15-min Theta
    ce_otm_strikes = [s for s in strikes if s > atm]
    pe_otm_strikes = [s for s in strikes if s < atm]

    def _otm_fig(strike_list, col, title, yaxis_title, data_col_key):
        """Build a Plotly figure for one OTM group + one metric column."""
        fig = go.Figure()
        for idx, strike in enumerate(strike_list):
            df    = straddle_data[strike]
            if data_col_key not in df.columns:
                continue
            color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[data_col_key],
                name=str(strike),
                line=dict(color=color, width=2),
                hovertemplate=f"Strike {strike}<br>%{{x}}<br>{yaxis_title}: %{{y:.4f}}<extra></extra>",
            ))
        fig.update_layout(
            title=title, template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
            height=380, yaxis_title=yaxis_title, margin=dict(l=10, r=10, t=45, b=10),
            legend=dict(orientation="h", y=-0.20),
        )
        return fig

    # CE OTM: 4 charts
    fig_ce_price    = _otm_fig(ce_otm_strikes, "close_x", "CE OTM — Price (₹)",              "CE Price (₹)",    "close_x")
    fig_ce_iv       = _otm_fig(ce_otm_strikes, "iv_ce",   "CE OTM — Implied Volatility (%)",  "IV (%)",          "iv_ce")
    fig_ce_theta    = _otm_fig(ce_otm_strikes, "theta_ce", "CE OTM — Theta (₹/day)",          "Theta (₹/day)",   "theta_ce")
    fig_ce_theta15  = _otm_fig(ce_otm_strikes, "theta_ce_15min", f"CE OTM — Trailing {THETA_WINDOW_MINUTES}-min Theta (₹)", "Theta (₹/15min)", "theta_ce_15min")

    # PE OTM: 4 charts
    fig_pe_price    = _otm_fig(pe_otm_strikes, "close_y", "PE OTM — Price (₹)",              "PE Price (₹)",    "close_y")
    fig_pe_iv       = _otm_fig(pe_otm_strikes, "iv_pe",   "PE OTM — Implied Volatility (%)",  "IV (%)",          "iv_pe")
    fig_pe_theta    = _otm_fig(pe_otm_strikes, "theta_pe", "PE OTM — Theta (₹/day)",          "Theta (₹/day)",   "theta_pe")
    fig_pe_theta15  = _otm_fig(pe_otm_strikes, "theta_pe_15min", f"PE OTM — Trailing {THETA_WINDOW_MINUTES}-min Theta (₹)", "Theta (₹/15min)", "theta_pe_15min")

    # Convert all IV series from raw (0–1) to % for display (iv_ce is a fraction from implied_vol)
    for _fig in (fig_ce_iv, fig_pe_iv):
        for _trace in _fig.data:
            _trace.y = [v * 100.0 if (v is not None and v == v) else v for v in (_trace.y if _trace.y is not None else [])]

    # OTM label strings and Plotly HTML for the f-string
    ce_otm_label   = ", ".join(str(s) for s in ce_otm_strikes) if ce_otm_strikes else "none"
    pe_otm_label   = ", ".join(str(s) for s in pe_otm_strikes) if pe_otm_strikes else "none"

    # Plotly.js is embedded once (include_plotlyjs=True) in the very first chart
    # rendered in the page (the ranking bar chart in the Overview tab), and every
    # other figure on the page reuses that same library copy via include_plotlyjs=False.
    # Using True here (rather than 'cdn') makes the whole dashboard self-contained,
    # so charts render even with no internet access / a blocked CDN domain.
    _pjs = False   # plotlyjs already embedded by the fig_rank chart above
    _pcfg = {"responsive": True}   # must be defined before any .to_html() calls below
    ce_price_html   = fig_ce_price.to_html(  full_html=False, include_plotlyjs=_pjs, div_id="cePriceChart",   config=_pcfg)
    ce_iv_html      = fig_ce_iv.to_html(     full_html=False, include_plotlyjs=_pjs, div_id="ceIvChart",      config=_pcfg)
    ce_theta_html   = fig_ce_theta.to_html(  full_html=False, include_plotlyjs=_pjs, div_id="ceThetaChart",   config=_pcfg)
    ce_theta15_html = fig_ce_theta15.to_html(full_html=False, include_plotlyjs=_pjs, div_id="ceTheta15Chart", config=_pcfg)
    pe_price_html   = fig_pe_price.to_html(  full_html=False, include_plotlyjs=_pjs, div_id="pePriceChart",   config=_pcfg)
    pe_iv_html      = fig_pe_iv.to_html(     full_html=False, include_plotlyjs=_pjs, div_id="peIvChart",      config=_pcfg)
    pe_theta_html   = fig_pe_theta.to_html(  full_html=False, include_plotlyjs=_pjs, div_id="peThetaChart",   config=_pcfg)
    pe_theta15_html = fig_pe_theta15.to_html(full_html=False, include_plotlyjs=_pjs, div_id="peTheta15Chart", config=_pcfg)

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
        .tabs {{ display: flex; gap: 4px; padding: 0 20px; background: {BG}; border-bottom: 1px solid {BORDER}; flex-wrap: wrap; }}
        .tab-btn {{ background: transparent; border: none; color: {MUTED}; padding: 12px 18px; font-size: 13px; letter-spacing: 0.5px; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab-btn:hover {{ color: {TEXT}; }}
        .tab-btn.active {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .toggle-bar {{ display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 14px 20px; background: {CARD}; border-bottom: 1px solid {BORDER}; }}
        .toggle-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: {TEXT}; cursor: pointer; user-select: none; }}
        .toggle-item input {{ accent-color: {ACCENT}; cursor: pointer; }}
        .metric-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; margin: 20px; }}
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

    <div class="tabs">
        <button class="tab-btn active" id="btn-overview"    onclick="showTab('overview')">OVERVIEW</button>
        <button class="tab-btn"        id="btn-iv"          onclick="showTab('iv')">IMPLIED VOL</button>
        <button class="tab-btn"        id="btn-theta"       onclick="showTab('theta')">THETA</button>
        <button class="tab-btn"        id="btn-delta"       onclick="showTab('delta')">DELTA</button>
        <button class="tab-btn"        id="btn-gamma"       onclick="showTab('gamma')">GAMMA</button>
        <button class="tab-btn"        id="btn-theta15"     onclick="showTab('theta15')">15 MIN THETA</button>
        <button class="tab-btn"        id="btn-gex"         onclick="showTab('gex')">GEX &amp; FLIP</button>
        <button class="tab-btn"        id="btn-gexcombined" onclick="showTab('gexcombined')">COMBINED GEX</button>
        <button class="tab-btn"        id="btn-oidiff"      onclick="showTab('oidiff')">OI DIFFERENCE</button>
        <button class="tab-btn"        id="btn-otm"         onclick="showTab('otm')">OTM ANALYSIS</button>
        <button class="tab-btn"        id="btn-basket"      onclick="showTab('basket')">BASKET STRATEGY</button>
    </div>

    <div class="toggle-bar" id="strikeToggleBar">
        {toggle_items_html}
    </div>

    <div class="tab-content active" id="tab-overview">
        <div class="content">
            <div class="sidebar">
                <div class="card">
                    <h2>SMOOTHNESS RANKING</h2>
                    {fig_rank.to_html(full_html=False, include_plotlyjs=True, div_id='rankChart')}
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
                    {fig_main.to_html(full_html=False, include_plotlyjs=False, div_id='mainChart')}
                </div>
            </div>
        </div>
    </div>

    <div class="tab-content" id="tab-iv">
        <div class="metric-card">
            {fig_iv.to_html(full_html=False, include_plotlyjs=False, div_id='ivChart', config={'responsive': True})}
        </div>
    </div>
    <div class="tab-content" id="tab-theta">
        <div class="metric-card">
            {fig_theta.to_html(full_html=False, include_plotlyjs=False, div_id='thetaChart', config={'responsive': True})}
        </div>
    </div>
    <div class="tab-content" id="tab-delta">
        <div class="metric-card">
            {fig_delta.to_html(full_html=False, include_plotlyjs=False, div_id='deltaChart', config={'responsive': True})}
        </div>
    </div>
    <div class="tab-content" id="tab-gamma">
        <div class="metric-card">
            {fig_gamma.to_html(full_html=False, include_plotlyjs=False, div_id='gammaChart', config={'responsive': True})}
        </div>
    </div>
    <div class="tab-content" id="tab-theta15">
        <div class="metric-card">
            {fig_theta15.to_html(full_html=False, include_plotlyjs=False, div_id='theta15Chart', config={'responsive': True})}
        </div>
    </div>

    <div class="tab-content" id="tab-gex">
        <div class="metric-card">
            <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
                Net GEX &gt; 0 → dealers net long gamma (moves dampened / pinned).
                Net GEX &lt; 0 → dealers net short gamma (moves accelerate / trend).
                Time series builds from the first run today — OI has no historical intraday series via the broker API.
            </div>
            {fig_gex.to_html(full_html=False, include_plotlyjs=False, div_id='gexChart', config={'responsive': True})}
        </div>
    </div>

    <div class="tab-content" id="tab-gexcombined">
        <div class="metric-card">
            <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
                Combines GEX across expiries {', '.join(GEX_MULTI_EXPIRY_CODES)} into one aggregate flip level
                and net GEX. Thin dotted lines show each expiry's individual contribution.
                Each expiry uses its own time-to-expiry T when recomputing gamma on the flip grid.
            </div>
            {fig_gex_combined.to_html(full_html=False, include_plotlyjs=False, div_id='gexCombinedChart', config={'responsive': True})}
        </div>
    </div>

    <div class="tab-content" id="tab-oidiff">
        <div class="metric-card">
            <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
                Total Open Interest across all strikes for the current (nearest) expiry.
                OI Diff = Total PE OI − Total CE OI. Positive → put writers dominate (support building below spot).
                Negative → call writers dominate (resistance building above spot). Builds intraday from the first run today.
            </div>
            {fig_oi_diff.to_html(full_html=False, include_plotlyjs=False, div_id='oiDiffChart', config={'responsive': True})}
        </div>
    </div>

    <div class="tab-content" id="tab-otm">
        <div style="padding:20px;">

            <!-- ── CE OTM Section ── -->
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px 20px;margin-bottom:8px;">
                <h2 style="color:{ACCENT};font-size:14px;letter-spacing:1px;margin:0 0 4px 0;">
                    📈 CE OTM STRIKES — {ce_otm_label}
                </h2>
                <div style="font-size:11px;color:{MUTED};">
                    Strikes above ATM ({atm}). CE leg price, IV, Theta and 15-min trailing Theta.
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {ce_price_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {ce_iv_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {ce_theta_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {ce_theta15_html}
                </div>
            </div>

            <!-- ── PE OTM Section ── -->
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px 20px;margin-bottom:8px;">
                <h2 style="color:{RED};font-size:14px;letter-spacing:1px;margin:0 0 4px 0;">
                    📉 PE OTM STRIKES — {pe_otm_label}
                </h2>
                <div style="font-size:11px;color:{MUTED};">
                    Strikes below ATM ({atm}). PE leg price, IV, Theta and 15-min trailing Theta.
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {pe_price_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {pe_iv_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {pe_theta_html}
                </div>
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;">
                    {pe_theta15_html}
                </div>
            </div>

        </div>
    </div>

    <div class="tab-content" id="tab-basket">
        <div class="metric-card">
            <div style="color:{MUTED};font-size:12px;margin-bottom:12px;">
                Short a 3-strike straddle basket (N−100, N, N+100 — N = NIFTY close at/after 09:40, rounded to nearest 100).
                Entry: combined premium below combined VWAP (09:45 check, or any fresh cross-under, until 14:30) — max {BASKET_MAX_TRADES_PER_DAY} trades/day.
                Exit: VWAP reclaim, trailing ATR({BASKET_ATR_PERIOD}) stop (×{BASKET_ATR_MULTIPLIER}), or forced square-off at {BASKET_EOD_EXIT_TIME}.
            </div>
            {fig_basket.to_html(full_html=False, include_plotlyjs=False, div_id='basketChart', config={'responsive': True})}
        </div>
        <div class="metric-card">
            {fig_basket_pnl.to_html(full_html=False, include_plotlyjs=False, div_id='basketPnlChart', config={'responsive': True})}
        </div>
        <div class="card" style="margin:20px;">
            <h2>TRADE LOG {f'&nbsp;&nbsp;<span style="color:{ACCENT if total_pnl_rupees>=0 else RED};">Total: ₹{total_pnl_rupees:+,.0f}</span>' if basket_trades else ''}</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th><th>Entry Time</th><th>Entry ₹</th><th>Exit Time</th><th>Exit ₹</th>
                        <th>Exit Reason</th><th>PnL (pts)</th><th>PnL (₹)</th>
                    </tr>
                </thead>
                <tbody>{basket_trade_rows_html}</tbody>
            </table>
        </div>
    </div>

    <script>
    const strikes   = {json.dumps([str(s) for s in strikes])};
    const colors    = {json.dumps(STRIKE_COLORS)};
    const ATM       = "{atm}";

    const metricTabs = ['iv', 'theta', 'delta', 'gamma', 'theta15'];
    const tabChartIds = {{
        overview:    ['rankChart', 'mainChart'],
        iv:          ['ivChart'],
        theta:       ['thetaChart'],
        delta:       ['deltaChart'],
        gamma:       ['gammaChart'],
        theta15:     ['theta15Chart'],
        gex:         ['gexChart'],
        gexcombined: ['gexCombinedChart'],
        oidiff:      ['oiDiffChart'],
        basket:      ['basketChart', 'basketPnlChart'],
        otm:         ['cePriceChart', 'ceIvChart', 'ceThetaChart', 'ceTheta15Chart',
                       'pePriceChart', 'peIvChart', 'peThetaChart', 'peTheta15Chart']
    }};
    function showTab(name) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        document.getElementById('btn-' + name).classList.add('active');
        document.getElementById('strikeToggleBar').style.display = metricTabs.includes(name) ? 'flex' : 'none';
        // Plotly charts inside a tab that was just made visible were rendered
        // into a 0x0 (display:none) container, so they need an explicit resize
        // once the tab becomes visible - a generic window resize event alone
        // is not enough to fix this for every browser/Plotly version.
        (tabChartIds[name] || []).forEach(id => {{
            const el = document.getElementById(id);
            if (el && window.Plotly) {{
                try {{ Plotly.Plots.resize(el); }} catch (e) {{ /* chart not ready yet */ }}
            }}
        }});
        window.dispatchEvent(new Event('resize'));
    }}
    document.getElementById('strikeToggleBar').style.display = 'none';
    // Make sure the Overview tab's charts are sized correctly on first load too.
    (tabChartIds.overview || []).forEach(id => {{
        const el = document.getElementById(id);
        if (el && window.Plotly) {{
            try {{ Plotly.Plots.resize(el); }} catch (e) {{ /* chart not ready yet */ }}
        }}
    }});

    const metricChartIds = ['ivChart', 'thetaChart', 'deltaChart', 'gammaChart', 'theta15Chart'];
    function toggleStrike(idx, checked) {{
        metricChartIds.forEach(id => {{
            const el = document.getElementById(id);
            if (el && el.data) Plotly.restyle(id, {{ visible: checked }}, [idx]);
        }});
    }}

    setTimeout(function() {{ location.reload(); }}, 300000);
    </script>
</body>
</html>"""

    docs_path = "docs/index.html"
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    timestamp   = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(OUTPUT_DIR, f"dashboard_{timestamp}.html")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    return docs_path, backup_path

# --- DATA FETCHING ---

def fetch_candles(fyers, symbol, date_from, date_to=None, resolution="5"):
    """Fetch candles for symbol between date_from and date_to (inclusive).
    date_to defaults to date_from for a single-day fetch (backward compatible).
    """
    date_to = date_to or date_from
    data = {"symbol": symbol, "resolution": resolution, "date_format": "1",
            "range_from": date_from, "range_to": date_to, "cont_flag": "1"}
    try:
        resp = fyers.history(data=data)
        if resp.get("s") == "ok" and resp.get("candles"):
            df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} (resolution={resolution}, {date_from}->{date_to}): {e}")
    return pd.DataFrame()

def get_lookback_date_range(target_date, calendar_buffer=VWAP_LOOKBACK_CALENDAR_BUFFER):
    """Returns (range_from, range_to) strings to request from the history API,
    padded with extra calendar days so weekends/holidays don't shrink the
    number of actual trading days returned below VWAP_LOOKBACK_TRADING_DAYS.
    """
    target_d = datetime.strptime(target_date, "%Y-%m-%d").date()
    range_from = (target_d - timedelta(days=calendar_buffer)).strftime("%Y-%m-%d")
    return range_from, target_date

def resample_ohlcv(df, freq="5min"):
    if df.empty:
        return df
    return (
        df.set_index("time")
          .resample(freq)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna(subset=["close"])
          .reset_index()
    )

def compute_atm(fyers, spot_df, step=STRIKE_STEP, fallback=FALLBACK_ATM, use_live_quote=True, reference="last"):
    spot_price = None
    if use_live_quote:
        try:
            quote_resp = fyers.quotes(data={"symbols": SPOT_SYMBOL})
            if quote_resp.get("s") == "ok" and quote_resp.get("d"):
                spot_price = quote_resp["d"][0]["v"]["lp"]
        except Exception as e:
            logger.warning(f"Live quote fetch failed: {e}")
    if spot_price is None and not spot_df.empty:
        row = spot_df.iloc[0] if reference == "first" else spot_df.iloc[-1]
        spot_price = row["spot"]
    if spot_price is None or spot_price != spot_price:
        logger.warning(f"Could not determine spot price — fallback ATM {fallback}")
        return fallback
    atm = int(round(spot_price / step) * step)
    logger.info(f"Spot price: {spot_price} -> ATM strike: {atm}")
    return atm

def enrich_ce_pe(ce_df, pe_df, spot_df, strike, target_date=TARGET_DATE,
                  lookback_trading_days=VWAP_LOOKBACK_TRADING_DAYS):
    """Compute straddle, VWAP, EMA9, IV, Greeks and trailing Theta from CE/PE candles.

    ce_df/pe_df may span several trading days (see get_lookback_date_range).
    VWAP is a standard SESSION VWAP — it resets at the start of every trading
    day (the industry-standard definition, and what any broker/charting
    platform shows). Blending multiple days into one running average makes
    the line barely move during the day (today's volume gets diluted by 1-2
    full prior days already sitting in the sum), so each day gets its own
    independent cumulative VWAP. The extra lookback days are still fetched
    and used to warm-start EMA9 (avoids the cold-start jump/lag EMA9 would
    otherwise show in the first few candles of the session). The result is
    then trimmed down to just target_date rows before Greeks are computed
    (keeps compute cost down and matches what the dashboard displays for
    "today").
    """
    if ce_df.empty or pe_df.empty:
        return None
    merged = pd.merge(ce_df[['time','close','volume']], pe_df[['time','close','volume']], on='time')
    merged = merged.sort_values('time').reset_index(drop=True)
    merged['date'] = merged['time'].dt.date

    # Keep only the last `lookback_trading_days` trading days up to & including target_date
    target_d = datetime.strptime(target_date, "%Y-%m-%d").date()
    trading_days = sorted({d for d in merged['date'] if d <= target_d})
    trading_days = trading_days[-lookback_trading_days:] if trading_days else []
    if trading_days:
        merged = merged[merged['date'].isin(trading_days)].reset_index(drop=True)

    merged['straddle'] = merged['close_x'] + merged['close_y']
    merged['v']        = merged['volume_x'] + merged['volume_y']

    # Session VWAP: cumulative sum WITHIN each date group only, so it resets
    # every trading day instead of blending across days.
    pv      = merged['straddle'] * merged['v']
    pv_cum  = pv.groupby(merged['date']).cumsum()
    vol_cum = merged['v'].groupby(merged['date']).cumsum()
    merged['vwap'] = pv_cum / vol_cum.replace(0, np.nan)

    # EMA9 uses the full multi-day window as a warm start (reduces the "cold
    # start" lag you'd otherwise see in the first few candles of the session).
    merged['ema9'] = merged['straddle'].ewm(span=9, adjust=False).mean()

    # Trim to target_date only for display/Greeks.
    merged = merged[merged['date'] == target_d].drop(columns=['date']).reset_index(drop=True)
    if merged.empty:
        return None

    if not spot_df.empty:
        merged = pd.merge(merged, spot_df, on='time', how='left')
        merged['spot'] = merged['spot'].ffill()
        greeks = merged.apply(lambda row: compute_greeks_row(row, strike), axis=1)
        merged = pd.concat([merged, greeks], axis=1)
    else:
        merged['iv_pct'] = float('nan')
        merged['gamma_total'] = merged['vega_total'] = merged['theta_total'] = 0.0
        merged['theta_ce'] = merged['theta_pe'] = 0.0
        merged['delta_ce'] = merged['delta_pe'] = merged['delta_total'] = 0.0
    if len(merged) >= 2:
        inferred_interval = (merged['time'].iloc[1] - merged['time'].iloc[0]).total_seconds() / 60.0
    else:
        inferred_interval = CANDLE_INTERVAL_MINUTES
    candles_per_window = max(1, round(THETA_WINDOW_MINUTES / inferred_interval))
    scale = inferred_interval / 1440.0
    # Straddle trailing theta (existing)
    merged['theta_15min']    = (merged['theta_total'] * scale).rolling(window=candles_per_window, min_periods=1).sum()
    # Individual leg trailing theta for OTM analysis (NEW)
    merged['theta_ce_15min'] = (merged['theta_ce']    * scale).rolling(window=candles_per_window, min_periods=1).sum()
    merged['theta_pe_15min'] = (merged['theta_pe']    * scale).rolling(window=candles_per_window, min_periods=1).sum()
    return merged

def fetch_and_enrich_strike(fyers, strike, spot_df):
    ce_symbol = f"NSE:NIFTY{EXPIRY}{strike}CE"
    pe_symbol = f"NSE:NIFTY{EXPIRY}{strike}PE"
    range_from, range_to = get_lookback_date_range(TARGET_DATE)
    ce_df = fetch_candles(fyers, ce_symbol, range_from, range_to)
    pe_df = fetch_candles(fyers, pe_symbol, range_from, range_to)
    return enrich_ce_pe(ce_df, pe_df, spot_df, strike)


# --- MAIN ---

def main():
    if not CLIENT_ID or not TOKEN:
        logger.error("API Credentials missing.")
        return
    try:
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=TOKEN, log_path="")
        profile = fyers.get_profile()
        if profile.get("s") != "ok":
            logger.error(f"Authentication failed: {profile}")
            return
        logger.info(f"Successfully authenticated. Running analysis for {TARGET_DATE}")

        spot_df = fetch_candles(fyers, SPOT_SYMBOL, TARGET_DATE)
        if spot_df.empty:
            logger.warning("Could not fetch spot index data — IV/Greeks will be unavailable.")
        else:
            spot_df = spot_df[["time", "close"]].rename(columns={"close": "spot"})

        atm           = compute_atm(fyers, spot_df)
        fallback_spot = spot_df.iloc[-1]["spot"] if not spot_df.empty else None

        gex_history, gex_status = update_gex_and_get_history(fyers, fallback_spot)
        combined_gex_history     = update_combined_gex_and_get_history(fyers, fallback_spot)

        # ── Basket strategy (short straddle basket below VWAP) ──────────────
        basket_df, basket_trades, basket_open_position, basket_N = None, [], None, None
        try:
            basket_df, basket_N = build_basket_dataframe(fyers, TARGET_DATE, spot_df)
            if basket_df is not None:
                basket_trades, basket_open_position, basket_df = run_basket_strategy(basket_df)
                notify_basket_events(basket_trades, basket_open_position, basket_N)
                logger.info(
                    f"✓ Basket strategy: N={basket_N}  trades={len(basket_trades)}  "
                    f"open={'yes' if basket_open_position else 'no'}"
                )
            else:
                logger.warning("Skipping basket strategy this run (no basket data).")
        except Exception as e:
            logger.error(f"Basket strategy computation failed: {e}", exc_info=True)

        results = {}
        successful_fetches = 0
        for offset in OFFSETS:
            strike = atm + offset
            logger.info(f"Fetching data for strike {strike}")
            merged = fetch_and_enrich_strike(fyers, strike, spot_df)
            if merged is not None:
                results[strike] = merged
                successful_fetches += 1
                logger.info(f"✓ Successfully processed strike {strike}")
            else:
                logger.warning(f"✗ Failed to fetch data for strike {strike}")

        if results:
            rankings = compute_rankings(results)
            docs_path, backup_path = build_dashboard_html(
                results, atm, rankings, gex_history, combined_gex_history,
                basket_df=basket_df, basket_trades=basket_trades,
                basket_open_position=basket_open_position, basket_N=basket_N,
            )
            logger.info(f"✓ Dashboard generated: {docs_path}")

            gex_status_line = ""
            if gex_status:
                arrow = "🔼" if gex_status["position"] == "ABOVE" else ("🔽" if gex_status["position"] == "BELOW" else "▪️")
                gex_status_line = (
                    f"\n{arrow} Spot <b>{gex_status['spot']:.1f}</b> is <b>{gex_status['position']}</b> "
                    f"Gamma Flip (<b>{gex_status['flip']:.1f}</b>)  |  Net GEX: {gex_status['net_gex']/GEX_SCALE:.2f} Cr"
                )

            basket_status_line = ""
            if basket_N:
                open_pnl = basket_open_position['entry_price'] - basket_df['close'].iloc[-1] if (basket_open_position and basket_df is not None and not basket_df.empty) else None
                total_realized = sum(t['pnl_rupees'] for t in basket_trades) if basket_trades else 0.0
                basket_status_line = (
                    f"\n🧺 Basket ({basket_N-100}/{basket_N}/{basket_N+100}): "
                    f"{len(basket_trades)}/{BASKET_MAX_TRADES_PER_DAY} trades  |  Realized: ₹{total_realized:+,.0f}"
                )
                if open_pnl is not None:
                    basket_status_line += f"  |  Open: {open_pnl:+.2f} pts"

            send_telegram_document(
                docs_path,
                caption=(
                    f"📊 <b>Straddle Dashboard</b> — {TARGET_DATE}\n"
                    f"Open in browser for interactive charts."
                    f"{gex_status_line}"
                    f"{basket_status_line}"
                )
            )
            logger.info(f"✓ Successfully processed {successful_fetches}/{len(OFFSETS)} strikes")
        else:
            logger.error("No data fetched for any strike")
            send_telegram_message(f"❌ <b>Straddle Analyser Failed</b>\nDate: {TARGET_DATE}\nNo data fetched.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        send_telegram_message(f"❌ <b>Straddle Analyser Error</b>\nDate: {TARGET_DATE}\n<code>{str(e)}</code>")

if __name__ == "__main__":
    main()
