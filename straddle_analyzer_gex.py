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
EXPIRY = os.getenv("OPTION_EXPIRY_CODE", "26714")  # Update as needed (Fyers symbol expiry code)
EXPIRY_DATE = os.getenv("OPTION_EXPIRY_DATE", "2026-07-14")  # Update as needed - actual calendar expiry date, must match EXPIRY above
EXPIRY_TIME = "15:30"  # Market close time on expiry day
RISK_FREE_RATE = 0.065  # Annualised risk-free rate used for Black-Scholes / IV solving
SPOT_SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_STEP = 100  # Nifty weekly strikes are in steps of 50
FALLBACK_ATM = 24300  # Used only if the spot fetch fails
CANDLE_INTERVAL_MINUTES = 5  # Must match the "resolution" used in fetch_candles
THETA_WINDOW_MINUTES = 15  # Trailing window for the "15 Min Theta" tab
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

LOT_SIZE = 75  # Update to the current NIFTY lot size - used for GEX (rupee) scaling
GEX_STRIKE_COUNT = 40  # Strikes fetched on each side of ATM from the Option Chain API for GEX
GEX_SCALE = 1e7  # Display GEX in ₹ Crore (1e7) for readability
GEX_SPOT_RANGE_POINTS = 1000  # How far above/below spot to scan when solving for the gamma flip
GEX_SPOT_STEP = 25  # Grid step (points) for the flip scan

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

GEX_HISTORY_FILE = os.path.join(OUTPUT_DIR, f"gex_history_{TARGET_DATE}.json")
GEX_HISTORY_DOCS_FILE = os.path.join("docs", f"gex_history_{TARGET_DATE}.json")

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

EXPIRY_DT = datetime.combine(
    datetime.strptime(EXPIRY_DATE, "%Y-%m-%d").date(),
    dtime.fromisoformat(EXPIRY_TIME),
    tzinfo=ZoneInfo("Asia/Kolkata"),
)

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

# --- OPTIONS PRICING: IV & GREEKS (Black-Scholes) ---

def _time_to_expiry_years(candle_time):
    """candle_time is a tz-naive Asia/Kolkata timestamp. Returns years to expiry (>=0)."""
    aware = candle_time.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    seconds = (EXPIRY_DT - aware).total_seconds()
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
    """Solve for IV using Brent's method. Returns NaN if it can't be solved."""
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
    """Gamma and Vega (per 1% IV move) - identical formula for calls and puts."""
    if T <= 0 or sigma <= 0 or sigma != sigma:  # NaN check
        return 0.0, 0.0
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T) / 100.0
    return gamma, vega

def bs_theta(S, K, T, r, sigma, opt_type):
    """Theta expressed per calendar day."""
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
    """Given a merged row (with spot, ce close, pe close, time), compute IV & Greeks for the straddle."""
    S = row.get("spot")
    t_years = _time_to_expiry_years(row["time"])
    if S is None or S != S or S <= 0:
        return pd.Series({
            "iv_ce": float("nan"), "iv_pe": float("nan"), "iv_pct": float("nan"),
            "gamma_total": 0.0, "vega_total": 0.0, "theta_total": 0.0
        })

    iv_ce = implied_vol(row["close_x"], S, strike, t_years, RISK_FREE_RATE, "CE")
    iv_pe = implied_vol(row["close_y"], S, strike, t_years, RISK_FREE_RATE, "PE")

    gamma_ce, vega_ce = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_ce)
    gamma_pe, vega_pe = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_pe)
    theta_ce = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_ce, "CE")
    theta_pe = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_pe, "PE")

    ivs = [v for v in (iv_ce, iv_pe) if v == v]  # drop NaN
    iv_pct = (sum(ivs) / len(ivs) * 100.0) if ivs else float("nan")

    return pd.Series({
        "iv_ce": iv_ce, "iv_pe": iv_pe, "iv_pct": iv_pct,
        "gamma_total": (gamma_ce if gamma_ce == gamma_ce else 0.0) + (gamma_pe if gamma_pe == gamma_pe else 0.0),
        "vega_total": (vega_ce if vega_ce == vega_ce else 0.0) + (vega_pe if vega_pe == vega_pe else 0.0),
        "theta_total": (theta_ce if theta_ce == theta_ce else 0.0) + (theta_pe if theta_pe == theta_pe else 0.0),
    })

# --- GAMMA EXPOSURE (GEX) & GAMMA FLIP ---
#
# GEX convention used here (the common retail/open-source convention - NOT the
# only one in use, but the most widely seen): calls contribute POSITIVE gamma
# exposure, puts contribute NEGATIVE. Net GEX > 0 is read as a "pinned" /
# lower-realised-vol regime (dealers long gamma, dampen moves); Net GEX < 0 is
# read as a "trending" / higher-realised-vol regime (dealers short gamma,
# amplify moves). The Gamma Flip is the hypothetical spot level at which total
# GEX (recomputed at that spot, holding today's OI and each strike's already
# solved IV fixed) would cross zero.
#
# NOTE ON DATA SOURCE: Fyers' history() candle API has no Open Interest field.
# OI only comes from the Option Chain endpoint (fyers.optionchain), which is a
# live snapshot - there is no historical intraday OI series available. So GEX
# can only be computed for "right now" each time this script runs; the GEX
# tab's time series is built up by logging one point per run into a small
# JSON file that must persist across runs (see GEX_HISTORY_FILE below - your
# GitHub Actions workflow needs to commit this file back, the same way it
# already must be committing docs/index.html for GitHub Pages).
#
# CAVEAT: I'm parsing fyers.optionchain()'s response based on the commonly
# documented Fyers v3 schema, but I can't test this live. The parser below is
# defensive and logs the raw response if the expected fields aren't found -
# please sanity check that log line against your actual account once.

def fetch_option_chain(fyers, strike_count=GEX_STRIKE_COUNT):
    """Fetch a live option-chain snapshot (OI + LTP per strike) from Fyers.
    Returns (chain_df, spot_price) or (None, None) on failure. chain_df has
    one row per strike with columns: strike, ltp_ce, ltp_pe, oi_ce, oi_pe.
    """
    try:
        resp = fyers.optionchain(data={"symbol": SPOT_SYMBOL, "strikecount": strike_count, "timestamp": ""})
    except Exception as e:
        logger.error(f"Option chain fetch failed: {e}")
        return None, None

    if not isinstance(resp, dict) or resp.get("s") != "ok":
        logger.error(f"Option chain error response: {resp}")
        return None, None

    chain = (resp.get("data") or {}).get("optionsChain") or []
    if not chain:
        logger.warning(f"Option chain returned no rows. Raw response (truncated): {json.dumps(resp)[:800]}")
        return None, None

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
            # Usually the underlying index itself is included as a non-CE/PE row.
            maybe_spot = _get(rec, "ltp", "lp")
            if maybe_spot:
                spot_price = maybe_spot
            continue
        strike = _get(rec, "strike_price", "strikePrice", "strike")
        ltp = _get(rec, "ltp", "lp") or 0.0
        oi = _get(rec, "oi", "openInterest") or 0
        if strike is None:
            continue
        rows.append({"strike": strike, "option_type": opt_type, "ltp": ltp, "oi": oi})

    if not rows:
        logger.warning(f"Could not parse any option chain rows - schema mismatch. Sample records: {json.dumps(chain[:2])}")
        return None, None

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="strike", values=["ltp", "oi"], columns="option_type", aggfunc="first")
    pivot.columns = [f"{val}_{opt.lower()}" for val, opt in pivot.columns]
    pivot = pivot.reset_index()
    for col in ["ltp_ce", "ltp_pe", "oi_ce", "oi_pe"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    return pivot, spot_price

def compute_gex_snapshot(chain_df, spot, as_of=None):
    """Per-strike Gamma Exposure using our own BS gamma (IV solved from each
    leg's live LTP). Returns chain_df with iv/gamma/gex columns added.
    """
    as_of = as_of or datetime.now(ZoneInfo("Asia/Kolkata"))
    t_years = _time_to_expiry_years(as_of.replace(tzinfo=None))

    records = []
    for _, row in chain_df.iterrows():
        strike = row["strike"]
        iv_ce = implied_vol(row["ltp_ce"], spot, strike, t_years, RISK_FREE_RATE, "CE")
        iv_pe = implied_vol(row["ltp_pe"], spot, strike, t_years, RISK_FREE_RATE, "PE")
        gamma_ce, _ = bs_gamma_vega(spot, strike, t_years, RISK_FREE_RATE, iv_ce)
        gamma_pe, _ = bs_gamma_vega(spot, strike, t_years, RISK_FREE_RATE, iv_pe)

        oi_ce = row["oi_ce"] if row["oi_ce"] == row["oi_ce"] else 0.0
        oi_pe = row["oi_pe"] if row["oi_pe"] == row["oi_pe"] else 0.0

        gex_ce = gamma_ce * oi_ce * LOT_SIZE * spot ** 2 * 0.01
        gex_pe = -gamma_pe * oi_pe * LOT_SIZE * spot ** 2 * 0.01

        records.append({
            "strike": strike, "oi_ce": oi_ce, "oi_pe": oi_pe,
            "iv_ce": iv_ce, "iv_pe": iv_pe,
            "gamma_ce": gamma_ce, "gamma_pe": gamma_pe,
            "gex_ce": gex_ce, "gex_pe": gex_pe, "gex_net": gex_ce + gex_pe,
        })

    return pd.DataFrame(records)

def compute_gamma_flip(gex_df, spot, as_of=None):
    """Scan a grid of hypothetical spot levels around the current spot,
    recomputing gamma at each level (holding OI and each strike's already-
    solved IV fixed), and find where total GEX crosses zero via interpolation.
    Returns (flip_level_or_None, [(spot, total_gex), ...]).
    """
    as_of = as_of or datetime.now(ZoneInfo("Asia/Kolkata"))
    t_years = _time_to_expiry_years(as_of.replace(tzinfo=None))

    grid = np.arange(spot - GEX_SPOT_RANGE_POINTS, spot + GEX_SPOT_RANGE_POINTS + GEX_SPOT_STEP, GEX_SPOT_STEP)
    totals = []
    for s_hyp in grid:
        total = 0.0
        for _, row in gex_df.iterrows():
            strike = row["strike"]
            if row["iv_ce"] == row["iv_ce"]:
                g_ce, _ = bs_gamma_vega(s_hyp, strike, t_years, RISK_FREE_RATE, row["iv_ce"])
                total += g_ce * row["oi_ce"] * LOT_SIZE * s_hyp ** 2 * 0.01
            if row["iv_pe"] == row["iv_pe"]:
                g_pe, _ = bs_gamma_vega(s_hyp, strike, t_years, RISK_FREE_RATE, row["iv_pe"])
                total -= g_pe * row["oi_pe"] * LOT_SIZE * s_hyp ** 2 * 0.01
        totals.append(total)

    totals = np.array(totals)
    flip = None
    for i in range(len(totals) - 1):
        if totals[i] == 0:
            flip = float(grid[i])
            break
        if totals[i] * totals[i + 1] < 0:
            frac = totals[i] / (totals[i] - totals[i + 1])
            flip = float(grid[i] + frac * (grid[i + 1] - grid[i]))
            break

    return flip, list(zip(grid.tolist(), totals.tolist()))

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

def update_gex_and_get_history(fyers, fallback_spot):
    """Fetch a fresh option-chain snapshot, compute GEX + flip, append it to
    the persisted history log, and return the full day's history so far
    (list of dicts: time, spot, net_gex, flip). Never raises - logs and
    falls back to returning whatever history already exists on failure.
    """
    history = load_gex_history()
    try:
        chain_df, chain_spot = fetch_option_chain(fyers)
        spot = chain_spot or fallback_spot
        if chain_df is not None and spot:
            gex_df = compute_gex_snapshot(chain_df, spot)
            net_gex = float(gex_df["gex_net"].sum())
            flip, _curve = compute_gamma_flip(gex_df, spot)
            history.append({"time": RUN_TIMESTAMP, "spot": float(spot), "net_gex": net_gex, "flip": flip})
            save_gex_history(history)
            logger.info(f"✓ GEX snapshot: spot={spot:.1f} net_gex={net_gex/GEX_SCALE:.2f} Cr flip={flip}")
        else:
            logger.warning("Skipping GEX snapshot this run (no option chain / spot available).")
    except Exception as e:
        logger.error(f"GEX computation failed: {e}", exc_info=True)
    return history



def _metric_figure(straddle_data, strikes, atm, column, title, yaxis_title):
    fig = go.Figure()
    for idx, strike in enumerate(strikes):
        df = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        label = f"{strike}" + (" (ATM)" if strike == atm else "")
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[column], name=label, visible=True,
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        title=title, template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, yaxis_title=yaxis_title, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def build_dashboard_html(straddle_data, atm, rankings, gex_history=None):
    strikes = sorted(straddle_data.keys())
    gex_history = gex_history or []

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

        fig_main.add_trace(go.Scatter(
            x=df["time"], y=df["straddle"], name=f"{strike}",
            line=dict(color=color, width=2)
        ), row=row, col=1)

        fig_main.add_trace(go.Scatter(
            x=df["time"], y=df["vwap"], name="VWAP",
            line=dict(color=MUTED, width=1, dash="dot")
        ), row=row, col=1)

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

    # Greek figures - one trace per strike, in the SAME strike order so trace
    # indices line up across charts for the strike toggle checkboxes.
    fig_iv = _metric_figure(straddle_data, strikes, atm, "iv_pct", "Implied Volatility (Straddle Avg)", "IV (%)")
    fig_theta = _metric_figure(straddle_data, strikes, atm, "theta_total", "Theta (Straddle Total, ₹/day)", "Theta")
    fig_vega = _metric_figure(straddle_data, strikes, atm, "vega_total", "Vega (Straddle Total, per 1% IV)", "Vega")
    fig_gamma = _metric_figure(straddle_data, strikes, atm, "gamma_total", "Gamma (Straddle Total)", "Gamma")
    fig_theta15 = _metric_figure(straddle_data, strikes, atm, "theta_15min", f"Theta Decay (Trailing {THETA_WINDOW_MINUTES} min, ₹)", "Theta (₹ / 15 min)")

    # --- GEX chart: NIFTY spot + gamma flip level (price axis) and Net GEX (secondary axis) ---
    # Only has data from whenever this feature started running (OI has no history via the API).
    fig_gex = make_subplots(specs=[[{"secondary_y": True}]])
    if gex_history:
        gex_times = [h["time"] for h in gex_history]
        gex_spots = [h["spot"] for h in gex_history]
        gex_flips = [h["flip"] for h in gex_history]
        gex_nets = [h["net_gex"] / GEX_SCALE if h["net_gex"] is not None else None for h in gex_history]

        fig_gex.add_trace(go.Scatter(
            x=gex_times, y=gex_spots, name="NIFTY Spot",
            line=dict(color=BLUE, width=2)
        ), secondary_y=False)
        fig_gex.add_trace(go.Scatter(
            x=gex_times, y=gex_flips, name="Gamma Flip Level",
            line=dict(color=ACCENT, width=2, dash="dash")
        ), secondary_y=False)
        fig_gex.add_trace(go.Scatter(
            x=gex_times, y=gex_nets, name="Net GEX (₹ Cr)",
            line=dict(color=RED, width=2), fill="tozeroy", fillcolor="rgba(255,69,96,0.08)"
        ), secondary_y=True)
        fig_gex.add_trace(go.Scatter(
            x=gex_times, y=[0] * len(gex_times), name="Zero GEX",
            line=dict(color=MUTED, width=1, dash="dot"), showlegend=False
        ), secondary_y=True)

    fig_gex.update_layout(
        title="NIFTY Spot vs Gamma Flip Level  |  Net GEX (₹ Cr)",
        template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15)
    )
    fig_gex.update_yaxes(title_text="NIFTY Price", secondary_y=False)
    fig_gex.update_yaxes(title_text="Net GEX (₹ Cr)", secondary_y=True)

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

    # Strike toggle checkboxes (shared across the IV / Theta / Vega / Gamma tabs)
    toggle_items_html = "".join([
        f"""<label class="toggle-item">
            <input type="checkbox" checked onchange="toggleStrike({idx}, this.checked)">
            <span class="strike-dot" style="background:{STRIKE_COLORS[idx % len(STRIKE_COLORS)]};"></span>
            {strike}{' <small style="color:' + ACCENT + '">ATM</small>' if strike == atm else ''}
        </label>""" for idx, strike in enumerate(strikes)
    ])

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
        <button class="tab-btn active" id="btn-overview" onclick="showTab('overview')">OVERVIEW</button>
        <button class="tab-btn" id="btn-iv" onclick="showTab('iv')">IMPLIED VOL</button>
        <button class="tab-btn" id="btn-theta" onclick="showTab('theta')">THETA</button>
        <button class="tab-btn" id="btn-vega" onclick="showTab('vega')">VEGA</button>
        <button class="tab-btn" id="btn-gamma" onclick="showTab('gamma')">GAMMA</button>
        <button class="tab-btn" id="btn-theta15" onclick="showTab('theta15')">15 MIN THETA</button>
        <button class="tab-btn" id="btn-gex" onclick="showTab('gex')">GEX &amp; FLIP</button>
        <button class="tab-btn" id="btn-momentum" onclick="showTab('momentum')">MOMENTUM</button>
    </div>

    <div class="toggle-bar" id="strikeToggleBar">
        {toggle_items_html}
    </div>

    <div class="tab-content active" id="tab-overview">
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

    <div class="tab-content" id="tab-vega">
        <div class="metric-card">
            {fig_vega.to_html(full_html=False, include_plotlyjs=False, div_id='vegaChart', config={'responsive': True})}
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
                Net GEX &gt; 0 → dealers net long gamma (moves tend to get dampened / "pinned").
                Net GEX &lt; 0 → dealers net short gamma (moves tend to accelerate / trend).
                Only tracks from whenever this feature started running today - Open Interest has no historical time series via the broker API.
            </div>
            {fig_gex.to_html(full_html=False, include_plotlyjs=False, div_id='gexChart', config={'responsive': True})}
        </div>
    </div>

    <div class="tab-content" id="tab-momentum">
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

    // --- Tabs ---
    const metricTabs = ['iv', 'theta', 'vega', 'gamma', 'theta15'];
    function showTab(name) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        document.getElementById('btn-' + name).classList.add('active');
        document.getElementById('strikeToggleBar').style.display = metricTabs.includes(name) ? 'flex' : 'none';
        window.dispatchEvent(new Event('resize'));
    }}
    document.getElementById('strikeToggleBar').style.display = 'none';

    // --- Strike toggle (shared across IV / Theta / Vega / Gamma tabs) ---
    const metricChartIds = ['ivChart', 'thetaChart', 'vegaChart', 'gammaChart', 'theta15Chart'];
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

    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(OUTPUT_DIR, f"dashboard_{timestamp}.html")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return docs_path, backup_path

# --- DATA FETCHING ---

def fetch_candles(fyers, symbol, date, resolution="5"):
    data = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": date, "range_to": date, "cont_flag": "1"}
    try:
        resp = fyers.history(data=data)
        if resp.get("s") == "ok" and resp.get("candles"):
            df = pd.DataFrame(resp["candles"], columns=["epoch","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} (resolution={resolution}): {e}")
    return pd.DataFrame()

def resample_ohlcv(df, freq="5min"):
    """Resample a 1-min (or finer) OHLCV candle dataframe (with a 'time' column)
    up to a coarser frequency, e.g. 5-min bars built from 1-min candles.
    NSE's 09:15 session open is already aligned to 5-min boundaries from
    midnight, so a plain pandas resample lines up with Fyers' native 5-min bars.
    """
    if df.empty:
        return df
    out = (
        df.set_index("time")
          .resample(freq)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna(subset=["close"])
          .reset_index()
    )
    return out

def compute_atm(fyers, spot_df, step=STRIKE_STEP, fallback=FALLBACK_ATM, use_live_quote=True, reference="last"):
    """Determine the ATM strike from the available spot price.
    Live use (use_live_quote=True): tries a live quote first (most accurate
    intraday), falls back to the last close in spot_df, then FALLBACK_ATM.
    Backtest use (use_live_quote=False): a live quote would return TODAY's
    price, not the historical day's, so it must be skipped. `reference`
    controls whether to anchor off the first ('first', i.e. day-open - what
    a live run would have seen that morning) or last ('last') candle.
    """
    spot_price = None
    if use_live_quote:
        try:
            quote_resp = fyers.quotes(data={"symbols": SPOT_SYMBOL})
            if quote_resp.get("s") == "ok" and quote_resp.get("d"):
                spot_price = quote_resp["d"][0]["v"]["lp"]
        except Exception as e:
            logger.warning(f"Live quote fetch failed, will fall back to candle data: {e}")

    if spot_price is None and not spot_df.empty:
        row = spot_df.iloc[0] if reference == "first" else spot_df.iloc[-1]
        spot_price = row["spot"]

    if spot_price is None or spot_price != spot_price:  # None or NaN
        logger.warning(f"Could not determine spot price - falling back to hardcoded ATM {fallback}")
        return fallback

    atm = int(round(spot_price / step) * step)
    logger.info(f"Spot price: {spot_price} -> ATM strike: {atm}")
    return atm

def enrich_ce_pe(ce_df, pe_df, spot_df, strike):
    """Given already-fetched CE/PE candle dataframes (each with time, close,
    volume - any resolution, as long as both are on the same time grid) plus
    a spot dataframe, compute straddle price, VWAP, EMA9, IV, Gamma, Vega,
    Theta (per-day) and Theta (trailing 15-min). Returns None if CE/PE are
    empty. This is the shared core used by both the live dashboard and the
    backtester (which may feed it 5-min bars resampled from 1-min data).
    """
    if ce_df.empty or pe_df.empty:
        return None

    merged = pd.merge(ce_df[['time','close','volume']], pe_df[['time','close','volume']], on='time')
    merged['straddle'] = merged['close_x'] + merged['close_y']
    merged['v'] = merged['volume_x'] + merged['volume_y']
    merged['vwap'] = (merged['straddle'] * merged['v']).cumsum() / merged['v'].cumsum()
    merged['ema9'] = merged['straddle'].ewm(span=9, adjust=False).mean()

    if not spot_df.empty:
        merged = pd.merge(merged, spot_df, on='time', how='left')
        merged['spot'] = merged['spot'].ffill()
        greeks = merged.apply(lambda row: compute_greeks_row(row, strike), axis=1)
        merged = pd.concat([merged, greeks], axis=1)
    else:
        merged['iv_pct'] = float('nan')
        merged['gamma_total'] = 0.0
        merged['vega_total'] = 0.0
        merged['theta_total'] = 0.0

    # Trailing 15-minute theta decay: theta_total is a per-calendar-day
    # (annualised/365) figure, so scale it down to a per-candle contribution
    # (using the actual spacing of the data passed in), then take a rolling
    # sum over enough candles to cover THETA_WINDOW_MINUTES.
    if len(merged) >= 2:
        inferred_interval_minutes = (merged['time'].iloc[1] - merged['time'].iloc[0]).total_seconds() / 60.0
    else:
        inferred_interval_minutes = CANDLE_INTERVAL_MINUTES
    candles_per_window = max(1, round(THETA_WINDOW_MINUTES / inferred_interval_minutes))
    theta_per_candle = merged['theta_total'] * (inferred_interval_minutes / 1440.0)
    merged['theta_15min'] = theta_per_candle.rolling(window=candles_per_window, min_periods=1).sum()

    return merged

def fetch_and_enrich_strike(fyers, strike, spot_df):
    """Fetch CE+PE 5-min candles for a strike from the API and enrich them.
    Used by the live dashboard (main()). The backtester instead resamples
    1-min data down to 5-min and calls enrich_ce_pe() directly.
    """
    ce_symbol = f"NSE:NIFTY{EXPIRY}{strike}CE"
    pe_symbol = f"NSE:NIFTY{EXPIRY}{strike}PE"

    ce_df = fetch_candles(fyers, ce_symbol, TARGET_DATE)
    pe_df = fetch_candles(fyers, pe_symbol, TARGET_DATE)

    return enrich_ce_pe(ce_df, pe_df, spot_df, strike)

    return merged

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

        # Spot index candles - needed to compute IV / Greeks AND to derive ATM
        spot_df = fetch_candles(fyers, SPOT_SYMBOL, TARGET_DATE)
        if spot_df.empty:
            logger.warning("Could not fetch spot index data - IV/Greeks will be unavailable.")
        else:
            spot_df = spot_df[["time", "close"]].rename(columns={"close": "spot"})

        atm = compute_atm(fyers, spot_df)

        fallback_spot = spot_df.iloc[-1]["spot"] if not spot_df.empty else None
        gex_history = update_gex_and_get_history(fyers, fallback_spot)

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
            docs_path, backup_path = build_dashboard_html(results, atm, rankings, gex_history)
            logger.info(f"✓ Dashboard generated: {docs_path}")

            # Send to Telegram
            #send_telegram_summary(rankings, atm, successful_fetches, len(OFFSETS))
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
