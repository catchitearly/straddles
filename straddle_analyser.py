import os
import math
import json
import logging
import requests
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
EXPIRY = os.getenv("OPTION_EXPIRY_CODE", "26707")
EXPIRY_DATE = os.getenv("OPTION_EXPIRY_DATE", "2026-07-07")
EXPIRY_TIME = "15:30"
RISK_FREE_RATE = 0.065
SPOT_SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_STEP = 100
FALLBACK_ATM = 24300
CANDLE_INTERVAL_MINUTES = 5
THETA_WINDOW_MINUTES = 15
OFFSETS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]

# ── NEW ──────────────────────────────────────────────────────────────────────
LOT_SIZE = 65          # Current Nifty lot size (as of Sep 2025 expiry revision)
GEX_SCALE = 1e7        # Divide raw GEX by this → result in crores
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

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
GEX_GREEN = "#00e5b0"   # positive gamma / CE GEX
GEX_RED   = "#ff4560"   # negative gamma / PE GEX
GEX_FLIP  = "#fbbf24"   # flip level line
STRIKE_COLORS = ["#38bdf8","#a78bfa","#f97316","#ff4560","#fbbf24","#00e5b0","#ec4899","#84cc16","#64748b"]

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
    lines += ["", f"✅ <b>Best Strike:</b> {top['strike']} (Smoothness: {top['smoothness']}%)", f"📎 Full dashboard attached below."]
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

def _time_to_expiry_years(candle_time):
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
            "gamma_ce": 0.0, "gamma_pe": 0.0,          # ── NEW: individual gammas
            "gamma_total": 0.0, "vega_total": 0.0, "theta_total": 0.0
        })

    iv_ce = implied_vol(row["close_x"], S, strike, t_years, RISK_FREE_RATE, "CE")
    iv_pe = implied_vol(row["close_y"], S, strike, t_years, RISK_FREE_RATE, "PE")

    gamma_ce, vega_ce = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_ce)
    gamma_pe, vega_pe = bs_gamma_vega(S, strike, t_years, RISK_FREE_RATE, iv_pe)
    theta_ce = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_ce, "CE")
    theta_pe = bs_theta(S, strike, t_years, RISK_FREE_RATE, iv_pe, "PE")

    ivs = [v for v in (iv_ce, iv_pe) if v == v]
    iv_pct = (sum(ivs) / len(ivs) * 100.0) if ivs else float("nan")

    return pd.Series({
        "iv_ce": iv_ce, "iv_pe": iv_pe, "iv_pct": iv_pct,
        "gamma_ce": gamma_ce if gamma_ce == gamma_ce else 0.0,   # ── NEW
        "gamma_pe": gamma_pe if gamma_pe == gamma_pe else 0.0,   # ── NEW
        "gamma_total": (gamma_ce if gamma_ce == gamma_ce else 0.0) + (gamma_pe if gamma_pe == gamma_pe else 0.0),
        "vega_total":  (vega_ce  if vega_ce  == vega_ce  else 0.0) + (vega_pe  if vega_pe  == vega_pe  else 0.0),
        "theta_total": (theta_ce if theta_ce == theta_ce else 0.0) + (theta_pe if theta_pe == theta_pe else 0.0),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# NEW — GEX (GAMMA EXPOSURE) ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(v):
    """Return float or 0.0, never NaN."""
    try:
        f = float(v)
        return 0.0 if f != f else f
    except Exception:
        return 0.0

def compute_gex_all_candles(straddle_data):
    """
    Pre-compute GEX snapshots at every candle across all loaded strikes.

    GEX per strike per candle:
        GEX_CE(K) = +gamma_CE(K) × OI_CE(K) × S² × LOT_SIZE   (positive — call side)
        GEX_PE(K) = −gamma_PE(K) × OI_PE(K) × S² × LOT_SIZE   (negative — put side)
        GEX_net(K) = GEX_CE(K) + GEX_PE(K)

    Gamma Flip Level: strike where cumulative net GEX (low→high) changes sign.
    Above flip → positive gamma regime (MM long gamma → mean-revert).
    Below flip → negative gamma regime (MM short gamma → trend / vol expansion).

    Returns a list of dicts (one per candle), each containing:
        time, spot, flip_level, total_net_gex,
        gex_ce[]  (one value per strike, ₹Cr),
        gex_pe[]  (one value per strike, ₹Cr),
        gex_net[] (one value per strike, ₹Cr),

    NOTE: OI comes from the 7th column of Fyers F&O history candles (oi_ce /
    oi_pe fields set by fetch_and_enrich_strike). If the broker returns zero OI
    for every bar the GEX chart will be flat — in that case verify that
    Fyers history is returning the OI column for the selected instrument.
    """
    strikes = sorted(straddle_data.keys())
    ref_df  = straddle_data[strikes[0]]
    n_candles = len(ref_df)

    snapshots = []

    for t_idx in range(n_candles):
        snap_time = ref_df.iloc[t_idx]["time"]
        t_str     = snap_time.strftime("%H:%M")
        t_years   = _time_to_expiry_years(snap_time)

        gex_ce_list  = []
        gex_pe_list  = []
        spot_val     = None

        for strike in strikes:
            df = straddle_data[strike]
            if t_idx >= len(df):
                gex_ce_list.append(0.0)
                gex_pe_list.append(0.0)
                continue

            row    = df.iloc[t_idx]
            S      = _safe_float(row.get("spot", 0))
            if S > 0 and spot_val is None:
                spot_val = S

            if S <= 0:
                gex_ce_list.append(0.0)
                gex_pe_list.append(0.0)
                continue

            # Use pre-computed individual gammas from compute_greeks_row
            gamma_ce = _safe_float(row.get("gamma_ce", 0.0))
            gamma_pe = _safe_float(row.get("gamma_pe", 0.0))
            oi_ce    = _safe_float(row.get("oi_ce", 0))
            oi_pe    = _safe_float(row.get("oi_pe", 0))

            scale = (S ** 2) * LOT_SIZE / GEX_SCALE   # units: ₹ Crore
            gex_ce_list.append(round( gamma_ce * oi_ce * scale, 4))
            gex_pe_list.append(round(-gamma_pe * oi_pe * scale, 4))

        gex_net   = [round(c + p, 4) for c, p in zip(gex_ce_list, gex_pe_list)]
        total_net = round(sum(gex_net), 4)

        # ── Gamma Flip Level ─────────────────────────────────────────────────
        # Walk cumulative net GEX from the lowest strike to highest.
        # The flip is where the running sum crosses zero.
        flip_level = None
        cumulative  = 0.0
        for i, (s, g) in enumerate(zip(strikes, gex_net)):
            prev_cum   = cumulative
            cumulative += g
            if i > 0 and prev_cum != 0.0 and (prev_cum * cumulative) <= 0.0:
                denom = abs(prev_cum) + abs(cumulative)
                frac  = abs(prev_cum) / denom if denom > 0 else 0.5
                flip_level = round(strikes[i - 1] + frac * (s - strikes[i - 1]), 1)
                break
        # ─────────────────────────────────────────────────────────────────────

        snapshots.append({
            "time":          t_str,
            "spot":          round(spot_val, 2) if spot_val else None,
            "flip_level":    flip_level,
            "total_net_gex": total_net,
            "gex_ce":        gex_ce_list,
            "gex_pe":        gex_pe_list,
            "gex_net":       gex_net,
        })

    return snapshots


def build_gex_timeseries_figure(gex_snapshots):
    """
    Static Plotly figure with two sub-panels:
      Row 1 — Nifty Spot (blue) vs Gamma Flip Level (amber dashed)
      Row 2 — Total Net GEX bar chart (green above zero / red below)
    """
    times     = [s["time"]          for s in gex_snapshots]
    spots     = [s["spot"]          for s in gex_snapshots]
    flips     = [s["flip_level"]    for s in gex_snapshots]
    net_gex   = [s["total_net_gex"] for s in gex_snapshots]

    bar_colors = [GEX_GREEN if v >= 0 else GEX_RED for v in net_gex]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Nifty Spot  vs  Gamma Flip Level", "Net GEX  (₹ Cr)"],
        vertical_spacing=0.10,
        row_heights=[0.60, 0.40],
    )

    # Spot
    fig.add_trace(go.Scatter(
        x=times, y=spots, name="Nifty Spot",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="Time: %{x}<br>Spot: %{y:.0f}<extra></extra>",
    ), row=1, col=1)

    # Flip level
    fig.add_trace(go.Scatter(
        x=times, y=flips, name="Flip Level",
        line=dict(color=GEX_FLIP, width=2, dash="dash"),
        connectgaps=True,
        hovertemplate="Time: %{x}<br>Flip: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    # Shaded gap between spot & flip to highlight regime
    # (filled area — spot above flip = green tint, below = red tint)
    # We add two separate fills: spot above and spot below
    fig.add_trace(go.Scatter(
        x=times + times[::-1],
        y=spots + flips[::-1],
        fill="toself",
        fillcolor="rgba(0,229,176,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
        name="_fill",
    ), row=1, col=1)

    # Net GEX bars
    fig.add_trace(go.Bar(
        x=times, y=net_gex, name="Net GEX",
        marker_color=bar_colors,
        hovertemplate="Time: %{x}<br>Net GEX: %{y:.2f} Cr<extra></extra>",
    ), row=2, col=1)

    # Zero line
    fig.add_hline(y=0, line_color=MUTED, line_dash="dot", line_width=1, row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.05),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="GEX (₹ Cr)", row=2, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

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


def build_dashboard_html(straddle_data, atm, rankings):
    strikes = sorted(straddle_data.keys())

    # ── Pre-compute GEX data ──────────────────────────────────────────────────
    logger.info("Computing GEX data for all candles…")
    gex_snapshots = compute_gex_all_candles(straddle_data)
    fig_gex_ts    = build_gex_timeseries_figure(gex_snapshots)

    # Summary: last (most recent) snapshot
    last_snap       = gex_snapshots[-1]
    last_spot       = last_snap.get("spot") or 0
    last_flip       = last_snap.get("flip_level")
    last_total_gex  = last_snap.get("total_net_gex", 0)

    if last_flip is not None:
        gamma_regime      = "POSITIVE GAMMA" if last_spot >= last_flip else "NEGATIVE GAMMA"
        gamma_regime_color = GEX_GREEN         if last_spot >= last_flip else GEX_RED
        gamma_regime_tip  = "Market makers are LONG gamma → expect mean-reversion / range-bound action." \
                            if last_spot >= last_flip else \
                            "Market makers are SHORT gamma → expect trending / volatile moves."
        flip_display      = f"{last_flip:.0f}"
    else:
        gamma_regime, gamma_regime_color = "UNKNOWN", MUTED
        gamma_regime_tip  = "Flip level could not be determined (insufficient OI data?)."
        flip_display      = "N/A"
    # ─────────────────────────────────────────────────────────────────────────

    fig_main = make_subplots(
        rows=len(strikes), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Strike {s}{' ◄ ATM' if s == atm else ''}" for s in strikes]
    )
    for idx, strike in enumerate(strikes):
        df    = straddle_data[strike]
        color = STRIKE_COLORS[idx % len(STRIKE_COLORS)]
        row   = idx + 1
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["straddle"], name=f"{strike}",
                                       line=dict(color=color, width=2)), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["vwap"],     name="VWAP",
                                       line=dict(color=MUTED, width=1, dash="dot")), row=row, col=1)
        fig_main.add_trace(go.Scatter(x=df["time"], y=df["ema9"],     name="EMA9",
                                       line=dict(color=EMA_COLOR, width=1.5, dash="dash")), row=row, col=1)
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

    fig_iv      = _metric_figure(straddle_data, strikes, atm, "iv_pct",      "Implied Volatility (Straddle Avg)", "IV (%)")
    fig_theta   = _metric_figure(straddle_data, strikes, atm, "theta_total", "Theta (Straddle Total, ₹/day)",     "Theta")
    fig_vega    = _metric_figure(straddle_data, strikes, atm, "vega_total",  "Vega (Straddle Total, per 1% IV)",  "Vega")
    fig_gamma   = _metric_figure(straddle_data, strikes, atm, "gamma_total", "Gamma (Straddle Total)",            "Gamma")
    fig_theta15 = _metric_figure(straddle_data, strikes, atm, "theta_15min", f"Theta Decay (Trailing {THETA_WINDOW_MINUTES} min, ₹)", "Theta (₹ / 15 min)")

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
    all_times  = [d["time"] for d in speed_data[ref_strike]]

    toggle_items_html = "".join([
        f"""<label class="toggle-item">
            <input type="checkbox" checked onchange="toggleStrike({idx}, this.checked)">
            <span class="strike-dot" style="background:{STRIKE_COLORS[idx % len(STRIKE_COLORS)]};"></span>
            {strike}{' <small style="color:' + ACCENT + '">ATM</small>' if strike == atm else ''}
        </label>""" for idx, strike in enumerate(strikes)
    ])

    # ── GEX bar-chart summary rows (last snapshot) ────────────────────────────
    gex_table_rows = "".join([
        f"""<tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:8px;color:{STRIKE_COLORS[i % len(STRIKE_COLORS)]};"><b>{strike}</b>
                {' <small style="color:' + ACCENT + '">ATM</small>' if strike == atm else ''}</td>
            <td style="padding:8px;color:{GEX_GREEN};">{last_snap['gex_ce'][i]:+.3f}</td>
            <td style="padding:8px;color:{GEX_RED};">{last_snap['gex_pe'][i]:+.3f}</td>
            <td style="padding:8px;color:{GEX_GREEN if last_snap['gex_net'][i]>=0 else GEX_RED};font-weight:bold;">{last_snap['gex_net'][i]:+.3f}</td>
        </tr>""" for i, strike in enumerate(strikes)
    ])

    # ── Embed GEX snapshots as JSON for the interactive JS bar chart ──────────
    gex_json = json.dumps(gex_snapshots)
    strikes_json = json.dumps([str(s) for s in strikes])

    final_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="300">
    <title>Straddle Dashboard — {TARGET_DATE}</title>
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

        /* ── GEX-specific styles ── */
        .gex-outer {{ padding: 20px; }}
        .gex-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        @media (max-width: 1100px) {{ .gex-grid {{ grid-template-columns: 1fr; }} }}
        .gex-regime-banner {{
            border-radius: 8px; padding: 16px 24px; margin-bottom: 20px;
            display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
            border: 1px solid;
        }}
        .regime-badge {{
            font-size: 18px; font-weight: 900; letter-spacing: 2px; padding: 6px 16px;
            border-radius: 6px;
        }}
        .gex-stat {{ display: flex; flex-direction: column; gap: 2px; }}
        .gex-stat-label {{ font-size: 10px; color: {MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
        .gex-stat-value {{ font-size: 16px; font-weight: bold; }}
        .gex-bar-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; }}
        .gex-ts-card  {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; }}
        .gex-table-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .gex-slider-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
        #gexTimeVal {{ font-size: 16px; font-weight: bold; color: {ACCENT}; min-width: 50px; }}
        #gexSpotVal {{ font-size: 13px; color: {BLUE}; }}
        #gexFlipVal {{ font-size: 13px; color: {GEX_FLIP}; }}
        #gexNetVal  {{ font-size: 13px; font-weight: bold; }}

        @media (max-width: 768px) {{
            .content {{ flex-direction: column; }}
            .sidebar, .main-charts {{ min-width: auto; }}
            .speed-controls {{ flex-direction: column; align-items: flex-start; }}
            input[type=range] {{ width: 100%; }}
            .gex-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div><span style="color:{ACCENT};font-size:24px;">▣</span> <b style="font-size:20px;letter-spacing:2px;">STRADDLE ANALYSER</b></div>
        <div class="timestamp">Last Update: {RUN_TIMESTAMP}<br>Auto-refresh: Every 5 min</div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" id="btn-overview"  onclick="showTab('overview')">OVERVIEW</button>
        <button class="tab-btn"        id="btn-iv"        onclick="showTab('iv')">IMPLIED VOL</button>
        <button class="tab-btn"        id="btn-theta"     onclick="showTab('theta')">THETA</button>
        <button class="tab-btn"        id="btn-vega"      onclick="showTab('vega')">VEGA</button>
        <button class="tab-btn"        id="btn-gamma"     onclick="showTab('gamma')">GAMMA</button>
        <button class="tab-btn"        id="btn-theta15"   onclick="showTab('theta15')">15 MIN THETA</button>
        <button class="tab-btn"        id="btn-gex"       onclick="showTab('gex')" style="color:#fbbf24;">⚡ GEX</button>
        <button class="tab-btn"        id="btn-momentum"  onclick="showTab('momentum')">MOMENTUM</button>
    </div>

    <div class="toggle-bar" id="strikeToggleBar">
        {toggle_items_html}
    </div>

    <!-- ════════ OVERVIEW TAB ════════ -->
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

    <!-- ════════ IV / THETA / VEGA / GAMMA / THETA15 TABS ════════ -->
    <div class="tab-content" id="tab-iv">
        <div class="metric-card">{fig_iv.to_html(full_html=False, include_plotlyjs=False, div_id='ivChart', config={{'responsive':True}})}</div>
    </div>
    <div class="tab-content" id="tab-theta">
        <div class="metric-card">{fig_theta.to_html(full_html=False, include_plotlyjs=False, div_id='thetaChart', config={{'responsive':True}})}</div>
    </div>
    <div class="tab-content" id="tab-vega">
        <div class="metric-card">{fig_vega.to_html(full_html=False, include_plotlyjs=False, div_id='vegaChart', config={{'responsive':True}})}</div>
    </div>
    <div class="tab-content" id="tab-gamma">
        <div class="metric-card">{fig_gamma.to_html(full_html=False, include_plotlyjs=False, div_id='gammaChart', config={{'responsive':True}})}</div>
    </div>
    <div class="tab-content" id="tab-theta15">
        <div class="metric-card">{fig_theta15.to_html(full_html=False, include_plotlyjs=False, div_id='theta15Chart', config={{'responsive':True}})}</div>
    </div>

    <!-- ════════════════════════════════════════════════════════════
         GEX TAB
         ════════════════════════════════════════════════════════════ -->
    <div class="tab-content" id="tab-gex">
        <div class="gex-outer">

            <!-- Regime Banner -->
            <div class="gex-regime-banner"
                 style="background:{gamma_regime_color}11; border-color:{gamma_regime_color}44;">
                <div class="regime-badge"
                     style="background:{gamma_regime_color}22; color:{gamma_regime_color};">
                    {gamma_regime}
                </div>
                <div class="gex-stat">
                    <span class="gex-stat-label">Gamma Flip Level</span>
                    <span class="gex-stat-value" style="color:{GEX_FLIP};">{flip_display}</span>
                </div>
                <div class="gex-stat">
                    <span class="gex-stat-label">Nifty Spot (last bar)</span>
                    <span class="gex-stat-value" style="color:{BLUE};">{last_spot:.0f}</span>
                </div>
                <div class="gex-stat">
                    <span class="gex-stat-label">Net GEX (₹ Cr)</span>
                    <span class="gex-stat-value"
                          style="color:{GEX_GREEN if last_total_gex>=0 else GEX_RED};">
                        {last_total_gex:+.2f}
                    </span>
                </div>
                <div style="flex:1; min-width:200px; font-size:12px; color:{MUTED}; border-left:1px solid {BORDER}; padding-left:20px;">
                    {gamma_regime_tip}
                </div>
            </div>

            <!-- Two-column: GEX Profile (interactive) | Spot vs Flip (static) -->
            <div class="gex-grid">

                <!-- Left: Interactive GEX Profile bar chart -->
                <div class="gex-bar-card">
                    <h2>GEX PROFILE BY STRIKE</h2>
                    <div class="gex-slider-row">
                        <div class="slider-group">
                            <div class="slider-label">Select Time</div>
                            <input type="range" id="gexSlider" min="0" max="1" value="0" step="1" style="width:260px;">
                        </div>
                        <div>
                            <span id="gexTimeVal">--:--</span>
                            &nbsp;|&nbsp; Spot: <span id="gexSpotVal">--</span>
                            &nbsp;|&nbsp; Flip: <span id="gexFlipVal">--</span>
                            &nbsp;|&nbsp; Net GEX: <span id="gexNetVal">--</span>
                        </div>
                    </div>
                    <div id="gexBarDiv" style="width:100%;height:420px;"></div>
                    <div style="margin-top:10px;font-size:11px;color:{MUTED};">
                        <span style="color:{GEX_GREEN};">■</span> CE GEX (positive, dealer short call hedge)&nbsp;&nbsp;
                        <span style="color:{GEX_RED};">■</span> PE GEX (negative, dealer long put hedge)&nbsp;&nbsp;
                        <span style="color:{TEXT};">◆</span> Net GEX per strike
                    </div>
                </div>

                <!-- Right: Spot vs Flip Level + Net GEX time series (static Plotly) -->
                <div class="gex-ts-card">
                    <h2>NIFTY SPOT vs FLIP LEVEL  ·  NET GEX TIMELINE</h2>
                    {fig_gex_ts.to_html(full_html=False, include_plotlyjs=False, div_id='gexTsChart', config={{'responsive':True}})}
                </div>

            </div>

            <!-- GEX summary table (last snapshot) -->
            <div class="gex-table-card">
                <h2>GEX BY STRIKE — LAST CANDLE SNAPSHOT (₹ Cr)</h2>
                <div style="overflow-x:auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Strike</th>
                                <th style="color:{GEX_GREEN};">CE GEX</th>
                                <th style="color:{GEX_RED};">PE GEX</th>
                                <th>Net GEX</th>
                            </tr>
                        </thead>
                        <tbody>{gex_table_rows}</tbody>
                    </table>
                </div>
                <div style="margin-top:12px;font-size:11px;color:{MUTED};">
                    Units: ₹ Crore &nbsp;·&nbsp; Lot size: {LOT_SIZE} &nbsp;·&nbsp;
                    Formula: Gamma × OI × S² × {LOT_SIZE} ÷ 10⁷ &nbsp;·&nbsp;
                    Zero OI values → check Fyers history returns OI column for F&O symbols.
                </div>
            </div>

        </div>
    </div>
    <!-- ── end GEX tab ────────────────────────────────────────────────────── -->

    <!-- ════════ MOMENTUM TAB ════════ -->
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
    // ── Shared data ──────────────────────────────────────────────────────────
    const speedData  = {json.dumps(speed_data)};
    const allTimes   = {json.dumps(all_times)};
    const strikesArr = {strikes_json};
    const colors     = {json.dumps(STRIKE_COLORS)};
    const ATM        = "{atm}";

    // ── GEX data (pre-computed Python-side) ──────────────────────────────────
    const gexSnapshots = {gex_json};
    const GEX_GREEN    = "{GEX_GREEN}";
    const GEX_RED      = "{GEX_RED}";
    const GEX_FLIP_C   = "{GEX_FLIP}";
    const BLUE_C       = "{BLUE}";
    const MUTED_C      = "{MUTED}";
    const TEXT_C       = "{TEXT}";
    const BORDER_C     = "{BORDER}";
    const CARD_C       = "{CARD}";
    const BG_C         = "{BG}";

    // ════════════════════════════════════════════════════════════════
    // GEX PROFILE INTERACTIVE BAR CHART (Plotly.react)
    // ════════════════════════════════════════════════════════════════
    const gexSlider  = document.getElementById('gexSlider');
    const gexTimeVal = document.getElementById('gexTimeVal');
    const gexSpotVal = document.getElementById('gexSpotVal');
    const gexFlipVal = document.getElementById('gexFlipVal');
    const gexNetVal  = document.getElementById('gexNetVal');

    gexSlider.max   = gexSnapshots.length - 1;
    gexSlider.value = gexSnapshots.length - 1;

    function buildGexBarData(snap) {{
        const strikeLabels = strikesArr.map(s => Number(s));
        const hasData = snap.gex_ce.some(v => v !== 0) || snap.gex_pe.some(v => v !== 0);

        const trCe = {{
            x: strikeLabels,
            y: snap.gex_ce,
            type: 'bar',
            name: 'CE GEX',
            marker: {{ color: GEX_GREEN + '99' }},
            hovertemplate: 'Strike %{{x}}<br>CE GEX: %{{y:.3f}} Cr<extra></extra>'
        }};
        const trPe = {{
            x: strikeLabels,
            y: snap.gex_pe,
            type: 'bar',
            name: 'PE GEX',
            marker: {{ color: GEX_RED + '99' }},
            hovertemplate: 'Strike %{{x}}<br>PE GEX: %{{y:.3f}} Cr<extra></extra>'
        }};
        const trNet = {{
            x: strikeLabels,
            y: snap.gex_net,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Net GEX',
            line: {{ color: TEXT_C, width: 2 }},
            marker: {{ size: 5, color: snap.gex_net.map(v => v >= 0 ? GEX_GREEN : GEX_RED) }},
            hovertemplate: 'Strike %{{x}}<br>Net GEX: %{{y:.3f}} Cr<extra></extra>'
        }};

        // Spot line
        const spotShapes = [];
        if (snap.spot) {{
            spotShapes.push({{
                type: 'line', x0: snap.spot, x1: snap.spot, y0: 0, y1: 1,
                xref: 'x', yref: 'paper',
                line: {{ color: BLUE_C, width: 2, dash: 'dot' }}
            }});
        }}
        // Flip line
        if (snap.flip_level) {{
            spotShapes.push({{
                type: 'line', x0: snap.flip_level, x1: snap.flip_level, y0: 0, y1: 1,
                xref: 'x', yref: 'paper',
                line: {{ color: GEX_FLIP_C, width: 2, dash: 'dash' }}
            }});
        }}
        // Zero GEX horizontal line
        spotShapes.push({{
            type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
            xref: 'paper', yref: 'y',
            line: {{ color: MUTED_C, width: 1, dash: 'dot' }}
        }});

        const annotations = [];
        if (snap.spot) {{
            annotations.push({{
                x: snap.spot, y: 1.02, xref: 'x', yref: 'paper',
                text: 'SPOT', showarrow: false,
                font: {{ color: BLUE_C, size: 10 }}, xanchor: 'center'
            }});
        }}
        if (snap.flip_level) {{
            annotations.push({{
                x: snap.flip_level, y: 0.98, xref: 'x', yref: 'paper',
                text: 'FLIP', showarrow: false,
                font: {{ color: GEX_FLIP_C, size: 10 }}, xanchor: 'center'
            }});
        }}
        if (!hasData) {{
            annotations.push({{
                x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                text: '⚠ OI data is zero — verify Fyers returns OI column for F&O history',
                showarrow: false, font: {{ color: MUTED_C, size: 12 }}
            }});
        }}

        const layout = {{
            template: 'plotly_dark',
            paper_bgcolor: CARD_C, plot_bgcolor: CARD_C,
            height: 420,
            barmode: 'relative',
            margin: {{ l: 10, r: 10, t: 30, b: 40 }},
            xaxis: {{ title: 'Strike', color: TEXT_C, tickfont: {{ size: 10 }} }},
            yaxis: {{ title: 'GEX (₹ Cr)', color: TEXT_C, zeroline: false }},
            legend: {{ orientation: 'h', y: -0.15 }},
            shapes: spotShapes,
            annotations: annotations
        }};

        return {{ traces: [trCe, trPe, trNet], layout }};
    }}

    let gexBarInitialised = false;

    function updateGexBar() {{
        const idx  = parseInt(gexSlider.value);
        const snap = gexSnapshots[idx];
        if (!snap) return;

        gexTimeVal.textContent = snap.time;
        gexSpotVal.textContent = snap.spot ? snap.spot.toFixed(0) : '--';
        gexFlipVal.textContent = snap.flip_level ? snap.flip_level.toFixed(1) : 'N/A';
        const netGex = snap.total_net_gex;
        gexNetVal.textContent  = netGex.toFixed(2) + ' Cr';
        gexNetVal.style.color  = netGex >= 0 ? GEX_GREEN : GEX_RED;

        const {{ traces, layout }} = buildGexBarData(snap);
        if (!gexBarInitialised) {{
            Plotly.newPlot('gexBarDiv', traces, layout, {{responsive: true}});
            gexBarInitialised = true;
        }} else {{
            Plotly.react('gexBarDiv', traces, layout);
        }}
    }}

    gexSlider.addEventListener('input', updateGexBar);

    // ════════════════════════════════════════════════════════════════
    // TABS
    // ════════════════════════════════════════════════════════════════
    const metricTabs = ['iv', 'theta', 'vega', 'gamma', 'theta15'];

    function showTab(name) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        document.getElementById('btn-' + name).classList.add('active');
        document.getElementById('strikeToggleBar').style.display =
            metricTabs.includes(name) ? 'flex' : 'none';
        window.dispatchEvent(new Event('resize'));

        // Initialise GEX bar chart when tab first opens
        if (name === 'gex' && !gexBarInitialised) {{
            updateGexBar();
        }}
    }}
    document.getElementById('strikeToggleBar').style.display = 'none';

    // ════════════════════════════════════════════════════════════════
    // STRIKE TOGGLE (IV / Theta / Vega / Gamma / Theta15 tabs)
    // ════════════════════════════════════════════════════════════════
    const metricChartIds = ['ivChart','thetaChart','vegaChart','gammaChart','theta15Chart'];
    function toggleStrike(idx, checked) {{
        metricChartIds.forEach(id => {{
            const el = document.getElementById(id);
            if (el && el.data) Plotly.restyle(id, {{ visible: checked }}, [idx]);
        }});
    }}

    // ════════════════════════════════════════════════════════════════
    // MOMENTUM TAB
    // ════════════════════════════════════════════════════════════════
    const timeSlider   = document.getElementById('timeSlider');
    const timeVal      = document.getElementById('timeVal');
    const timeDisplay  = document.getElementById('timeDisplay');
    const tbody        = document.getElementById('speedTableBody');
    timeSlider.max     = allTimes.length - 1;
    timeSlider.value   = allTimes.length - 1;

    function calcStats(strike, tIdx, durationCandles) {{
        const series  = speedData[strike];
        const fromIdx = Math.max(0, tIdx - durationCandles);
        const fromPt  = series[fromIdx];
        const toPt    = series[tIdx];
        if (!fromPt || !toPt) return null;
        const mins  = (tIdx - fromIdx) * 5;
        const delta = toPt.price - fromPt.price;
        const speed = mins > 0 ? (delta / mins) : 0;
        const slice = series.slice(fromIdx, tIdx + 1).map(d => d.price);
        let net = Math.abs(slice[slice.length - 1] - slice[0]);
        let total = 0;
        for (let i = 1; i < slice.length; i++) total += Math.abs(slice[i] - slice[i-1]);
        let smooth = total > 0 ? (net / total) * 100 : 100.0;
        const dir  = delta > 2 ? 'UP' : delta < -2 ? 'DOWN' : 'FLAT';
        const accentC = "{ACCENT}", redC = "{RED}", textC = "{TEXT}", blueC = "{BLUE}";
        const badge = dir === 'UP'   ? '<span class="badge badge-up">▲ UP</span>' :
                      dir === 'DOWN' ? '<span class="badge badge-down">▼ DOWN</span>' :
                                       '<span class="badge badge-flat">— FLAT</span>';
        return {{
            dur: mins + 'm',
            speed: (speed >= 0 ? '+' : '') + speed.toFixed(2),
            smooth: smooth.toFixed(1) + '%',
            badge: badge,
            speedColor: Math.abs(speed) > 5 ? (delta > 0 ? accentC : redC) : textC,
            smoothColor: smooth > 70 ? accentC : smooth > 40 ? blueC : redC
        }};
    }}

    function updateTable() {{
        const tIdx = parseInt(timeSlider.value);
        timeVal.textContent     = allTimes[tIdx];
        timeDisplay.textContent = `Analysis Window End: ${{allTimes[tIdx]}}`;
        const accentC = "{ACCENT}", redC = "{RED}", borderC = "{BORDER}";
        const rows = strikesArr.map((strike, idx) => {{
            const s30 = calcStats(strike, tIdx, 6);
            const s60 = calcStats(strike, tIdx, 12);
            if (!s30 || !s60) return '';
            const atmMark = strike === ATM ? ` <small style="color:${{accentC}}">ATM</small>` : '';
            return `<tr>
                <td style="font-weight:bold;border-right:1px solid ${{borderC}}44;">
                    <span class="strike-dot" style="background:${{colors[idx % colors.length]}};"></span>${{strike}}${{atmMark}}
                </td>
                <td style="color:{MUTED}">${{s30.dur}}</td>
                <td style="color:${{s30.speedColor}};font-weight:bold;">${{s30.speed}}</td>
                <td style="color:${{s30.smoothColor}};">${{s30.smooth}}</td>
                <td style="border-right:1px solid ${{borderC}}44;">${{s30.badge}}</td>
                <td style="color:{MUTED}">${{s60.dur}}</td>
                <td style="color:${{s60.speedColor}};font-weight:bold;">${{s60.speed}}</td>
                <td style="color:${{s60.smoothColor}};">${{s60.smooth}}</td>
                <td>${{s60.badge}}</td>
            </tr>`;
        }});
        tbody.innerHTML = rows.join('');
    }}
    timeSlider.addEventListener('input', updateTable);
    updateTable();

    setTimeout(function() {{ location.reload(); }}, 300000);
    </script>
</body>
</html>"""

    docs_path   = "docs/index.html"
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    timestamp   = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(OUTPUT_DIR, f"dashboard_{timestamp}.html")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return docs_path, backup_path


# --- DATA FETCHING ---

def fetch_candles(fyers, symbol, date):
    """
    Fetch 5-min OHLCV candles.  Fyers F&O history returns 7 columns (the 7th
    is Open Interest); equity/index returns only 6.  We capture OI when
    present so GEX can be computed without a separate round-trip.
    """
    data = {
        "symbol": symbol, "resolution": "5", "date_format": "1",
        "range_from": date, "range_to": date, "cont_flag": "1"
    }
    try:
        resp = fyers.history(data=data)
        if resp.get("s") == "ok" and resp.get("candles"):
            candles = resp["candles"]
            # Detect whether OI column is present
            if candles and len(candles[0]) >= 7:
                cols = ["epoch", "open", "high", "low", "close", "volume", "oi"]
            else:
                cols = ["epoch", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame(candles, columns=cols)
            if "oi" not in df.columns:
                df["oi"] = 0          # equity/index: no OI
            df["time"] = (pd.to_datetime(df["epoch"], unit="s", utc=True)
                          .dt.tz_convert("Asia/Kolkata")
                          .dt.tz_localize(None))
            return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
    return pd.DataFrame()


def compute_atm(fyers, spot_df, step=STRIKE_STEP, fallback=FALLBACK_ATM, use_live_quote=True, reference="last"):
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

    if spot_price is None or spot_price != spot_price:
        logger.warning(f"Could not determine spot price - falling back to hardcoded ATM {fallback}")
        return fallback

    atm = int(round(spot_price / step) * step)
    logger.info(f"Spot price: {spot_price} -> ATM strike: {atm}")
    return atm


def fetch_and_enrich_strike(fyers, strike, spot_df):
    """
    Fetch CE+PE candles for a strike and enrich with straddle price, VWAP,
    EMA9, IV, Greeks and trailing Theta.  Now also carries oi_ce / oi_pe for
    GEX computation downstream.
    """
    ce_symbol = f"NSE:NIFTY{EXPIRY}{strike}CE"
    pe_symbol = f"NSE:NIFTY{EXPIRY}{strike}PE"

    ce_df = fetch_candles(fyers, ce_symbol, TARGET_DATE)
    pe_df = fetch_candles(fyers, pe_symbol, TARGET_DATE)

    if ce_df.empty or pe_df.empty:
        return None

    # ── MODIFIED: include OI columns in merge ────────────────────────────────
    merged = pd.merge(
        ce_df[["time", "close", "volume", "oi"]],
        pe_df[["time", "close", "volume", "oi"]],
        on="time"
    )
    merged.rename(columns={"oi_x": "oi_ce", "oi_y": "oi_pe"}, inplace=True)
    # ─────────────────────────────────────────────────────────────────────────

    merged["straddle"] = merged["close_x"] + merged["close_y"]
    merged["v"]        = merged["volume_x"] + merged["volume_y"]
    merged["vwap"]     = (merged["straddle"] * merged["v"]).cumsum() / merged["v"].cumsum()
    merged["ema9"]     = merged["straddle"].ewm(span=9, adjust=False).mean()

    if not spot_df.empty:
        merged = pd.merge(merged, spot_df, on="time", how="left")
        merged["spot"] = merged["spot"].ffill()
        greeks = merged.apply(lambda row: compute_greeks_row(row, strike), axis=1)
        merged = pd.concat([merged, greeks], axis=1)
    else:
        merged["iv_ce"] = merged["iv_pe"] = merged["iv_pct"] = float("nan")
        merged["gamma_ce"]    = merged["gamma_pe"]    = 0.0
        merged["gamma_total"] = merged["vega_total"]  = merged["theta_total"] = 0.0

    candles_per_window  = THETA_WINDOW_MINUTES // CANDLE_INTERVAL_MINUTES
    theta_per_candle    = merged["theta_total"] * (CANDLE_INTERVAL_MINUTES / 1440.0)
    merged["theta_15min"] = theta_per_candle.rolling(window=candles_per_window, min_periods=1).sum()

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

        spot_df = fetch_candles(fyers, SPOT_SYMBOL, TARGET_DATE)
        if spot_df.empty:
            logger.warning("Could not fetch spot index data — IV/Greeks/GEX will be unavailable.")
        else:
            spot_df = spot_df[["time", "close"]].rename(columns={"close": "spot"})

        atm = compute_atm(fyers, spot_df)

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
            docs_path, backup_path = build_dashboard_html(results, atm, rankings)
            logger.info(f"✓ Dashboard generated: {docs_path}")

            send_telegram_document(
                docs_path,
                caption=f"📊 <b>Straddle Dashboard</b> — {TARGET_DATE}\nOpen in browser to view interactive charts."
            )
            logger.info(f"✓ Successfully processed {successful_fetches}/{len(OFFSETS)} strikes")
        else:
            logger.error("No data successfully fetched for any strike")
            send_telegram_message(
                f"❌ <b>Straddle Analyser Failed</b>\nDate: {TARGET_DATE}\nNo data fetched for any strike."
            )

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        send_telegram_message(
            f"❌ <b>Straddle Analyser Error</b>\nDate: {TARGET_DATE}\n<code>{str(e)}</code>"
        )


if __name__ == "__main__":
    main()
