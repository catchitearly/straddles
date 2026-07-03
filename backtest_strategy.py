"""
backtest_strategy.py

Backtests a short-straddle strategy:

  Every 5 minutes, from 09:45 onward:
    - Rank all tracked strikes by 15-min theta decay (most negative = most decay)
      and take the top 2 ("top2_theta").
    - Rank all tracked strikes by Vega and take the top 2 ("top2_vega").
    - If a strike appears in BOTH top-2 lists (common_entry) AND its straddle
      price is below its own VWAP, SELL that straddle (only one open position
      at a time; if several strikes qualify in the same candle, the one with
      the biggest (VWAP - price) gap is chosen).

  While a position is open, every 5 minutes:
    - Exit if the strike is no longer in the "top 3" set (top-3 theta decay
      INTERSECT top-3 vega — see EXIT_METHOD below to use a different rule).
    - Exit if loss reaches SL_POINTS (hard stop-loss).
    - Exit if peak profit has reached TSL_ACTIVATION_PROFIT: from that point
      on, a trailing stop is active. Its distance from the peak starts at
      TSL_INITIAL_DISTANCE and grows by TSL_INCREMENT_PER_POINT for every
      point of profit beyond TSL_ACTIVATION_PROFIT (this is a literal
      implementation of "tsl is 2 pts at 5 pts profit, +1 pt tsl per +1 pt
      profit after that" - with the stated numbers this locks in a constant
      floor of 3 points profit once first touched; see the assumptions note
      in chat if you intended a different trailing behaviour).
    - Forced close at FORCE_EXIT_TIME (end-of-day square off).
    - A same-candle re-entry is allowed immediately after an exit if the
      entry criteria are met again (except on the EOD forced close).

Run:
    python backtest_strategy.py --date 2026-07-01 --expiry-code 26707 --expiry-date 2026-07-07

Or via environment variables (as set by the GitHub Actions workflow):
    TARGET_DATE, OPTION_EXPIRY_CODE, OPTION_EXPIRY_DATE
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CLI / ENV ARGS (must be applied BEFORE importing straddle_analyser, since
#     that module reads TARGET_DATE / OPTION_EXPIRY_CODE / OPTION_EXPIRY_DATE
#     from the environment at import time) ---

def parse_args():
    p = argparse.ArgumentParser(description="Backtest the theta-decay + vega straddle strategy")
    p.add_argument("--date", default=os.getenv("TARGET_DATE"), help="Backtest date, YYYY-MM-DD")
    p.add_argument("--expiry-code", default=os.getenv("OPTION_EXPIRY_CODE"), help="Fyers symbol expiry code, e.g. 26707")
    p.add_argument("--expiry-date", default=os.getenv("OPTION_EXPIRY_DATE"), help="Actual calendar expiry date, YYYY-MM-DD")
    return p.parse_args()

_args = parse_args()
if not _args.date:
    logger.error("No --date / TARGET_DATE supplied. Aborting.")
    sys.exit(1)
if not _args.expiry_code or not _args.expiry_date:
    logger.error("No --expiry-code/--expiry-date (or OPTION_EXPIRY_CODE/OPTION_EXPIRY_DATE) supplied. Aborting.")
    sys.exit(1)

os.environ["TARGET_DATE"] = _args.date
os.environ["OPTION_EXPIRY_CODE"] = _args.expiry_code
os.environ["OPTION_EXPIRY_DATE"] = _args.expiry_date

import straddle_analyser as sa  # noqa: E402  (must import after env vars are set)

# --- STRATEGY CONFIG ---
ENTRY_START_TIME = dtime(9, 45)
FORCE_EXIT_TIME = dtime(15, 20)
ENTRY_TOP_N = 2
EXIT_TOP_N = 3
EXIT_METHOD = "intersection"  # "intersection" = top3_theta & top3_vega ; "combined_rank" = lowest combined rank sum

SL_POINTS = 10.0
TSL_ACTIVATION_PROFIT = 5.0
TSL_INITIAL_DISTANCE = 2.0
TSL_INCREMENT_PER_POINT = 1.0

LOT_SIZE = 75  # Update to the current NIFTY lot size

OUTPUT_DIR = "backtest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG = "#0b0f1a"
CARD = "#111b27"
BORDER = "#1e2d40"
TEXT = "#e2e8f0"
MUTED = "#64748b"
ACCENT = "#00e5b0"
RED = "#ff4560"


# --- DATA LOADING ---

def load_day_data(fyers):
    """Fetch spot + all tracked strikes for the backtest day, enrich with Greeks."""
    spot_df = sa.fetch_candles(fyers, sa.SPOT_SYMBOL, sa.TARGET_DATE)
    if spot_df.empty:
        raise RuntimeError(f"No spot data available for {sa.TARGET_DATE} - cannot backtest.")
    spot_df = spot_df[["time", "close"]].rename(columns={"close": "spot"})

    # Historical ATM: anchor off the FIRST candle of the day (i.e. what a live
    # run would have seen that morning), never a live quote.
    atm = sa.compute_atm(fyers, spot_df, use_live_quote=False, reference="first")

    strike_frames = {}
    for offset in sa.OFFSETS:
        strike = atm + offset
        merged = sa.fetch_and_enrich_strike(fyers, strike, spot_df)
        if merged is not None:
            strike_frames[strike] = merged.set_index("time")
            logger.info(f"✓ Loaded strike {strike}")
        else:
            logger.warning(f"✗ No data for strike {strike}")

    if not strike_frames:
        raise RuntimeError("No strike data could be loaded for the backtest day.")

    return atm, strike_frames


def align_frames(strike_frames):
    """Reindex every strike's dataframe onto the same time axis (union, forward-filled)."""
    all_times = sorted(set().union(*[df.index for df in strike_frames.values()]))
    aligned = {s: df.reindex(all_times).ffill() for s, df in strike_frames.items()}
    return all_times, aligned


# --- STRATEGY ENGINE ---

def rank_and_select(aligned, strikes, t):
    theta_vals = pd.Series({s: aligned[s].loc[t, "theta_15min"] for s in strikes})
    vega_vals = pd.Series({s: aligned[s].loc[t, "vega_total"] for s in strikes})

    if theta_vals.isna().any() or vega_vals.isna().any():
        return None  # incomplete data this candle - skip

    top_n_theta_entry = theta_vals.nsmallest(ENTRY_TOP_N).index.tolist()   # most negative = most decay
    top_n_vega_entry = vega_vals.nlargest(ENTRY_TOP_N).index.tolist()
    common_entry = sorted(set(top_n_theta_entry) & set(top_n_vega_entry))

    if EXIT_METHOD == "combined_rank":
        combined = theta_vals.rank(ascending=True) + vega_vals.rank(ascending=False)
        exit_universe = set(combined.nsmallest(EXIT_TOP_N).index.tolist())
    else:  # intersection (default)
        top_n_theta_exit = theta_vals.nsmallest(EXIT_TOP_N).index.tolist()
        top_n_vega_exit = vega_vals.nlargest(EXIT_TOP_N).index.tolist()
        exit_universe = set(top_n_theta_exit) & set(top_n_vega_exit)

    return common_entry, exit_universe


def try_enter(aligned, common_entry, t):
    candidates = []
    for s in common_entry:
        price = aligned[s].loc[t, "straddle"]
        vwap = aligned[s].loc[t, "vwap"]
        if price != price or vwap != vwap:
            continue
        if price < vwap:
            candidates.append((s, vwap - price, price))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])  # biggest below-VWAP gap first
    strike, gap, price = candidates[0]
    return {"strike": strike, "entry_time": t, "entry_price": price, "peak_profit": 0.0}


def run_backtest(atm, strike_frames):
    all_times, aligned = align_frames(strike_frames)
    strikes = sorted(aligned.keys())

    trades = []
    position = None

    for t in all_times:
        candle_time = t.time()
        if candle_time < ENTRY_START_TIME:
            continue

        forced_eod = candle_time >= FORCE_EXIT_TIME
        selection = rank_and_select(aligned, strikes, t)

        # --- manage an open position ---
        if position is not None:
            strike = position["strike"]
            current_price = aligned[strike].loc[t, "straddle"]
            if current_price == current_price:  # not NaN
                profit = position["entry_price"] - current_price
                position["peak_profit"] = max(position["peak_profit"], profit)

                exit_reason = None
                if profit <= -SL_POINTS:
                    exit_reason = "SL"
                elif position["peak_profit"] >= TSL_ACTIVATION_PROFIT:
                    distance = TSL_INITIAL_DISTANCE + max(0.0, position["peak_profit"] - TSL_ACTIVATION_PROFIT) * TSL_INCREMENT_PER_POINT
                    floor = position["peak_profit"] - distance
                    if profit <= floor:
                        exit_reason = "TSL"

                if exit_reason is None and selection is not None:
                    _, exit_universe = selection
                    if strike not in exit_universe:
                        exit_reason = "RANK_EXIT"

                if exit_reason is None and forced_eod:
                    exit_reason = "EOD"

                if exit_reason:
                    trades.append({
                        "strike": strike,
                        "entry_time": position["entry_time"],
                        "entry_price": round(position["entry_price"], 2),
                        "exit_time": t,
                        "exit_price": round(current_price, 2),
                        "points": round(profit, 2),
                        "rupees": round(profit * LOT_SIZE, 2),
                        "exit_reason": exit_reason,
                    })
                    logger.info(f"EXIT  {strike} @ {t} price={current_price:.2f} points={profit:.2f} reason={exit_reason}")
                    position = None

        # --- look for a new entry (same candle re-entry allowed, except on forced EOD) ---
        if position is None and not forced_eod and selection is not None:
            common_entry, _ = selection
            new_pos = try_enter(aligned, common_entry, t)
            if new_pos:
                position = new_pos
                logger.info(f"ENTER {position['strike']} @ {t} price={position['entry_price']:.2f}")

    # Force-close anything still open at the very end of the loaded data
    if position is not None:
        strike = position["strike"]
        last_t = all_times[-1]
        current_price = aligned[strike].loc[last_t, "straddle"]
        profit = position["entry_price"] - current_price
        trades.append({
            "strike": strike,
            "entry_time": position["entry_time"],
            "entry_price": round(position["entry_price"], 2),
            "exit_time": last_t,
            "exit_price": round(current_price, 2),
            "points": round(profit, 2),
            "rupees": round(profit * LOT_SIZE, 2),
            "exit_reason": "EOD_FORCED",
        })
        logger.info(f"EXIT  {strike} @ {last_t} price={current_price:.2f} points={profit:.2f} reason=EOD_FORCED")

    return pd.DataFrame(trades)


# --- REPORTING ---

def summarize(trades_df):
    if trades_df.empty:
        return {"total_trades": 0}
    wins = trades_df[trades_df["points"] > 0]
    losses = trades_df[trades_df["points"] <= 0]
    return {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / len(trades_df), 1),
        "total_points": round(trades_df["points"].sum(), 2),
        "total_rupees": round(trades_df["rupees"].sum(), 2),
        "avg_points_per_trade": round(trades_df["points"].mean(), 2),
        "best_trade_points": round(trades_df["points"].max(), 2),
        "worst_trade_points": round(trades_df["points"].min(), 2),
    }


def build_report_html(trades_df, summary, atm):
    equity = trades_df["points"].cumsum() if not trades_df.empty else pd.Series(dtype=float)
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(equity) + 1)), y=equity,
        mode="lines+markers", line=dict(color=ACCENT, width=2)
    ))
    fig.update_layout(
        title="Cumulative Points (per trade)", template="plotly_dark",
        paper_bgcolor=CARD, plot_bgcolor=CARD, height=400,
        xaxis_title="Trade #", yaxis_title="Cumulative Points",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    rows_html = "".join([
        f"""<tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:8px;">{r.strike}</td>
            <td style="padding:8px;">{r.entry_time}</td>
            <td style="padding:8px;">{r.entry_price}</td>
            <td style="padding:8px;">{r.exit_time}</td>
            <td style="padding:8px;">{r.exit_price}</td>
            <td style="padding:8px;color:{ACCENT if r.points>0 else RED};"><b>{r.points}</b></td>
            <td style="padding:8px;">{r.rupees}</td>
            <td style="padding:8px;">{r.exit_reason}</td>
        </tr>""" for r in trades_df.itertuples()
    ]) if not trades_df.empty else "<tr><td style='padding:12px;'>No trades taken.</td></tr>"

    summary_html = "".join([f"<div class='stat'><div class='stat-label'>{k}</div><div class='stat-value'>{v}</div></div>" for k, v in summary.items()])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Backtest Report - {sa.TARGET_DATE}</title>
<style>
body {{ background:{BG}; color:{TEXT}; font-family:'Segoe UI',sans-serif; margin:0; padding:20px; }}
.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:20px; margin-bottom:20px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; text-align:left; }}
th {{ color:{MUTED}; text-transform:uppercase; font-size:9px; padding:8px; border-bottom:1px solid {BORDER}; }}
.stats {{ display:flex; flex-wrap:wrap; gap:16px; }}
.stat {{ background:{BG}; border:1px solid {BORDER}; border-radius:6px; padding:10px 16px; min-width:120px; }}
.stat-label {{ font-size:10px; color:{MUTED}; text-transform:uppercase; }}
.stat-value {{ font-size:18px; color:{ACCENT}; font-weight:bold; }}
h1 {{ font-size:20px; }}
h2 {{ font-size:14px; color:{ACCENT}; letter-spacing:1px; }}
</style></head>
<body>
<h1>▣ BACKTEST REPORT — {sa.TARGET_DATE} (Expiry {sa.EXPIRY_DATE}, ATM {atm})</h1>
<div class="card">
    <h2>SUMMARY</h2>
    <div class="stats">{summary_html}</div>
</div>
<div class="card">
    <h2>EQUITY CURVE</h2>
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
</div>
<div class="card">
    <h2>TRADE LOG</h2>
    <table>
        <thead><tr><th>Strike</th><th>Entry Time</th><th>Entry Px</th><th>Exit Time</th><th>Exit Px</th><th>Points</th><th>Rupees</th><th>Reason</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
</body></html>"""

    path = os.path.join(OUTPUT_DIR, f"report_{sa.TARGET_DATE}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    if not sa.CLIENT_ID or not sa.TOKEN:
        logger.error("API Credentials missing. Please set CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.")
        return

    fyers = sa.fyersModel.FyersModel(client_id=sa.CLIENT_ID, token=sa.TOKEN, log_path="")
    profile = fyers.get_profile()
    if profile.get("s") != "ok":
        logger.error(f"Authentication failed: {profile}")
        return

    logger.info(f"Backtesting {sa.TARGET_DATE} | expiry code {sa.EXPIRY} | expiry date {sa.EXPIRY_DATE}")

    atm, strike_frames = load_day_data(fyers)
    logger.info(f"ATM for the day: {atm} | strikes loaded: {sorted(strike_frames.keys())}")

    trades_df = run_backtest(atm, strike_frames)
    summary = summarize(trades_df)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    csv_path = os.path.join(OUTPUT_DIR, f"trades_{sa.TARGET_DATE}.csv")
    trades_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Trade log saved: {csv_path}")

    report_path = build_report_html(trades_df, summary, atm)
    logger.info(f"✓ Report saved: {report_path}")

    if sa.TELEGRAM_BOT_TOKEN and sa.TELEGRAM_CHAT_ID:
        lines = [
            f"<b>📊 BACKTEST — {sa.TARGET_DATE}</b>",
            f"<code>Expiry: {sa.EXPIRY_DATE} | ATM: {atm}</code>",
            "",
        ]
        if summary.get("total_trades", 0) == 0:
            lines.append("No trades were taken.")
        else:
            lines += [
                f"Trades: {summary['total_trades']} | Win rate: {summary['win_rate_pct']}%",
                f"Total points: {summary['total_points']} | ₹{summary['total_rupees']}",
                f"Best: {summary['best_trade_points']} pts | Worst: {summary['worst_trade_points']} pts",
            ]
        sa.send_telegram_message("\n".join(lines))
        sa.send_telegram_document(report_path, caption=f"📎 Backtest report — {sa.TARGET_DATE}")


if __name__ == "__main__":
    main()
