from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

from indicators import vwap as calc_vwap, atr as calc_atr, rolling_swing_lows, rolling_swing_highs
from sessions import classify_session


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/OFF


PRESETS: Dict[str, Dict[str, float]] = {
    # Fast scalp: more frequent, faster trigger, slightly lower confirmation thresholds
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,          # 0/1
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
        "min_rsi14_long": 0,          # allow any, but we still cap overheated in scoring
        "max_rsi14_short": 100,
    },
    # Cleaner signals: fewer alerts, higher win-rate bias
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
        "min_rsi14_long": 0,
        "max_rsi14_short": 100,
    },
}


def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    lookback_bars: int = 160,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF")

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = ohlcv.copy().tail(int(lookback_bars)).copy()
    df["vwap"] = calc_vwap(df)
    df["atr14"] = calc_atr(df, 14)

    # Align indicators
    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap = df["vwap"]

    last_ts = df.index[-1]
    session = classify_session(last_ts)

    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
    )
    if not allowed:
        return SignalResult(symbol, "NEUTRAL", 0, f"Filtered by time-of-day ({session})", None, None, None, None, float(close.iloc[-1]), last_ts, session)

    last_price = float(close.iloc[-1])

    # Events
    was_below_vwap = (close.shift(3) < vwap.shift(3)).iloc[-1] or (close.shift(5) < vwap.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap.shift(3)).iloc[-1] or (close.shift(5) > vwap.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap.shift(1).iloc[-1])

    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Stops based on recent swing
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(5)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(5)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Score components (simple, interpretable)
    long_points = 0
    long_reasons = []
    if was_below_vwap and reclaim_vwap:
        long_points += 35; long_reasons.append("VWAP reclaim")
    if rsi_snap and rsi14 < 60:
        long_points += 20; long_reasons.append("RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        long_points += 20; long_reasons.append("MACD hist turning up")
    if vol_ok:
        long_points += 15; long_reasons.append("Volume confirmation")
    # micro higher-low
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        long_points += 10; long_reasons.append("Higher-low micro structure")

    short_points = 0
    short_reasons = []
    if was_above_vwap and reject_vwap:
        short_points += 35; short_reasons.append("VWAP rejection")
    if rsi_downshift and rsi14 > 40:
        short_points += 20; short_reasons.append("RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        short_points += 20; short_reasons.append("MACD hist turning down")
    if vol_ok:
        short_points += 15; short_reasons.append("Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        short_points += 10; short_reasons.append("Lower-high micro structure")

    # "Cleaner" hard requirements to reduce noise
    if int(cfg["require_vwap_event"]) == 1:
        if not ((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap)):
            # no VWAP event => usually chop
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session)
    if int(cfg["require_rsi_event"]) == 1:
        if not (rsi_snap or rsi_downshift):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session)
    if int(cfg["require_macd_turn"]) == 1:
        if not (macd_turn_up or macd_turn_down):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session)
    if int(cfg["require_volume"]) == 1 and not vol_ok:
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No volume confirmation", None, None, None, None, last_price, last_ts, session)

    min_score = int(cfg["min_actionable_score"])

    if long_points >= min_score and long_points > short_points:
        entry = last_price
        stop = float(min(recent_swing_low, last_price - max(df["atr14"].iloc[-1], 0.0) * 0.8))
        risk = max(entry - stop, 0.01)
        return SignalResult(
            symbol, "LONG", min(100, int(long_points)),
            ", ".join(long_reasons[:4]),
            entry, stop, entry + risk, entry + 2 * risk,
            last_price, last_ts, session
        )

    if short_points >= min_score and short_points > long_points:
        entry = last_price
        stop = float(max(recent_swing_high, last_price + max(df["atr14"].iloc[-1], 0.0) * 0.8))
        risk = max(stop - entry, 0.01)
        return SignalResult(
            symbol, "SHORT", min(100, int(short_points)),
            ", ".join(short_reasons[:4]),
            entry, stop, entry - risk, entry - 2 * risk,
            last_price, last_ts, session
        )

    reason = f"LongScore={long_points} ({', '.join(long_reasons)}); ShortScore={short_points} ({', '.join(short_reasons)})"
    return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), reason, None, None, None, None, last_price, last_ts, session)
