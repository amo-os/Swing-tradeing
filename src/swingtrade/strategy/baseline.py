"""Transparent baseline swing-trading strategy.

Strategy Rules:
---------------
1. Universe: Strictly restricted to Phase A approved constituents of NIFTY 100.
2. Regime Filter (Macro):
   - Daily Close > EMA(200): Only trade when stock is in a macro uptrend regime.
3. Trend & Momentum Setup (Intermediate):
   - Daily Close > EMA(50): Stock is above its intermediate trend baseline.
   - RSI(14) between 40 and 65: Momentum is healthy and not overextended.
   - MACD line > MACD Signal line: Bullish momentum confirmation.
4. Liquidity & Volatility:
   - Volume > 0.8 * Average 20-day Volume.
   - ATR(14) is positive and available.
5. Risk & Order Parameters:
   - Stop Loss: Close - 2.0 * ATR(14).
   - Profit Target: Close + 3.0 * ATR(14) (1.5:1 reward-to-risk ratio).
6. Discretionary Technical Exits:
   - Trend breakdown: Close < EMA(50) triggers exit signal.
   - Overbought blow-off: RSI(14) > 75 triggers exit signal.
   - Time stop: Position held longer than max_holding_bars (default 20 bars).

Limitations:
------------
- Long-only strategy; does not hedge during broad market corrections.
- Sensitive to choppy, range-bound market regimes where whipsaws occur around EMA(50).
- Uses unadjusted raw prices for approved stocks; corporate action stocks are barred.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swingtrade.strategy.base import Signal, SignalDirection, Strategy
from swingtrade.strategy.indicators import calculate_indicators


@dataclass
class BaselineSwingStrategy(Strategy):
    """Configurable trend-following momentum swing trading strategy."""

    ema_trend: int = 50
    ema_regime: int = 200
    rsi_lower: float = 40.0
    rsi_upper: float = 65.0
    atr_multiplier_stop: float = 2.0
    atr_multiplier_target: float = 3.0
    max_holding_bars: int = 20
    volume_factor: float = 0.8

    @property
    def name(self) -> str:
        return "BaselineTrendPullbackSwing"

    def generate_signals(
        self,
        current_date: pd.Timestamp,
        historical_data: dict[str, pd.DataFrame],
        open_positions: dict[str, object],
    ) -> list[Signal]:
        """Generate signals point-in-time at current_date session close."""
        signals: list[Signal] = []

        for symbol, df in historical_data.items():
            if df.empty or len(df) < self.ema_regime:
                continue

            # Indicators are calculated strictly on available slice <= current_date
            # Optimization: if indicators already calculated in df, reuse; else compute
            req_cols = {"ema_50", "ema_200", "rsi_14", "macd", "macd_signal", "atr_14"}
            if req_cols.issubset(df.columns):
                ind_df = df
            else:
                try:
                    ind_df = calculate_indicators(df)
                except ValueError:
                    continue

            if ind_df.empty:
                continue

            last_bar = ind_df.iloc[-1]
            if last_bar["timestamp"] != current_date:
                # Symbol didn't trade on this calendar date
                continue

            close = float(last_bar["close"])
            ema_50 = float(last_bar["ema_50"])
            ema_200 = float(last_bar["ema_200"])
            rsi = float(last_bar["rsi_14"])
            macd = float(last_bar["macd"])
            macd_sig = float(last_bar["macd_signal"])
            atr = float(last_bar["atr_14"])
            vol = float(last_bar["volume"])
            avg_vol = (
                float(last_bar["average_volume_20"])
                if pd.notna(last_bar["average_volume_20"])
                else 0.0
            )

            # Check if symbol is currently held
            if symbol in open_positions:
                pos = open_positions[symbol]
                days_held = getattr(pos, "days_held", 0)

                # Exit Condition 1: Trend break below EMA(50)
                if close < ema_50:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=current_date,
                            direction=SignalDirection.SELL,
                            reason=f"Close ({close:.2f}) fell below EMA-50 ({ema_50:.2f})",
                        )
                    )
                # Exit Condition 2: Overbought blow-off
                elif rsi > 75.0:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=current_date,
                            direction=SignalDirection.SELL,
                            reason=f"RSI ({rsi:.1f}) reached overbought threshold (>75)",
                        )
                    )
                # Exit Condition 3: Time stop
                elif days_held >= self.max_holding_bars:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=current_date,
                            direction=SignalDirection.SELL,
                            reason=(
                                f"Max holding period reached ({days_held} sessions)"
                            ),
                        )
                    )
                continue

            # Entry Logic (when NOT in open position)
            # 1. Macro Regime: Close > EMA-200
            if close <= ema_200 or pd.isna(ema_200):
                continue

            # 2. Intermediate Trend: Close > EMA-50
            if close <= ema_50 or pd.isna(ema_50):
                continue

            # 3. Momentum Setup: RSI between rsi_lower and rsi_upper
            if pd.isna(rsi) or not (self.rsi_lower <= rsi <= self.rsi_upper):
                continue

            # 4. MACD confirmation: MACD > Signal
            if pd.isna(macd) or pd.isna(macd_sig) or macd <= macd_sig:
                continue

            # 5. Volatility & Stop sanity
            if pd.isna(atr) or atr <= 0.0:
                continue

            # 6. Volume filter
            if avg_vol > 0 and vol < (avg_vol * self.volume_factor):
                continue

            stop_loss = close - (self.atr_multiplier_stop * atr)
            profit_target = close + (self.atr_multiplier_target * atr)

            if stop_loss >= close or stop_loss <= 0:
                continue

            signals.append(
                Signal(
                    symbol=symbol,
                    date=current_date,
                    direction=SignalDirection.BUY,
                    stop_loss=round(stop_loss, 2),
                    profit_target=round(profit_target, 2),
                    reason=(
                        f"Trend pullback entry: Close > EMA50 > EMA200, "
                        f"RSI={rsi:.1f}, MACD={macd:.2f}>{macd_sig:.2f}"
                    ),
                )
            )

        return signals
