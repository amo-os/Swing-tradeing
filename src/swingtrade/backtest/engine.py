"""Deterministic daily-bar historical backtesting engine.

Enforces point-in-time signal and order execution:
- Signals generated at close of bar T using historical data strictly <= T.
- Orders executed on bar T+1 at Open, subject to execution cost and slippage models.
- Stop losses evaluate intraday Low and gap-downs with realistic adverse fills.
- Universe strictly restricted to approved symbols validated by Phase A gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from swingtrade.backtest.costs import CostModel, OrderSide
from swingtrade.risk.position_sizer import PositionSizer
from swingtrade.strategy.base import Signal, SignalDirection, Strategy

logger = logging.getLogger(__name__)


class ExitReason(StrEnum):
    """Reason for exiting an open position."""

    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_GAP = "STOP_LOSS_GAP"
    PROFIT_TARGET = "PROFIT_TARGET"
    PROFIT_TARGET_GAP = "PROFIT_TARGET_GAP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    END_OF_TEST = "END_OF_TEST"


@dataclass
class Position:
    """Active open position in the portfolio."""

    symbol: str
    shares: int
    entry_date: pd.Timestamp
    nominal_entry_price: float
    effective_entry_price: float
    stop_loss: float
    profit_target: float | None
    entry_costs: float
    initial_risk: float
    last_price: float
    days_held: int = 0


@dataclass
class Trade:
    """Completed or final-state trade record."""

    trade_id: int
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares: int
    nominal_entry_price: float
    effective_entry_price: float
    nominal_exit_price: float
    effective_exit_price: float
    exit_reason: str
    entry_costs: float
    exit_costs: float
    total_costs: float
    pnl_gross: float
    pnl_net: float
    return_pct: float
    holding_days: int
    is_closed: bool


@dataclass(frozen=True)
class DailySnapshot:
    """End-of-day portfolio valuation snapshot."""

    date: pd.Timestamp
    cash: float
    holdings_value: float
    total_equity: float
    open_positions_count: int
    daily_return: float


@dataclass
class BacktestResult:
    """Complete simulation output container."""

    initial_capital: float
    final_equity: float
    trades: list[Trade]
    equity_curve: pd.DataFrame
    daily_snapshots: list[DailySnapshot]
    approved_symbols: list[str]
    rejected_signals_count: int = 0


@dataclass
class BacktestEngine:
    """Simulates daily swing-trading strategies chronologically."""

    initial_capital: float = 1_000_000.0
    cost_model: CostModel = field(default_factory=CostModel)
    position_sizer: PositionSizer = field(default_factory=PositionSizer)
    close_open_positions_at_end: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

    def run(
        self,
        *,
        strategy: Strategy,
        dataset: dict[str, pd.DataFrame],
        approved_universe: list[str] | set[str],
        start_date: pd.Timestamp | str | None = None,
        end_date: pd.Timestamp | str | None = None,
    ) -> BacktestResult:
        """Execute simulation across approved historical data.

        Parameters
        ----------
        strategy : Strategy
            The strategy instance generating signals.
        dataset : dict[str, pd.DataFrame]
            OHLCV data for symbols. Only approved symbols are traded.
        approved_universe : list[str] | set[str]
            Strictly approved symbols list from Phase A approval gate.
        """
        approved_set = {s.strip().upper() for s in approved_universe}
        clean_data: dict[str, pd.DataFrame] = {}

        # Validate that unapproved symbols are never loaded into simulation
        for sym, df in dataset.items():
            sym_norm = sym.strip().upper()
            if sym_norm not in approved_set:
                continue

            df_sorted = df.copy()
            df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
            df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)
            clean_data[sym_norm] = df_sorted

        if not clean_data:
            raise ValueError(
                "No valid data available for approved symbols in backtest universe"
            )

        # Build unified calendar of trading dates across all approved instruments
        all_timestamps: set[pd.Timestamp] = set()
        for df in clean_data.values():
            all_timestamps.update(df["timestamp"].tolist())
        calendar = sorted(all_timestamps)

        if calendar and calendar[0].tzinfo is not None:
            tz = calendar[0].tzinfo
            if start_date is not None:
                s_ts = pd.to_datetime(start_date)
                s_ts = s_ts.tz_localize(tz) if s_ts.tzinfo is None else s_ts.tz_convert(tz)
                calendar = [t for t in calendar if t >= s_ts]
            if end_date is not None:
                e_ts = pd.to_datetime(end_date)
                e_ts = e_ts.tz_localize(tz) if e_ts.tzinfo is None else e_ts.tz_convert(tz)
                calendar = [t for t in calendar if t <= e_ts]
        else:
            if start_date is not None:
                s_ts = pd.to_datetime(start_date)
                s_ts = s_ts.tz_localize(None) if s_ts.tzinfo is not None else s_ts
                calendar = [t for t in calendar if t >= s_ts]
            if end_date is not None:
                e_ts = pd.to_datetime(end_date)
                e_ts = e_ts.tz_localize(None) if e_ts.tzinfo is not None else e_ts
                calendar = [t for t in calendar if t <= e_ts]

        if not calendar:
            raise ValueError("No trading dates within specified start/end window")

        # Map each symbol to a date-indexed DataFrame for fast O(1) row lookup
        symbol_by_date: dict[str, dict[pd.Timestamp, pd.Series]] = {}
        for sym, df in clean_data.items():
            symbol_by_date[sym] = {
                row["timestamp"]: row for _, row in df.iterrows()
            }

        # Simulation state
        cash = float(self.initial_capital)
        open_positions: dict[str, Position] = {}
        trades: list[Trade] = []
        snapshots: list[DailySnapshot] = []
        pending_buy_orders: list[Signal] = []
        pending_sell_orders: set[str] = set()
        trade_id_counter = 1
        rejected_signals = 0

        prev_equity = cash

        for idx, current_date in enumerate(calendar):
            # -------------------------------------------------------------
            # STEP 1: Morning Execution on Bar T (current_date)
            # -------------------------------------------------------------
            # A) Process pending sell/exit orders generated at previous close
            for sym in list(pending_sell_orders):
                if sym in open_positions:
                    bar = symbol_by_date.get(sym, {}).get(current_date)
                    if bar is not None:
                        pos = open_positions.pop(sym)
                        exit_price = float(bar["open"])
                        cost = self.cost_model.calculate_costs(
                            OrderSide.SELL, pos.shares, exit_price
                        )
                        cash += cost.net_cash_impact
                        pnl_gross = (cost.nominal_price - pos.nominal_entry_price) * pos.shares
                        total_costs = pos.entry_costs + cost.total_costs
                        pnl_net = (cost.effective_price - pos.effective_entry_price) * pos.shares - (
                            pos.entry_costs + cost.total_statutory_charges + cost.brokerage
                        )
                        ret_pct = pnl_net / (pos.shares * pos.effective_entry_price)

                        trades.append(
                            Trade(
                                trade_id=trade_id_counter,
                                symbol=sym,
                                entry_date=pos.entry_date,
                                exit_date=current_date,
                                shares=pos.shares,
                                nominal_entry_price=pos.nominal_entry_price,
                                effective_entry_price=pos.effective_entry_price,
                                nominal_exit_price=exit_price,
                                effective_exit_price=cost.effective_price,
                                exit_reason=ExitReason.SIGNAL_EXIT.value,
                                entry_costs=pos.entry_costs,
                                exit_costs=cost.total_costs,
                                total_costs=total_costs,
                                pnl_gross=pnl_gross,
                                pnl_net=pnl_net,
                                return_pct=ret_pct,
                                holding_days=pos.days_held,
                                is_closed=True,
                            )
                        )
                        trade_id_counter += 1
            pending_sell_orders.clear()

            # B) Process pending BUY orders generated at previous close
            for signal in pending_buy_orders:
                sym = signal.symbol
                if sym in open_positions:
                    rejected_signals += 1
                    continue
                if len(open_positions) >= self.position_sizer.max_open_positions:
                    rejected_signals += 1
                    continue

                bar = symbol_by_date.get(sym, {}).get(current_date)
                if bar is None:
                    # Missing bar on open: cannot execute
                    rejected_signals += 1
                    continue

                open_price = float(bar["open"])
                stop_loss = signal.stop_loss
                if stop_loss is None or stop_loss >= open_price:
                    # Stop loss must be strictly below entry price
                    rejected_signals += 1
                    continue

                # Compute current equity for position sizer
                holdings_val = sum(
                    p.shares * p.last_price for p in open_positions.values()
                )
                current_equity = cash + holdings_val

                sizing = self.position_sizer.size_position(
                    symbol=sym,
                    current_equity=current_equity,
                    available_cash=cash,
                    entry_price=open_price,
                    stop_loss=stop_loss,
                    open_position_count=len(open_positions),
                )

                if not sizing.approved or sizing.shares < 1:
                    rejected_signals += 1
                    continue

                # Calculate buy transaction costs
                cost = self.cost_model.calculate_costs(
                    OrderSide.BUY, sizing.shares, open_price
                )
                cash_needed = abs(cost.net_cash_impact)

                if cash < cash_needed:
                    # Insufficient cash for full position + fees
                    rejected_signals += 1
                    continue

                cash -= cash_needed
                open_positions[sym] = Position(
                    symbol=sym,
                    shares=sizing.shares,
                    entry_date=current_date,
                    nominal_entry_price=open_price,
                    effective_entry_price=cost.effective_price,
                    stop_loss=stop_loss,
                    profit_target=signal.profit_target,
                    entry_costs=cost.total_costs,
                    initial_risk=sizing.risk_amount,
                    last_price=open_price,
                    days_held=0,
                )

            pending_buy_orders.clear()

            # -------------------------------------------------------------
            # STEP 2: Intraday Price Action on Bar T (Stops & Targets)
            # -------------------------------------------------------------
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                bar = symbol_by_date.get(sym, {}).get(current_date)
                if bar is None:
                    continue

                open_p = float(bar["open"])
                high_p = float(bar["high"])
                low_p = float(bar["low"])
                pos.last_price = float(bar["close"])
                pos.days_held += 1

                exited = False
                exit_price = 0.0
                exit_reason = ""

                # 1. Stop loss check
                if open_p <= pos.stop_loss:
                    # Gap down: filled at Open (worse than stop loss)
                    exit_price = open_p
                    exit_reason = ExitReason.STOP_LOSS_GAP.value
                    exited = True
                elif low_p <= pos.stop_loss:
                    # Intraday drop below stop loss: filled at stop price
                    exit_price = pos.stop_loss
                    exit_reason = ExitReason.STOP_LOSS.value
                    exited = True
                # 2. Profit target check (if target set and stop was not triggered)
                elif pos.profit_target is not None:
                    if open_p >= pos.profit_target:
                        exit_price = open_p
                        exit_reason = ExitReason.PROFIT_TARGET_GAP.value
                        exited = True
                    elif high_p >= pos.profit_target:
                        exit_price = pos.profit_target
                        exit_reason = ExitReason.PROFIT_TARGET.value
                        exited = True

                if exited:
                    open_positions.pop(sym)
                    cost = self.cost_model.calculate_costs(
                        OrderSide.SELL, pos.shares, exit_price
                    )
                    cash += cost.net_cash_impact
                    pnl_gross = (cost.nominal_price - pos.nominal_entry_price) * pos.shares
                    total_costs = pos.entry_costs + cost.total_costs
                    pnl_net = (cost.effective_price - pos.effective_entry_price) * pos.shares - (
                        pos.entry_costs + cost.total_statutory_charges + cost.brokerage
                    )
                    ret_pct = pnl_net / (pos.shares * pos.effective_entry_price)

                    trades.append(
                        Trade(
                            trade_id=trade_id_counter,
                            symbol=sym,
                            entry_date=pos.entry_date,
                            exit_date=current_date,
                            shares=pos.shares,
                            nominal_entry_price=pos.nominal_entry_price,
                            effective_entry_price=pos.effective_entry_price,
                            nominal_exit_price=exit_price,
                            effective_exit_price=cost.effective_price,
                            exit_reason=exit_reason,
                            entry_costs=pos.entry_costs,
                            exit_costs=cost.total_costs,
                            total_costs=total_costs,
                            pnl_gross=pnl_gross,
                            pnl_net=pnl_net,
                            return_pct=ret_pct,
                            holding_days=pos.days_held,
                            is_closed=True,
                        )
                    )
                    trade_id_counter += 1

            # -------------------------------------------------------------
            # STEP 3: End of Day Snapshot on Bar T
            # -------------------------------------------------------------
            holdings_val = sum(
                p.shares * p.last_price for p in open_positions.values()
            )
            day_equity = cash + holdings_val
            daily_ret = (day_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            prev_equity = day_equity

            snapshots.append(
                DailySnapshot(
                    date=current_date,
                    cash=cash,
                    holdings_value=holdings_val,
                    total_equity=day_equity,
                    open_positions_count=len(open_positions),
                    daily_return=daily_ret,
                )
            )

            # -------------------------------------------------------------
            # STEP 4: Session Close Strategy Evaluation (strictly historical slice <= T)
            # -------------------------------------------------------------
            # Create point-in-time slice up to current_date
            pit_data: dict[str, pd.DataFrame] = {}
            for sym, df in clean_data.items():
                slice_df = df[df["timestamp"] <= current_date]
                if not slice_df.empty:
                    pit_data[sym] = slice_df

            signals = strategy.generate_signals(
                current_date=current_date,
                historical_data=pit_data,
                open_positions=dict(open_positions),
            )

            for sig in signals:
                sym_n = sig.symbol.strip().upper()
                if sym_n not in approved_set:
                    # Enforce universe approval gate: reject unapproved symbol signal
                    rejected_signals += 1
                    continue

                if sig.direction == SignalDirection.BUY and sym_n not in open_positions:
                    pending_buy_orders.append(sig)
                elif sig.direction == SignalDirection.SELL and sym_n in open_positions:
                    pending_sell_orders.add(sym_n)

        # -------------------------------------------------------------
        # STEP 5: End-of-Test Open Position Handling
        # -------------------------------------------------------------
        if self.close_open_positions_at_end and open_positions:
            last_date = calendar[-1]
            for sym in list(open_positions.keys()):
                pos = open_positions.pop(sym)
                exit_price = pos.last_price
                cost = self.cost_model.calculate_costs(
                    OrderSide.SELL, pos.shares, exit_price
                )
                cash += cost.net_cash_impact
                pnl_gross = (cost.nominal_price - pos.nominal_entry_price) * pos.shares
                total_costs = pos.entry_costs + cost.total_costs
                pnl_net = (cost.effective_price - pos.effective_entry_price) * pos.shares - (
                    pos.entry_costs + cost.total_statutory_charges + cost.brokerage
                )
                ret_pct = pnl_net / (pos.shares * pos.effective_entry_price)

                trades.append(
                    Trade(
                        trade_id=trade_id_counter,
                        symbol=sym,
                        entry_date=pos.entry_date,
                        exit_date=last_date,
                        shares=pos.shares,
                        nominal_entry_price=pos.nominal_entry_price,
                        effective_entry_price=pos.effective_entry_price,
                        nominal_exit_price=exit_price,
                        effective_exit_price=cost.effective_price,
                        exit_reason=ExitReason.END_OF_TEST.value,
                        entry_costs=pos.entry_costs,
                        exit_costs=cost.total_costs,
                        total_costs=total_costs,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        return_pct=ret_pct,
                        holding_days=pos.days_held,
                        is_closed=True,
                    )
                )
                trade_id_counter += 1

        final_equity = cash + sum(
            p.shares * p.last_price for p in open_positions.values()
        )

        # Build equity curve DataFrame
        eq_records = [
            {
                "date": s.date,
                "cash": s.cash,
                "holdings_value": s.holdings_value,
                "total_equity": s.total_equity,
                "open_positions": s.open_positions_count,
                "daily_return": s.daily_return,
            }
            for s in snapshots
        ]
        eq_df = pd.DataFrame(eq_records)
        if not eq_df.empty:
            eq_df["peak"] = eq_df["total_equity"].cummax()
            eq_df["drawdown_pct"] = (
                (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
            ) * 100.0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            trades=trades,
            equity_curve=eq_df,
            daily_snapshots=snapshots,
            approved_symbols=sorted(approved_set),
            rejected_signals_count=rejected_signals,
        )
