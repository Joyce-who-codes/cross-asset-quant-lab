from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.backtest.order_book_replay import OrderBookReplay
from src.data.parse_tardis_csv import load_merged_events
from src.data.parse_tardis_snapshot import load_snapshot25_csv


@dataclass
class MarketState:
    best_bid: float
    best_ask: float
    bid_prices: list[float]
    bid_sizes: list[float]
    ask_prices: list[float]
    ask_sizes: list[float]
    timestamp_ns: int
    recent_trade_imbalance: float
    recent_signed_volume: float


@dataclass
class FillInfo:
    filled_qty: float
    avg_fill_price: float
    fee: float


class TardisExecutionWrapper:
    def __init__(
        self,
        book_path: str,
        trade_path: str,
        symbol: str,
        maker_fee: float,
        taker_fee: float,
        top_k: int = 5,
        snapshot_path: Optional[str] = None,
    ):
        self.book_path = book_path
        self.trade_path = trade_path
        self.snapshot_path = snapshot_path
        self.symbol = symbol
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.top_k = top_k

        self.events: pd.DataFrame = load_merged_events(
            book_path=book_path,
            trade_path=trade_path,
            symbol=symbol,
            sort_by="exch_ts",
        )
        self.snapshots: pd.DataFrame | None = None
        if snapshot_path is not None:
            self.snapshots = load_snapshot25_csv(snapshot_path, symbol=symbol)

        self.replay = OrderBookReplay(top_k=top_k)

        self.position = 0.0
        self.cash = 0.0
        self.active_order: dict | None = None

    def reset(self, start_idx: int = 0) -> MarketState:
        self.position = 0.0
        self.cash = 0.0
        self.active_order = None

        self.replay.reset(self.events, snapshots=self.snapshots, start_idx=start_idx)
        s = self.replay.get_market_state()
        return self._convert_state(s)

    def _convert_state(self, s) -> MarketState:
        return MarketState(
            best_bid=s.best_bid,
            best_ask=s.best_ask,
            bid_prices=s.bid_prices,
            bid_sizes=s.bid_sizes,
            ask_prices=s.ask_prices,
            ask_sizes=s.ask_sizes,
            timestamp_ns=s.timestamp_ns,
            recent_trade_imbalance=s.recent_trade_imbalance,
            recent_signed_volume=s.recent_signed_volume,
        )

    def get_market_state(self) -> MarketState:
        return self._convert_state(self.replay.get_market_state())

    def step_time(self, step_sec: float) -> None:
        current_state = self.replay.get_market_state()
        target_ts_us = int(current_state.timestamp_ns / 1000 + step_sec * 1_000_000)
        self.replay.step_until_ts(target_ts_us)
        self._maybe_fill_active_order()

    def _maybe_fill_active_order(self) -> None:
        if self.active_order is None:
            return

        state = self.replay.get_market_state()
        side = self.active_order["side"]
        price = float(self.active_order["price"])
        qty = float(self.active_order["qty"])

        if side != "buy":
            raise NotImplementedError("current MVP only supports buy side")

        if state.best_ask <= price:
            fee = price * qty * self.maker_fee
            self.position += qty
            self.cash -= price * qty + fee
            self.active_order = None

    def place_limit_buy(self, price: float, qty: float) -> None:
        state = self.replay.get_market_state()

        if price >= state.best_ask:
            fee = state.best_ask * qty * self.taker_fee
            self.position += qty
            self.cash -= state.best_ask * qty + fee
            self.active_order = None
            return

        self.active_order = {
            "side": "buy",
            "type": "limit",
            "price": float(price),
            "qty": float(qty),
        }

    def place_market_buy(self, qty: float) -> FillInfo:
        state = self.replay.get_market_state()
        px = state.best_ask
        fee = px * qty * self.taker_fee
        self.position += qty
        self.cash -= px * qty + fee
        return FillInfo(filled_qty=qty, avg_fill_price=px, fee=fee)

    def cancel_all(self) -> None:
        self.active_order = None

    def has_active_order(self) -> bool:
        return self.active_order is not None

    def active_order_price_offset(self) -> float:
        if self.active_order is None:
            return 0.0
        state = self.replay.get_market_state()
        return float(self.active_order["price"]) - state.best_bid

    def get_position(self) -> float:
        return self.position

    def mark_to_market_equity(self) -> float:
        state = self.replay.get_market_state()
        mid = 0.5 * (state.best_bid + state.best_ask)
        return self.cash + self.position * mid

    def is_done(self) -> bool:
        return self.replay.is_done()

    def num_events(self) -> int:
        return len(self.events)