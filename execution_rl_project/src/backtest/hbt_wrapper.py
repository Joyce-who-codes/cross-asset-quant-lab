from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MarketState:
    best_bid: float
    best_ask: float
    bid_sizes: list[float]
    ask_sizes: list[float]
    timestamp_ns: int


@dataclass
class FillInfo:
    filled_qty: float
    avg_fill_price: float
    fee: float


class HBTExecutionWrapper:
    """
    MVP wrapper.
    Current version is a mock simulator interface.
    Replace internals with real HFTBacktest calls later.
    """

    def __init__(self, npz_path: str, maker_fee: float, taker_fee: float):
        self.npz_path = npz_path
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self.current_idx = 0
        self.position = 0.0
        self.cash = 0.0

        self.active_order = None
        self.mid = 50000.0
        self.ts = 0

    def reset(self, start_idx: int = 0) -> MarketState:
        self.current_idx = start_idx
        self.position = 0.0
        self.cash = 0.0
        self.active_order = None
        self.mid = 50000.0 + np.random.randn() * 100
        self.ts = 1_700_000_000_000_000_000 + start_idx * 1_000_000_000
        return self.get_market_state()

    def get_market_state(self) -> MarketState:
        spread = 1.0
        best_bid = self.mid - spread / 2
        best_ask = self.mid + spread / 2
        bid_sizes = [1.0, 1.2, 1.5, 2.0, 2.5]
        ask_sizes = [1.1, 1.3, 1.6, 2.1, 2.6]
        return MarketState(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
            timestamp_ns=self.ts,
        )

    def step_time(self, step_sec: float) -> None:
        self.current_idx += 1
        self.ts += int(step_sec * 1e9)
        self.mid += np.random.randn() * 2.0
        self._maybe_fill_active_order()

    def _maybe_fill_active_order(self) -> None:
        if self.active_order is None:
            return

        side = self.active_order["side"]
        px = self.active_order["price"]
        qty = self.active_order["qty"]

        state = self.get_market_state()

        fill_prob = 0.2
        marketable = (side == "buy" and px >= state.best_ask)

        if marketable:
            fill_px = state.best_ask
            fee = fill_px * qty * self.taker_fee
            self.position += qty
            self.cash -= fill_px * qty + fee
            self.active_order = None
            return

        if np.random.rand() < fill_prob:
            fee = px * qty * self.maker_fee
            self.position += qty
            self.cash -= px * qty + fee
            self.active_order = None

    def place_limit_buy(self, price: float, qty: float) -> None:
        self.active_order = {
            "side": "buy",
            "type": "limit",
            "price": price,
            "qty": qty,
        }

    def place_market_buy(self, qty: float) -> FillInfo:
        state = self.get_market_state()
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
        state = self.get_market_state()
        return self.active_order["price"] - state.best_bid

    def get_position(self) -> float:
        return self.position

    def mark_to_market_equity(self) -> float:
        return self.cash + self.position * self.mid