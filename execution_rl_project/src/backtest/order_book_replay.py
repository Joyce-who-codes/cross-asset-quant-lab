from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd


@dataclass
class ReplayMarketState:
    best_bid: float
    best_ask: float
    bid_prices: list[float]
    bid_sizes: list[float]
    ask_prices: list[float]
    ask_sizes: list[float]
    timestamp_ns: int
    recent_trade_imbalance: float
    recent_signed_volume: float


class OrderBookReplay:
    """
    Lightweight L2 replay engine.

    Assumptions:
    - book event amount is the new absolute size at (side, price)
    - amount == 0 means delete that level
    - trade events do NOT mutate the book directly
    """

    def __init__(self, top_k: int = 5, trade_buffer_size: int = 200):
        self.top_k = top_k
        self.trade_buffer_size = trade_buffer_size

        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.trade_buffer: Deque[tuple[int, float]] = deque(maxlen=trade_buffer_size)

        self.current_idx = 0
        self.events: pd.DataFrame | None = None
        self.current_ts: int = 0

    def reset(self, events: pd.DataFrame, start_idx: int = 0) -> None:
        if len(events) == 0:
            raise ValueError("events is empty")

        self.events = events.reset_index(drop=True)
        self.current_idx = 0
        self.current_ts = int(self.events.iloc[0]["exch_ts"])
        self.bids.clear()
        self.asks.clear()
        self.trade_buffer.clear()

        self._bootstrap_book(start_idx=start_idx)

    def _bootstrap_book(self, start_idx: int) -> None:
        assert self.events is not None

        start_idx = max(0, min(start_idx, len(self.events) - 1))

        snap_ts = None
        for i in range(start_idx, -1, -1):
            row = self.events.iloc[i]
            if row["event_type"] == "book" and bool(row["is_snapshot"]):
                snap_ts = int(row["exch_ts"])
                break

        if snap_ts is None:
            # 没找到 snapshot，就从 0 累积到 start_idx
            for i in range(0, start_idx + 1):
                self._apply_event(self.events.iloc[i])
            self.current_idx = start_idx + 1
            return

        # 找到这一整个 snapshot 批次的起点
        snap_start = 0
        for i in range(start_idx, -1, -1):
            row = self.events.iloc[i]
            if (
                row["event_type"] == "book"
                and bool(row["is_snapshot"])
                and int(row["exch_ts"]) == snap_ts
            ):
                snap_start = i
            elif int(row["exch_ts"]) < snap_ts:
                break

        # 先清空，再灌入 snapshot batch
        self.bids.clear()
        self.asks.clear()

        i = snap_start
        while i < len(self.events):
            row = self.events.iloc[i]
            if not (
                row["event_type"] == "book"
                and bool(row["is_snapshot"])
                and int(row["exch_ts"]) == snap_ts
            ):
                break
            self._apply_book_event(row)
            i += 1

        # 再从 snapshot 之后推进到 start_idx
        for j in range(i, start_idx + 1):
            self._apply_event(self.events.iloc[j])

        self.current_idx = start_idx + 1

    def _apply_book_event(self, row: pd.Series) -> None:
        side = str(row["side"]).lower()
        price = float(row["price"])
        amount = float(row["amount"])

        book = self.bids if side == "bid" else self.asks

        if amount <= 0:
            book.pop(price, None)
        else:
            book[price] = amount

        self.current_ts = int(row["exch_ts"])

    def _apply_trade_event(self, row: pd.Series) -> None:
        side = str(row["side"]).lower()
        amount = float(row["amount"])
        sign = 1.0 if side == "buy" else -1.0
        self.trade_buffer.append((int(row["exch_ts"]), sign * amount))
        self.current_ts = int(row["exch_ts"])

    def _apply_event(self, row: pd.Series) -> None:
        if row["event_type"] == "book":
            self._apply_book_event(row)
        elif row["event_type"] == "trade":
            self._apply_trade_event(row)

    def step_n_events(self, n_events: int = 1) -> None:
        assert self.events is not None
        end_idx = min(self.current_idx + n_events, len(self.events))
        for i in range(self.current_idx, end_idx):
            self._apply_event(self.events.iloc[i])
        self.current_idx = end_idx

    def step_until_ts(self, target_exch_ts: int) -> None:
        assert self.events is not None
        while self.current_idx < len(self.events):
            row = self.events.iloc[self.current_idx]
            if int(row["exch_ts"]) > target_exch_ts:
                break
            self._apply_event(row)
            self.current_idx += 1

    def is_done(self) -> bool:
        assert self.events is not None
        return self.current_idx >= len(self.events)

    def _top_bids(self) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.top_k]

    def _top_asks(self) -> list[tuple[float, float]]:
        return sorted(self.asks.items(), key=lambda x: x[0])[: self.top_k]

    def _trade_features(self) -> tuple[float, float]:
        if not self.trade_buffer:
            return 0.0, 0.0

        signed_vol = float(sum(x[1] for x in self.trade_buffer))
        buy_vol = float(sum(max(x[1], 0.0) for x in self.trade_buffer))
        sell_vol = float(sum(max(-x[1], 0.0) for x in self.trade_buffer))
        denom = buy_vol + sell_vol
        imb = (buy_vol - sell_vol) / denom if denom > 1e-12 else 0.0
        return imb, signed_vol

    def get_market_state(self) -> ReplayMarketState:
        top_bids = self._top_bids()
        top_asks = self._top_asks()

        if not top_bids or not top_asks:
            raise RuntimeError("book is incomplete: missing bids or asks")

        bid_prices = [x[0] for x in top_bids]
        bid_sizes = [x[1] for x in top_bids]
        ask_prices = [x[0] for x in top_asks]
        ask_sizes = [x[1] for x in top_asks]

        while len(bid_prices) < self.top_k:
            bid_prices.append(bid_prices[-1])
            bid_sizes.append(0.0)
        while len(ask_prices) < self.top_k:
            ask_prices.append(ask_prices[-1])
            ask_sizes.append(0.0)

        trade_imb, signed_vol = self._trade_features()

        return ReplayMarketState(
            best_bid=bid_prices[0],
            best_ask=ask_prices[0],
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            timestamp_ns=int(self.current_ts * 1000),  # microseconds -> ns
            recent_trade_imbalance=trade_imb,
            recent_signed_volume=signed_vol,
        )