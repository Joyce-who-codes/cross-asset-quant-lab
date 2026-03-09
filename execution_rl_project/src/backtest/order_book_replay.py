from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import heapq

import numpy as np
import pandas as pd

from src.data.parse_tardis_snapshot import snapshot_row_to_books


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
    Faster lightweight L2 replay engine.

    Main optimizations:
    - preload event columns to numpy arrays
    - integer-coded event/side arrays
    - binary search snapshot/event bootstrap points
    - top-k cache with dirty flag
    - O(1) rolling trade feature updates
    """

    def __init__(self, top_k: int = 5, trade_buffer_size: int = 200):
        self.top_k = top_k
        self.trade_buffer_size = trade_buffer_size

        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

        # store signed trade amounts
        self.trade_buffer: Deque[float] = deque(maxlen=trade_buffer_size)
        self.buy_vol = 0.0
        self.sell_vol = 0.0

        self.current_idx = 0
        self.current_ts = 0

        self.events: pd.DataFrame | None = None
        self.snapshots: pd.DataFrame | None = None

        self.n_events = 0
        self.ev_code_arr: np.ndarray | None = None
        self.exch_ts_arr: np.ndarray | None = None
        self.local_ts_arr: np.ndarray | None = None
        self.is_snapshot_arr: np.ndarray | None = None
        self.side_code_arr: np.ndarray | None = None
        self.price_arr: np.ndarray | None = None
        self.amount_arr: np.ndarray | None = None

        self.snapshot_ts_arr: np.ndarray | None = None

        self._cached_top_bids: list[tuple[float, float]] = []
        self._cached_top_asks: list[tuple[float, float]] = []
        self._book_dirty = True

    # ----------------------------
    # public api
    # ----------------------------
    def reset(
        self,
        events: pd.DataFrame,
        snapshots: Optional[pd.DataFrame] = None,
        start_idx: int = 0,
    ) -> None:
        if len(events) == 0:
            raise ValueError("events is empty")

        if self.events is not events:
            self.events = events.reset_index(drop=True)
            self._cache_event_arrays(self.events)

        if snapshots is not None and self.snapshots is not snapshots:
            self.snapshots = snapshots.reset_index(drop=True)
            self._cache_snapshot_arrays(self.snapshots)
        elif snapshots is None:
            self.snapshots = None
            self.snapshot_ts_arr = None

        self.current_idx = 0
        self.current_ts = int(self.exch_ts_arr[0])

        self.bids.clear()
        self.asks.clear()

        self.trade_buffer.clear()
        self.buy_vol = 0.0
        self.sell_vol = 0.0

        self._book_dirty = True
        self._cached_top_bids = []
        self._cached_top_asks = []

        self._bootstrap_book(start_idx=start_idx)

    def step_until_ts(self, target_exch_ts: int) -> None:
        if self.n_events == 0:
            return

        idx = self.current_idx
        exch_ts_arr = self.exch_ts_arr

        while idx < self.n_events and int(exch_ts_arr[idx]) <= target_exch_ts:
            self._apply_event_idx(idx)
            idx += 1

        self.current_idx = idx

    def is_done(self) -> bool:
        return self.current_idx >= self.n_events

    def get_market_state(self) -> ReplayMarketState:
        self._ensure_top_levels()

        top_bids = self._cached_top_bids
        top_asks = self._cached_top_asks

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
            timestamp_ns=int(self.current_ts * 1000),  # us -> ns
            recent_trade_imbalance=trade_imb,
            recent_signed_volume=signed_vol,
        )

    # ----------------------------
    # caching
    # ----------------------------
    def _cache_event_arrays(self, events: pd.DataFrame) -> None:
        self.n_events = len(events)

        self.ev_code_arr = events["event_code"].to_numpy(dtype=np.int8, copy=False)
        self.exch_ts_arr = events["exch_ts"].to_numpy(dtype=np.int64, copy=False)
        self.local_ts_arr = events["local_ts"].to_numpy(dtype=np.int64, copy=False)
        self.is_snapshot_arr = events["is_snapshot"].to_numpy(dtype=bool, copy=False)
        self.side_code_arr = events["side_code"].to_numpy(dtype=np.int8, copy=False)
        self.price_arr = events["price"].to_numpy(dtype=np.float64, copy=False)
        self.amount_arr = events["amount"].to_numpy(dtype=np.float64, copy=False)

    def _cache_snapshot_arrays(self, snapshots: pd.DataFrame) -> None:
        self.snapshot_ts_arr = snapshots["timestamp"].to_numpy(dtype=np.int64, copy=False)

    # ----------------------------
    # bootstrap
    # ----------------------------
    def _bootstrap_book(self, start_idx: int) -> None:
        start_idx = max(0, min(start_idx, self.n_events - 1))
        start_ts = int(self.exch_ts_arr[start_idx])

        if (
            self.snapshots is not None
            and self.snapshot_ts_arr is not None
            and len(self.snapshot_ts_arr) > 0
        ):
            replay_start_idx = self._bootstrap_from_snapshot25(start_ts)
            for j in range(replay_start_idx, start_idx + 1):
                self._apply_event_idx(j)
            self.current_idx = start_idx + 1
            return

        replay_start_idx = self._bootstrap_from_embedded_snapshot(start_idx)
        for j in range(replay_start_idx, start_idx + 1):
            self._apply_event_idx(j)
        self.current_idx = start_idx + 1

    def _bootstrap_from_snapshot25(self, start_ts: int) -> int:
        assert self.snapshots is not None
        assert self.snapshot_ts_arr is not None

        pos = np.searchsorted(self.snapshot_ts_arr, start_ts, side="right") - 1
        if pos < 0:
            return 0

        snap_row = self.snapshots.iloc[int(pos)]
        snap_ts = int(snap_row["timestamp"])

        bids, asks = snapshot_row_to_books(snap_row)
        self.bids = bids
        self.asks = asks
        self.current_ts = snap_ts
        self._book_dirty = True

        replay_start_idx = int(np.searchsorted(self.exch_ts_arr, snap_ts, side="right"))
        return replay_start_idx

    def _bootstrap_from_embedded_snapshot(self, start_idx: int) -> int:
        snap_ts = None
        for i in range(start_idx, -1, -1):
            if self.ev_code_arr[i] == 0 and bool(self.is_snapshot_arr[i]):
                snap_ts = int(self.exch_ts_arr[i])
                break

        if snap_ts is None:
            self.bids.clear()
            self.asks.clear()
            self._book_dirty = True
            return 0

        snap_start = int(np.searchsorted(self.exch_ts_arr, snap_ts, side="left"))

        self.bids.clear()
        self.asks.clear()
        self._book_dirty = True

        i = snap_start
        while i < self.n_events:
            if not (
                self.ev_code_arr[i] == 0
                and bool(self.is_snapshot_arr[i])
                and int(self.exch_ts_arr[i]) == snap_ts
            ):
                break
            self._apply_book_idx(i)
            i += 1

        return i

    # ----------------------------
    # apply events
    # ----------------------------
    def _apply_book_idx(self, idx: int) -> None:
        side_code = int(self.side_code_arr[idx])
        price = float(self.price_arr[idx])
        amount = float(self.amount_arr[idx])

        book = self.bids if side_code == 0 else self.asks

        if amount <= 0.0:
            book.pop(price, None)
        else:
            book[price] = amount

        self.current_ts = int(self.exch_ts_arr[idx])
        self._book_dirty = True

    def _apply_trade_idx(self, idx: int) -> None:
        side_code = int(self.side_code_arr[idx])
        amount = float(self.amount_arr[idx])

        sign_amount = amount if side_code == 2 else -amount

        if len(self.trade_buffer) == self.trade_buffer_size:
            old_val = self.trade_buffer.popleft()
            if old_val > 0:
                self.buy_vol -= old_val
            else:
                self.sell_vol -= (-old_val)

        self.trade_buffer.append(sign_amount)
        if sign_amount > 0:
            self.buy_vol += sign_amount
        else:
            self.sell_vol += -sign_amount

        self.current_ts = int(self.exch_ts_arr[idx])

    def _apply_event_idx(self, idx: int) -> None:
        if self.ev_code_arr[idx] == 0:
            self._apply_book_idx(idx)
        else:
            self._apply_trade_idx(idx)

    # ----------------------------
    # features
    # ----------------------------
    def _trade_features(self) -> tuple[float, float]:
        signed_vol = self.buy_vol - self.sell_vol
        denom = self.buy_vol + self.sell_vol
        imb = signed_vol / denom if denom > 1e-12 else 0.0
        return imb, signed_vol

    def _ensure_top_levels(self) -> None:
        if not self._book_dirty:
            return

        # O(N log K), better than full sorting when book is deep
        self._cached_top_bids = heapq.nlargest(
            self.top_k,
            self.bids.items(),
            key=lambda x: x[0],
        )
        self._cached_top_asks = heapq.nsmallest(
            self.top_k,
            self.asks.items(),
            key=lambda x: x[0],
        )

        # make final output strictly ordered
        self._cached_top_bids.sort(key=lambda x: x[0], reverse=True)
        self._cached_top_asks.sort(key=lambda x: x[0])

        self._book_dirty = False