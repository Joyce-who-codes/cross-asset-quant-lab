from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle

import numpy as np


MICRO_ALPHA_FEATURES = [
    "spread_bps",
    "imb1",
    "imb3",
    "imb5",
    "microdev_bps",
    "depth_ratio_5",
    "ret_1_bps",
    "ret_5_bps",
    "ret_10_bps",
    "vol_10_bps",
    "ofi_1",
    "ofi_5",
    "trade_imb",
    "signed_vol_log",
]


def _safe_ratio(a: float, b: float) -> float:
    return a / (b + 1e-12)


class OnlineMicroAlpha:
    def __init__(self, model_path: str, alpha_scale: float = 1.0):
        model_path = str(Path(model_path))
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.mid_hist: deque[float] = deque(maxlen=32)
        self.alpha_scale = float(alpha_scale)

        self.prev_bid1: float | None = None
        self.prev_ask1: float | None = None
        self.prev_bid1_size: float | None = None
        self.prev_ask1_size: float | None = None
        self.prev_bid5_sum: float | None = None
        self.prev_ask5_sum: float | None = None

    def reset(self) -> None:
        self.mid_hist.clear()
        self.prev_bid1 = None
        self.prev_ask1 = None
        self.prev_bid1_size = None
        self.prev_ask1_size = None
        self.prev_bid5_sum = None
        self.prev_ask5_sum = None

    def _build_feature_vector(self, state) -> np.ndarray:
        best_bid = float(state.best_bid)
        best_ask = float(state.best_ask)
        mid = 0.5 * (best_bid + best_ask)

        bid_sizes = np.asarray(state.bid_sizes, dtype=float)
        ask_sizes = np.asarray(state.ask_sizes, dtype=float)

        bid1 = float(bid_sizes[0]) if len(bid_sizes) > 0 else 0.0
        ask1 = float(ask_sizes[0]) if len(ask_sizes) > 0 else 0.0

        bid3 = float(np.sum(bid_sizes[:3]))
        ask3 = float(np.sum(ask_sizes[:3]))
        bid5 = float(np.sum(bid_sizes[:5]))
        ask5 = float(np.sum(ask_sizes[:5]))

        spread_bps = _safe_ratio(best_ask - best_bid, mid) * 10000.0 if mid > 0 else 0.0
        imb1 = _safe_ratio(bid1 - ask1, bid1 + ask1)
        imb3 = _safe_ratio(bid3 - ask3, bid3 + ask3)
        imb5 = _safe_ratio(bid5 - ask5, bid5 + ask5)

        microprice = (best_ask * bid1 + best_bid * ask1) / (bid1 + ask1 + 1e-12)
        microdev_bps = _safe_ratio(microprice - mid, mid) * 10000.0 if mid > 0 else 0.0

        depth_ratio_5 = float(np.log((bid5 + 1e-12) / (ask5 + 1e-12)))

        prev_mid_1 = self.mid_hist[-1] if len(self.mid_hist) >= 1 else None
        prev_mid_5 = self.mid_hist[-5] if len(self.mid_hist) >= 5 else None
        prev_mid_10 = self.mid_hist[-10] if len(self.mid_hist) >= 10 else None

        ret_1_bps = 0.0 if prev_mid_1 is None or prev_mid_1 <= 0 else _safe_ratio(mid - prev_mid_1, prev_mid_1) * 10000.0
        ret_5_bps = 0.0 if prev_mid_5 is None or prev_mid_5 <= 0 else _safe_ratio(mid - prev_mid_5, prev_mid_5) * 10000.0
        ret_10_bps = 0.0 if prev_mid_10 is None or prev_mid_10 <= 0 else _safe_ratio(mid - prev_mid_10, prev_mid_10) * 10000.0

        if len(self.mid_hist) >= 10:
            mids_arr = np.asarray(list(self.mid_hist)[-10:], dtype=float)
            mid_rets = np.diff(mids_arr) / (mids_arr[:-1] + 1e-12)
            vol_10_bps = float(np.std(mid_rets) * 10000.0)
        else:
            vol_10_bps = 0.0

        if (
            self.prev_bid1 is None
            or self.prev_ask1 is None
            or self.prev_bid1_size is None
            or self.prev_ask1_size is None
        ):
            ofi_1 = 0.0
        else:
            bid_term = 0.0
            ask_term = 0.0

            if best_bid > self.prev_bid1:
                bid_term += bid1
            elif best_bid < self.prev_bid1:
                bid_term -= self.prev_bid1_size
            else:
                bid_term += (bid1 - self.prev_bid1_size)

            if best_ask < self.prev_ask1:
                ask_term += ask1
            elif best_ask > self.prev_ask1:
                ask_term -= self.prev_ask1_size
            else:
                ask_term += (ask1 - self.prev_ask1_size)

            ofi_1 = bid_term - ask_term

        if self.prev_bid5_sum is None or self.prev_ask5_sum is None:
            ofi_5 = 0.0
        else:
            ofi_5 = (bid5 - self.prev_bid5_sum) - (ask5 - self.prev_ask5_sum)

        signed_vol = float(state.recent_signed_volume)
        signed_vol_log = float(np.sign(signed_vol) * np.log1p(abs(signed_vol)))

        x = np.array(
            [
                spread_bps,
                imb1,
                imb3,
                imb5,
                microdev_bps,
                depth_ratio_5,
                ret_1_bps,
                ret_5_bps,
                ret_10_bps,
                vol_10_bps,
                ofi_1,
                ofi_5,
                float(state.recent_trade_imbalance),
                signed_vol_log,
            ],
            dtype=np.float64,
        )

        return x.reshape(1, -1)

    def predict(self, state) -> float:
        x = self._build_feature_vector(state)
        pred = float(self.model.predict(x)[0])

        mid = 0.5 * (float(state.best_bid) + float(state.best_ask))
        self.mid_hist.append(mid)

        self.prev_bid1 = float(state.best_bid)
        self.prev_ask1 = float(state.best_ask)
        self.prev_bid1_size = float(state.bid_sizes[0])
        self.prev_ask1_size = float(state.ask_sizes[0])
        self.prev_bid5_sum = float(np.sum(state.bid_sizes[:5]))
        self.prev_ask5_sum = float(np.sum(state.ask_sizes[:5]))

        return pred * self.alpha_scale