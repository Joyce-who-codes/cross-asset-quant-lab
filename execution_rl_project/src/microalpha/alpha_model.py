from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle

import numpy as np


MICRO_ALPHA_FEATURES = [
    "spread_bps",
    "imb1",
    "imb5",
    "microdev",
    "ret_1",
    "ret_5",
    "trade_imb",
    "signed_vol",
]


class OnlineMicroAlpha:
    def __init__(self, model_path: str, alpha_scale: float = 10000.0):
        model_path = str(Path(model_path))
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.mid_hist: deque[float] = deque(maxlen=10)
        self.alpha_scale = float(alpha_scale)

    def reset(self) -> None:
        self.mid_hist.clear()

    def _build_feature_vector(self, state) -> np.ndarray:
        best_bid = float(state.best_bid)
        best_ask = float(state.best_ask)
        mid = 0.5 * (best_bid + best_ask)

        bid_sizes = np.asarray(state.bid_sizes, dtype=float)
        ask_sizes = np.asarray(state.ask_sizes, dtype=float)

        spread_bps = (best_ask - best_bid) / mid if mid > 0 else 0.0

        bid1 = float(bid_sizes[0]) if len(bid_sizes) > 0 else 0.0
        ask1 = float(ask_sizes[0]) if len(ask_sizes) > 0 else 0.0
        imb1 = (bid1 - ask1) / (bid1 + ask1 + 1e-12)

        bid5 = float(np.sum(bid_sizes[:5]))
        ask5 = float(np.sum(ask_sizes[:5]))
        imb5 = (bid5 - ask5) / (bid5 + ask5 + 1e-12)

        microprice = (
            best_ask * bid1 + best_bid * ask1
        ) / (bid1 + ask1 + 1e-12)
        microdev = (microprice - mid) / mid if mid > 0 else 0.0

        prev_mid_1 = self.mid_hist[-1] if len(self.mid_hist) >= 1 else None
        prev_mid_5 = self.mid_hist[-5] if len(self.mid_hist) >= 5 else None

        ret_1 = 0.0 if prev_mid_1 is None or prev_mid_1 <= 0 else (mid - prev_mid_1) / prev_mid_1
        ret_5 = 0.0 if prev_mid_5 is None or prev_mid_5 <= 0 else (mid - prev_mid_5) / prev_mid_5

        x = np.array(
            [
                spread_bps,
                imb1,
                imb5,
                microdev,
                ret_1,
                ret_5,
                float(state.recent_trade_imbalance),
                float(state.recent_signed_volume),
            ],
            dtype=np.float64,
        )

        return x.reshape(1, -1)

    def predict(self, state) -> float:
        x = self._build_feature_vector(state)
        pred = float(self.model.predict(x)[0])

        mid = 0.5 * (float(state.best_bid) + float(state.best_ask))
        self.mid_hist.append(mid)

        # convert raw predicted return into a more RL-friendly scale
        return pred * self.alpha_scale