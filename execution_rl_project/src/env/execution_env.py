from __future__ import annotations

import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.backtest.tardis_wrapper import TardisExecutionWrapper
from src.features.lob_features import (
    build_execution_state_features,
    compute_basic_lob_features,
)
from src.env.reward import compute_step_reward, compute_terminal_penalty
from src.microalpha.alpha_model import OnlineMicroAlpha

HOLD = 0
PLACE_BID1 = 1
PLACE_BID2 = 2
MARKET_BUY_SMALL = 3
CANCEL_ALL = 4


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: dict,
        book_path: str,
        trade_path: str,
        snapshot_path: str | None = None,
    ):
        super().__init__()
        self.cfg = env_cfg

        bt_cfg = env_cfg["backtest"]
        ex_cfg = env_cfg["execution"]
        rw_cfg = env_cfg["reward"]
        asset_cfg = env_cfg["asset"]

        self.wrapper = TardisExecutionWrapper(
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
            symbol=asset_cfg["symbol"],
            maker_fee=bt_cfg["maker_fee"],
            taker_fee=bt_cfg["taker_fee"],
            tick_size=asset_cfg["tick_size"],
            roi_lb=bt_cfg["roi_lb"],
            roi_ub=bt_cfg["roi_ub"],
            top_k=5,
        )

        self.target_qty = ex_cfg["target_qty"]
        self.horizon_sec = ex_cfg["horizon_sec"]
        self.step_sec = ex_cfg["step_sec"]
        self.market_clip_qty = ex_cfg["market_clip_qty"]

        self.lambda_wait = rw_cfg["lambda_wait"]
        self.lambda_terminal_remain = rw_cfg["lambda_terminal_remain"]
        self.exec_cost_coef = rw_cfg["exec_cost_coef"]
        self.taker_penalty_coef = rw_cfg["taker_penalty_coef"]

        self.max_steps = int(self.horizon_sec / self.step_sec)
        self.current_step = 0
        self.filled_qty = 0.0
        self.alpha_model = OnlineMicroAlpha(
            model_path="results/microalpha_v2/ridge_alpha.pkl",
            alpha_scale=1,
        )
        obs_dim = 4 + 5 + 5 + 5 + 5 + 5 + 6+1
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _sample_start_idx(self) -> int:
        min_start = 2000
        usable = max(min_start + 1, self.wrapper.num_events() - 5000)
        return random.randint(min_start, usable - 1)

    def _get_obs(self) -> np.ndarray:
        state = self.wrapper.get_market_state()

        lob_feats = compute_basic_lob_features(
            best_bid=state.best_bid,
            best_ask=state.best_ask,
            bid_prices=state.bid_prices,
            bid_sizes=state.bid_sizes,
            ask_prices=state.ask_prices,
            ask_sizes=state.ask_sizes,
            recent_trade_imbalance=state.recent_trade_imbalance,
            recent_signed_volume=state.recent_signed_volume,
        )

        exec_feats = build_execution_state_features(
            target_qty=self.target_qty,
            filled_qty=self.filled_qty,
            remaining_steps=max(0, self.max_steps - self.current_step),
            has_active_order=self.wrapper.has_active_order(),
            active_order_price_offset=self.wrapper.active_order_price_offset(),
        )

        alpha_pred = float(self.alpha_model.predict(state))
        alpha_pred = np.clip(alpha_pred, -1.0, 1.0)
        alpha_feat = np.array([alpha_pred], dtype=np.float32)

        return np.concatenate([lob_feats, exec_feats, alpha_feat], axis=0).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "start_idx" in options:
            start_idx = int(options["start_idx"])
        else:
            start_idx = self._sample_start_idx()

        self.wrapper.reset(start_idx=start_idx)
        self.current_step = 0
        self.filled_qty = 0.0
        self.alpha_model.reset()
        obs = self._get_obs()
        info = {
            "start_idx": start_idx,
        }
        return obs, info

    def step(self, action: int):
        state = self.wrapper.get_market_state()
        best_bid = state.best_bid
        decision_mid = 0.5 * (state.best_bid + state.best_ask)
        alpha_pred_step = float(self.alpha_model.predict(state))
        filled_now = 0.0
        avg_fill_price = 0.0
        is_taker_fill = False

        prev_pos = self.wrapper.get_position()
        remaining_qty = max(0.0, self.target_qty - self.filled_qty)

        if action == HOLD:
            pass

        elif action == PLACE_BID1 and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)
            self.wrapper.place_limit_buy(price=best_bid, qty=qty)

        elif action == PLACE_BID2 and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)
            tick = self.cfg["asset"]["tick_size"]
            self.wrapper.place_limit_buy(price=best_bid - tick, qty=qty)

        elif action == MARKET_BUY_SMALL and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)
            fill = self.wrapper.place_market_buy(qty=qty)
            filled_now = fill.filled_qty
            avg_fill_price = fill.avg_fill_price
            self.filled_qty += filled_now
            is_taker_fill = True

        elif action == CANCEL_ALL:
            self.wrapper.cancel_all()

        self.wrapper.step_time(self.step_sec)
        self.current_step += 1

        new_pos = self.wrapper.get_position()
        delta_pos = new_pos - prev_pos

        # 被动挂单在 step_time 期间成交
        if delta_pos > 1e-12 and filled_now <= 1e-12:
            filled_now = delta_pos
            # 这里仍然用 best_bid 作为一个简化近似
            avg_fill_price = best_bid
            self.filled_qty = new_pos
            is_taker_fill = False

        remaining_qty = max(0.0, self.target_qty - self.filled_qty)
        urgency = ((self.current_step + 1) / max(self.max_steps, 1)) ** 2

        reward, reward_parts = compute_step_reward(
            decision_mid_price=decision_mid,
            filled_qty=filled_now,
            avg_fill_price=avg_fill_price,
            remaining_qty=remaining_qty,
            target_qty=self.target_qty,
            urgency=urgency,
            lambda_wait=self.lambda_wait,
            exec_cost_coef=self.exec_cost_coef,
            taker_penalty_coef=self.taker_penalty_coef,
            is_taker_fill=is_taker_fill,
            alpha_signal=alpha_pred_step,
        )

        terminated = remaining_qty <= 1e-12
        truncated = self.current_step >= self.max_steps or self.wrapper.is_done()

        terminal_penalty = 0.0
        if truncated and not terminated:
            terminal_penalty = compute_terminal_penalty(
                remaining_qty=remaining_qty,
                target_qty=self.target_qty,
                lambda_terminal_remain=self.lambda_terminal_remain,
            )
            reward += terminal_penalty

        obs = self._get_obs()
        info = {
            "filled_qty": self.filled_qty,
            "remaining_qty": remaining_qty,
            "equity": self.wrapper.mark_to_market_equity(),
            "position": self.wrapper.get_position(),
            "exec_reward": reward_parts["exec_reward"],
            "taker_penalty": reward_parts["taker_penalty"],
            "wait_penalty": reward_parts["wait_penalty"],
            "terminal_penalty": float(terminal_penalty),
            "alpha_pred": alpha_pred_step,
            "alpha_wait_multiplier": reward_parts["alpha_wait_multiplier"],
        }     
        return obs, reward, terminated, truncated, info