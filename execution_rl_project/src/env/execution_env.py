from __future__ import annotations

import gc
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.backtest.tardis_wrapper import TardisExecutionWrapper
from src.features.lob_features import (
    build_execution_state_features,
    compute_lob_state_features,
)
from src.env.reward import (
    compute_step_reward as compute_shortfall_reward,
    compute_terminal_penalty as compute_shortfall_terminal_penalty,
)
from src.env.reward_twap_relative import (
    compute_step_reward as compute_twap_relative_reward,
    compute_terminal_reward as compute_twap_relative_terminal_reward,
)


# ============================================================
# IMPORTANT
#
# 0 HOLD
# 1 PLACE_PASSIVE_1
# 2 PLACE_PASSIVE_2
# 3 MARKET_SMALL
# 4 CANCEL_ALL
#
# buy:
#   1 -> best_bid
#   2 -> best_bid - tick
#   3 -> market buy
#
# sell:
#   1 -> best_ask
#   2 -> best_ask + tick
#   3 -> market sell
# ============================================================

HOLD = 0
PLACE_PASSIVE_1 = 1
PLACE_PASSIVE_2 = 2
MARKET_SMALL = 3
CANCEL_ALL = 4


class ExecutionEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: dict,
        book_path: str | None = None,
        trade_path: str | None = None,
        snapshot_path: str | None = None,
        chunk_paths: list[dict[str, str]] | None = None,
    ):
        super().__init__()

        self.cfg = env_cfg

        bt_cfg = env_cfg["backtest"]
        ex_cfg = env_cfg["execution"]
        rw_cfg = env_cfg["reward"]
        asset_cfg = env_cfg["asset"]

        self.chunk_paths = chunk_paths
        self.wrapper: TardisExecutionWrapper | None = None
        self.current_chunk: str | None = None

        if self.chunk_paths is None:
            if book_path is None or trade_path is None:
                raise ValueError("book_path/trade_path must be provided when chunk_paths is None")

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

        self.side = str(ex_cfg["side"]).lower()
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"unsupported execution side: {self.side}")

        self.target_qty = float(ex_cfg["target_qty"])
        self.horizon_sec = float(ex_cfg["horizon_sec"])
        self.step_sec = float(ex_cfg["step_sec"])
        self.market_clip_qty = float(ex_cfg["market_clip_qty"])

        self.random_start = ex_cfg.get("random_start", False)
        self.fixed_start_idx = ex_cfg.get("start_idx", 2000)
        self.min_start_idx = int(ex_cfg.get("min_start_idx", 2000))
        self.tail_buffer_events = int(ex_cfg.get("tail_buffer_events", 5000))
        self.log_chunk_on_reset = bool(ex_cfg.get("log_chunk_on_reset", False))
        self.episodes_per_chunk = int(ex_cfg.get("episodes_per_chunk", 1))
        self.episodes_on_current_chunk = 0

        self.lambda_terminal_remain = float(rw_cfg["lambda_terminal_remain"])
        self.reward_mode = str(rw_cfg.get("mode", "shortfall")).lower()
        self.lambda_lag = float(rw_cfg.get("lambda_lag", 0.0))
        self.lambda_taker = float(rw_cfg.get("lambda_taker", 0.0))
        self.lambda_excess = float(rw_cfg.get("lambda_excess", 0.0))

        self.agent_cum_qty = 0.0
        self.agent_total_cost = 0.0
        self.benchmark_cum_qty = 0.0
        self.benchmark_total_cost = 0.0

        self.max_steps = int(self.horizon_sec / self.step_sec)
        self.current_step = 0
        self.filled_qty = 0.0
        self.agent_cum_qty = 0.0
        self.agent_total_cost = 0.0
        self.benchmark_cum_qty = 0.0
        self.benchmark_total_cost = 0.0
        self.arrival_price = 0.0

        obs_dim = 6 + 2

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _require_wrapper(self) -> TardisExecutionWrapper:
        if self.wrapper is None:
            raise RuntimeError("wrapper is not initialized")
        return self.wrapper

    def _select_chunk_config(self) -> dict[str, str]:
        if self.chunk_paths is None:
            raise ValueError("chunk_paths is None")

        random_chunk = bool(self.cfg["execution"].get("random_chunk", False))
        fixed_chunk_index = int(self.cfg["execution"].get("fixed_chunk_index", 0))

        if random_chunk:
            idx = random.randint(0, len(self.chunk_paths) - 1)
        else:
            idx = max(0, min(fixed_chunk_index, len(self.chunk_paths) - 1))

        return self.chunk_paths[idx]

    def _release_wrapper(self) -> None:
        if self.wrapper is None:
            return

        old_wrapper = self.wrapper

        try:
            setattr(old_wrapper, "active_order", None)
            setattr(old_wrapper, "events", None)
            setattr(old_wrapper, "snapshots", None)

            if old_wrapper.replay is not None:
                replay = old_wrapper.replay
                setattr(replay, "events", None)
                setattr(replay, "snapshots", None)
                setattr(replay, "ev_code_arr", None)
                setattr(replay, "exch_ts_arr", None)
                setattr(replay, "local_ts_arr", None)
                setattr(replay, "is_snapshot_arr", None)
                setattr(replay, "side_code_arr", None)
                setattr(replay, "price_arr", None)
                setattr(replay, "amount_arr", None)
                setattr(replay, "snapshot_ts_arr", None)
                replay.trade_buffer.clear()

            setattr(old_wrapper, "replay", None)
        except Exception:
            pass

        self.wrapper = None
        self.current_chunk = None
        del old_wrapper
        gc.collect()

    def _rebuild_wrapper_for_chunk(self, chunk_cfg: dict[str, str]) -> None:
        bt_cfg = self.cfg["backtest"]
        asset_cfg = self.cfg["asset"]

        self._release_wrapper()

        self.wrapper = TardisExecutionWrapper(
            book_path=chunk_cfg["book_path"],
            trade_path=chunk_cfg["trade_path"],
            snapshot_path=chunk_cfg["snapshot_path"],
            symbol=asset_cfg["symbol"],
            maker_fee=bt_cfg["maker_fee"],
            taker_fee=bt_cfg["taker_fee"],
            tick_size=asset_cfg["tick_size"],
            roi_lb=bt_cfg["roi_lb"],
            roi_ub=bt_cfg["roi_ub"],
            top_k=5,
        )
        self.current_chunk = chunk_cfg["chunk"]

    def _sample_start_idx(self) -> int:
        wrapper = self._require_wrapper()
        max_valid_idx = max(0, wrapper.num_events() - 1)

        if not self.random_start:
            return min(int(self.fixed_start_idx), max_valid_idx)

        min_start = min(max(0, self.min_start_idx), max_valid_idx)
        max_start = max(min_start, wrapper.num_events() - self.tail_buffer_events)

        return random.randint(min_start, max_start)

    def _get_obs(self) -> np.ndarray:
        wrapper = self._require_wrapper()
        state = wrapper.get_market_state()

        current_mid = 0.5 * (state.best_bid + state.best_ask)

        lob_feats = compute_lob_state_features(
            best_bid=state.best_bid,
            best_ask=state.best_ask,
            bid_prices=state.bid_prices,
            bid_sizes=state.bid_sizes,
            ask_prices=state.ask_prices,
            ask_sizes=state.ask_sizes,
            mid_history=[current_mid],
        )

        exec_feats = build_execution_state_features(
            target_qty=self.target_qty,
            filled_qty=self.filled_qty,
            current_step=self.current_step,
            max_steps=self.max_steps,
        )

        return np.concatenate([lob_feats, exec_feats], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.chunk_paths is not None:
            need_new_chunk = (
                self.wrapper is None
                or self.current_chunk is None
                or self.episodes_on_current_chunk >= self.episodes_per_chunk
            )

            if need_new_chunk:
                chunk_cfg = self._select_chunk_config()
                self._rebuild_wrapper_for_chunk(chunk_cfg)
                self.episodes_on_current_chunk = 0

                if self.log_chunk_on_reset:
                    print(f"[reset] NEW chunk={chunk_cfg['chunk']}")
            else:
                if self.log_chunk_on_reset:
                    print(f"[reset] reuse chunk={self.current_chunk}")

            self.episodes_on_current_chunk += 1

        wrapper = self._require_wrapper()

        if options is not None and "start_idx" in options:
            start_idx = int(options["start_idx"])
        else:
            start_idx = self._sample_start_idx()

        wrapper.reset(start_idx=start_idx)

        self.current_step = 0
        self.filled_qty = 0.0

        state = wrapper.get_market_state()
        self.arrival_price = 0.5 * (state.best_bid + state.best_ask)

        obs = self._get_obs()
        info = {
            "start_idx": int(start_idx),
            "arrival_price": float(self.arrival_price),
            "side": self.side,
            "chunk": self.current_chunk,
        }

        return obs, info

    def step(self, action: int):
        wrapper = self._require_wrapper()
        state = wrapper.get_market_state()

        best_bid = float(state.best_bid)
        best_ask = float(state.best_ask)
        tick = float(self.cfg["asset"]["tick_size"])

        filled_now = 0.0
        avg_fill_price = 0.0
        taker_fill_qty = 0.0

        prev_pos = float(wrapper.get_position())
        remaining_qty = max(0.0, self.target_qty - self.filled_qty)

        if action == HOLD:
            pass

        elif action == PLACE_PASSIVE_1 and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)

            if self.side == "buy":
                wrapper.place_limit_buy(price=best_bid, qty=qty)
            else:
                wrapper.place_limit_sell(price=best_ask, qty=qty)

        elif action == PLACE_PASSIVE_2 and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)

            if self.side == "buy":
                wrapper.place_limit_buy(price=best_bid - tick, qty=qty)
            else:
                wrapper.place_limit_sell(price=best_ask + tick, qty=qty)

        elif action == MARKET_SMALL and remaining_qty > 0:
            qty = min(self.market_clip_qty, remaining_qty)

            if self.side == "buy":
                fill = wrapper.place_market_buy(qty=qty)
            else:
                fill = wrapper.place_market_sell(qty=qty)

            filled_now = float(fill.filled_qty)
            avg_fill_price = float(fill.avg_fill_price)
            taker_fill_qty = filled_now

        elif action == CANCEL_ALL:
            wrapper.cancel_all()

        wrapper.step_time(self.step_sec)
        self.current_step += 1

        new_pos = float(wrapper.get_position())

        if self.side == "buy":
            delta_pos = new_pos - prev_pos
        else:
            delta_pos = prev_pos - new_pos

        if delta_pos > 1e-12 and filled_now <= 1e-12:
            filled_now = float(delta_pos)

            if self.side == "buy":
                avg_fill_price = best_bid
            else:
                avg_fill_price = best_ask

        if self.side == "buy":
            self.filled_qty = float(new_pos)
        else:
            self.filled_qty = float(-new_pos)

        self.agent_cum_qty = self.filled_qty
        if filled_now > 1e-12:
            self.agent_total_cost += avg_fill_price * filled_now

        benchmark_target_cum_qty = min(
            self.target_qty,
            self.target_qty * self.current_step / max(self.max_steps, 1),
        )
        benchmark_step_qty = max(0.0, benchmark_target_cum_qty - self.benchmark_cum_qty)
        benchmark_price = best_ask if self.side == "buy" else best_bid
        if benchmark_step_qty > 1e-12:
            self.benchmark_cum_qty += benchmark_step_qty
            self.benchmark_total_cost += benchmark_price * benchmark_step_qty

        remaining_qty = max(0.0, self.target_qty - self.filled_qty)

        if self.reward_mode == "twap_relative":
            reward, reward_parts = compute_twap_relative_reward(
                arrival_price=self.arrival_price,
                filled_qty=filled_now,
                avg_fill_price=avg_fill_price,
                target_qty=self.target_qty,
                side=self.side,
                agent_cum_qty=self.agent_cum_qty,
                benchmark_cum_qty=self.benchmark_cum_qty,
                taker_fill_qty=taker_fill_qty,
                lambda_lag=self.lambda_lag,
                lambda_taker=self.lambda_taker,
            )
        else:
            reward, reward_parts = compute_shortfall_reward(
                arrival_price=self.arrival_price,
                filled_qty=filled_now,
                avg_fill_price=avg_fill_price,
                target_qty=self.target_qty,
                side=self.side,
            )
            reward_parts["lag_qty"] = 0.0
            reward_parts["lag_penalty"] = 0.0
            reward_parts["taker_fill_qty"] = float(taker_fill_qty)
            reward_parts["taker_penalty"] = 0.0
            reward_parts["reward"] = float(reward)

        terminated = remaining_qty <= 1e-12
        truncated = self.current_step >= self.max_steps or wrapper.is_done()

        terminal_penalty = 0.0
        terminal_parts = {
            "remaining_qty": float(remaining_qty),
            "remaining_ratio": float(max(0.0, remaining_qty) / max(self.target_qty, 1e-12)),
            "remain_penalty": 0.0,
            "agent_total_cost": float(self.agent_total_cost),
            "benchmark_total_cost": float(self.benchmark_total_cost),
            "excess_cost": 0.0,
            "excess_reward": 0.0,
            "reward": 0.0,
        }
        if terminated or truncated:
            if self.reward_mode == "twap_relative":
                terminal_penalty, terminal_parts = compute_twap_relative_terminal_reward(
                    remaining_qty=remaining_qty,
                    target_qty=self.target_qty,
                    agent_total_cost=self.agent_total_cost,
                    benchmark_total_cost=self.benchmark_total_cost,
                    lambda_terminal_remain=self.lambda_terminal_remain,
                    lambda_excess=self.lambda_excess,
                )
            else:
                terminal_penalty = compute_shortfall_terminal_penalty(
                    remaining_qty=remaining_qty,
                    target_qty=self.target_qty,
                    lambda_terminal_remain=self.lambda_terminal_remain,
                )
                terminal_parts["remain_penalty"] = float(terminal_penalty)
                terminal_parts["reward"] = float(terminal_penalty)
            reward += terminal_penalty

        obs = self._get_obs()
        info = {
            "side": self.side,
            "arrival_price": float(self.arrival_price),
            "avg_fill_price": float(avg_fill_price),
            "filled_now": float(filled_now),
            "filled_qty": float(self.filled_qty),
            "remaining_qty": float(remaining_qty),
            "equity": float(wrapper.mark_to_market_equity()),
            "position": float(wrapper.get_position()),
            "shortfall": float(reward_parts["shortfall"]),
            "shortfall_reward": float(reward_parts["shortfall_reward"]),
            "lag_qty": float(reward_parts["lag_qty"]),
            "lag_penalty": float(reward_parts["lag_penalty"]),
            "taker_fill_qty": float(reward_parts["taker_fill_qty"]),
            "taker_penalty": float(reward_parts["taker_penalty"]),
            "agent_cum_qty": float(self.agent_cum_qty),
            "benchmark_cum_qty": float(self.benchmark_cum_qty),
            "agent_total_cost": float(self.agent_total_cost),
            "benchmark_total_cost": float(self.benchmark_total_cost),
            "excess_cost": float(terminal_parts["excess_cost"]),
            "excess_reward": float(terminal_parts["excess_reward"]),
            "exec_reward": float(reward_parts["reward"]),
            "terminal_penalty": float(terminal_penalty),
            "terminal_reward": float(terminal_parts["reward"]),
            "reward_mode": self.reward_mode,
            "chunk": self.current_chunk,
        }

        return obs, float(reward), terminated, truncated, info
