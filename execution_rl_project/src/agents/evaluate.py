from __future__ import annotations

from collections import Counter
from typing import Any, cast

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml


ACTION_NAME = {
    0: "HOLD",
    1: "BUY_LVL0",
    2: "BUY_LVL1",
    3: "BUY_LVL2",
    4: "BUY_MARKET",
}


def make_env(
    env_cfg: dict,
    book_path: str,
    trade_path: str,
    snapshot_path: str | None,
):
    def _init():
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
        )

    return _init


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    base_env = DummyVecEnv(
        [
            make_env(
                env_cfg=env_cfg,
                book_path=book_path,
                trade_path=trade_path,
                snapshot_path=snapshot_path,
            )
        ]
    )

    vecnorm_path = "/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints/vecnormalize_execution.pkl"
    model_path = "/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints/ppo_execution_agent_vecnorm.zip"

    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    obs = env.reset()

    done = False
    total_reward = 0.0
    action_counter: Counter[str] = Counter()
    trace_rows: list[dict[str, Any]] = []

    step_idx = 0
    prev_filled = 0.0
    final_info: dict[str, Any] | None = None
    raw_env = cast(ExecutionEnv, base_env.envs[0])

    while not done:
        state_before = raw_env.wrapper.get_market_state()

        action, _ = model.predict(cast(Any, obs), deterministic=True)
        action = int(action[0]) if not isinstance(action, int) else int(action)
        action_name = ACTION_NAME[action]
        action_counter[action_name] += 1

        obs, reward, dones, infos = env.step(np.array([action], dtype=np.int64))
        done = bool(dones[0])

        reward_scalar = float(reward[0])
        step_info = cast(dict[str, Any], infos[0])
        final_info = step_info
        total_reward += reward_scalar

        state_after = raw_env.wrapper.get_market_state()

        filled_qty = float(step_info["filled_qty"])
        remaining_qty = float(step_info["remaining_qty"])
        delta_fill = filled_qty - prev_filled
        prev_filled = filled_qty

        trace_rows.append(
            {
                "step": step_idx,
                "action": action_name,
                "best_bid_before": round(state_before.best_bid, 4),
                "best_ask_before": round(state_before.best_ask, 4),
                "best_bid_after": round(state_after.best_bid, 4),
                "best_ask_after": round(state_after.best_ask, 4),
                "reward": round(reward_scalar, 6),
                "shortfall": round(float(step_info["shortfall"]), 6),
                "shortfall_reward": round(float(step_info["shortfall_reward"]), 6),
                "terminal_penalty": round(float(step_info["terminal_penalty"]), 6),
                "avg_fill_price": round(float(step_info["avg_fill_price"]), 6),
                "delta_fill": round(delta_fill, 6),
                "cum_filled": round(filled_qty, 6),
                "remaining": round(remaining_qty, 6),
                "done": done,
            }
        )

        step_idx += 1

    if final_info is None:
        raise RuntimeError("Evaluation finished without producing any step info.")

    print("=== Evaluation Summary ===")
    print("total_reward:", round(total_reward, 6))
    print("arrival_price:", round(float(final_info["arrival_price"]), 6))
    print("filled_qty:", round(float(final_info["filled_qty"]), 6))
    print("remaining_qty:", round(float(final_info["remaining_qty"]), 6))
    print("equity:", round(float(final_info["equity"]), 6))
    print()

    print("=== Action Counts ===")
    total_actions = sum(action_counter.values())
    for action_name, count in action_counter.items():
        pct = 100.0 * count / max(total_actions, 1)
        print(f"{action_name:12s} {count:4d}  ({pct:6.2f}%)")
    print()

    print("=== Step Trace ===")
    for row in trace_rows:
        print(
            f"step={row['step']:02d} | "
            f"action={row['action']:12s} | "
            f"bid/ask_before=({row['best_bid_before']:.2f}, {row['best_ask_before']:.2f}) | "
            f"bid/ask_after=({row['best_bid_after']:.2f}, {row['best_ask_after']:.2f}) | "
            f"reward={row['reward']:+.6f} | "
            f"shortfall={row['shortfall']:+.6f} | "
            f"shortfall_reward={row['shortfall_reward']:+.6f} | "
            f"terminal={row['terminal_penalty']:+.6f} | "
            f"fill_px={row['avg_fill_price']:.6f} | "
            f"delta_fill={row['delta_fill']:.6f} | "
            f"cum_filled={row['cum_filled']:.6f} | "
            f"remaining={row['remaining']:.6f} | "
            f"done={row['done']}"
        )

    env.close()


if __name__ == "__main__":
    main()