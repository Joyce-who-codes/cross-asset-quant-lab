from __future__ import annotations

from collections import Counter
from stable_baselines3 import PPO

from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml


ACTION_NAME = {
    0: "HOLD",
    1: "PLACE_BID1",
    2: "PLACE_BID2",
    3: "MARKET_BUY_SMALL",
    4: "CANCEL_ALL",
}


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
    )
    model = PPO.load("/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints/ppo_execution_agent_with_alpha_reward2.zip")

    obs, info = env.reset(options={"start_idx": 2000})
    done = False
    total_reward = 0.0
    action_counter = Counter()
    trace_rows = []

    step_idx = 0
    prev_filled = 0.0

    while not done:
        state_before = env.wrapper.get_market_state()

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_name = ACTION_NAME[action]
        action_counter[action_name] += 1

        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        state_after = env.wrapper.get_market_state()

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
                "reward": round(float(reward), 6),
                "delta_fill": round(delta_fill, 6),
                "cum_filled": round(filled_qty, 6),
                "remaining": round(remaining_qty, 6),
                "exec_reward": round(float(step_info["exec_reward"]), 6),
                "taker_penalty": round(float(step_info["taker_penalty"]), 6),
                "wait_penalty": round(float(step_info["wait_penalty"]), 6),
                "terminal_penalty": round(float(step_info["terminal_penalty"]), 6),
                "alpha_pred": round(float(step_info["alpha_pred"]), 6) if step_info["alpha_pred"] is not None else None,
                "done": done,
            }
        )

        step_idx += 1

    print("=== Evaluation Summary ===")
    print("total_reward:", round(total_reward, 6))
    print("filled_qty:", round(float(step_info["filled_qty"]), 6))
    print("remaining_qty:", round(float(step_info["remaining_qty"]), 6))
    print("equity:", round(float(step_info["equity"]), 6))
    print()

    print("=== Action Counts ===")
    total_actions = sum(action_counter.values())
    for action_name, count in action_counter.items():
        pct = 100.0 * count / max(total_actions, 1)
        print(f"{action_name:18s} {count:4d}  ({pct:6.2f}%)")
    print()

    print("=== Step Trace ===")
    for row in trace_rows:
        print(
            f"step={row['step']:02d} | "
            f"action={row['action']:16s} | "
            f"bid/ask_before=({row['best_bid_before']:.2f}, {row['best_ask_before']:.2f}) | "
            f"bid/ask_after=({row['best_bid_after']:.2f}, {row['best_ask_after']:.2f}) | "
            f"reward={row['reward']:+.6f} | "
            f"exec={row['exec_reward']:+.6f} | "
            f"wait={row['wait_penalty']:+.6f} | "
            f"taker={row['taker_penalty']:+.6f} | "
            f"terminal={row['terminal_penalty']:+.6f} | "
            f"alpha={row['alpha_pred']:+.6f} | "
            f"delta_fill={row['delta_fill']:.6f} | "
            f"cum_filled={row['cum_filled']:.6f} | "
            f"remaining={row['remaining']:.6f} | "
            f"done={row['done']}"
        )


if __name__ == "__main__":
    main()