from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml
from src.utils.project_paths import PROJECT_ROOT
from src.utils.tardis_chunk import build_chunk_paths


# Must match ExecutionEnv action ids exactly
ACTION_NAME = {
    0: "HOLD",
    1: "PLACE_PASSIVE_1",
    2: "PLACE_PASSIVE_2",
    3: "MARKET_SMALL",
    4: "CANCEL_ALL",
}


def make_env(
    env_cfg: dict,
    chunk_paths: list[dict[str, str]],
):
    def _init():
        return ExecutionEnv(
            env_cfg=env_cfg,
            chunk_paths=chunk_paths,
        )

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO execution agent on chunk parquet data")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--train-config", default="configs/train_btc_long.yaml")
    parser.add_argument("--chunk-root", type=str, default=None)
    return parser.parse_args()


def build_env_cfg(symbol: str, side: str) -> dict:
    return {
        "asset": {"symbol": symbol, "tick_size": 0.1, "lot_size": 0.001},
        "backtest": {
            "maker_fee": 0.0,
            "taker_fee": 0.0002,
            "entry_latency_ns": 1000000,
            "response_latency_ns": 1000000,
            "queue_model_power": 2.0,
            "roi_lb": 10000,
            "roi_ub": 200000,
        },
        "execution": {
            "side": side,
            "target_qty": 0.05,
            "horizon_sec": 120,
            "step_sec": 2,
            "market_clip_qty": 0.005,
            "max_active_orders": 1,
            "start_idx": 2000,
            "random_start": True,
            "random_chunk": False,
        },
        "reward": {"lambda_terminal_remain": 3.0},
    }


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    side = args.side.lower()
    train_config_path = Path(args.train_config)
    if not train_config_path.is_absolute():
        train_config_path = PROJECT_ROOT / train_config_path
    train_cfg = load_yaml(str(train_config_path))

    chunk_paths = build_chunk_paths(
        symbol=symbol,
        start_day=train_cfg["test_start_day"],
        end_day=train_cfg["test_end_day"],
        chunk_hours=int(train_cfg.get("chunk_hours", 6)),
        chunk_root=args.chunk_root,
    )

    env_cfg = build_env_cfg(symbol=symbol, side=side)
    env_cfg["execution"]["fixed_chunk_index"] = int(args.chunk_index)

    run_name = f"{symbol.lower()}_{side}_1m"
    ckpt_dir = PROJECT_ROOT / "results" / f"checkpoints_{run_name}"
    vecnorm_path = str(ckpt_dir / f"{run_name}_vecnormalize.pkl")
    model_path = str(ckpt_dir / f"{run_name}.zip")

    base_env = DummyVecEnv(
        [
            make_env(
                env_cfg=env_cfg,
                chunk_paths=chunk_paths,
            )
        ]
    )

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
    final_info: dict[str, Any] | None = None

    raw_env = cast(ExecutionEnv, base_env.envs[0])

    while not done:
        state_before = raw_env._require_wrapper().get_market_state()

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

        state_after = raw_env._require_wrapper().get_market_state()

        filled_now = float(step_info["filled_now"])
        filled_qty = float(step_info["filled_qty"])
        remaining_qty = float(step_info["remaining_qty"])

        trace_rows.append(
            {
                "step": step_idx,
                "action": action_name,
                "best_bid_before": round(float(state_before.best_bid), 4),
                "best_ask_before": round(float(state_before.best_ask), 4),
                "best_bid_after": round(float(state_after.best_bid), 4),
                "best_ask_after": round(float(state_after.best_ask), 4),
                "reward": round(reward_scalar, 6),
                "shortfall": round(float(step_info["shortfall"]), 6),
                "shortfall_reward": round(float(step_info["shortfall_reward"]), 6),
                "terminal_penalty": round(float(step_info["terminal_penalty"]), 6),
                "avg_fill_price": round(float(step_info["avg_fill_price"]), 6),
                "filled_now": round(filled_now, 6),
                "cum_filled": round(filled_qty, 6),
                "remaining": round(remaining_qty, 6),
                "position": round(float(step_info["position"]), 6),
                "done": done,
            }
        )

        step_idx += 1

    if final_info is None:
        raise RuntimeError("Evaluation finished without producing any step info.")

    print("=== Evaluation Summary ===")
    print("side:", side)
    print("total_reward:", round(total_reward, 6))
    print("arrival_price:", round(float(final_info["arrival_price"]), 6))
    print("filled_qty:", round(float(final_info["filled_qty"]), 6))
    print("remaining_qty:", round(float(final_info["remaining_qty"]), 6))
    print("position:", round(float(final_info["position"]), 6))
    print("equity:", round(float(final_info["equity"]), 6))
    print()

    print("=== Action Counts ===")
    total_actions = sum(action_counter.values())
    for action_name, count in action_counter.items():
        pct = 100.0 * count / max(total_actions, 1)
        print(f"{action_name:16s} {count:4d}  ({pct:6.2f}%)")
    print()

    print("=== Step Trace ===")
    for row in trace_rows:
        print(
            f"step={row['step']:02d} | "
            f"action={row['action']:16s} | "
            f"bid/ask_before=({row['best_bid_before']:.2f}, {row['best_ask_before']:.2f}) | "
            f"bid/ask_after=({row['best_bid_after']:.2f}, {row['best_ask_after']:.2f}) | "
            f"reward={row['reward']:+.6f} | "
            f"shortfall={row['shortfall']:+.6f} | "
            f"shortfall_reward={row['shortfall_reward']:+.6f} | "
            f"terminal={row['terminal_penalty']:+.6f} | "
            f"fill_px={row['avg_fill_price']:.6f} | "
            f"filled_now={row['filled_now']:.6f} | "
            f"cum_filled={row['cum_filled']:.6f} | "
            f"remaining={row['remaining']:.6f} | "
            f"position={row['position']:.6f} | "
            f"done={row['done']}"
        )

    env.close()


if __name__ == "__main__":
    main()
