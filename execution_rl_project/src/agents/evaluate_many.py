from __future__ import annotations

import argparse
import statistics
from typing import Any, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.evaluate import ACTION_NAME, build_env_cfg, make_env
from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml
from src.utils.tardis_chunk import build_chunk_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO execution agent across many test chunks")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--train-config", default="configs/train_btc_long.yaml")
    return parser.parse_args()


def run_single_chunk(
    symbol: str,
    side: str,
    chunk_cfg: dict[str, str],
    model_path: str,
    vecnorm_path: str,
) -> dict[str, Any]:
    env_cfg = build_env_cfg(symbol=symbol, side=side)
    env_cfg["execution"]["random_start"] = True
    env_cfg["execution"]["random_chunk"] = False
    env_cfg["execution"]["fixed_chunk_index"] = 0

    base_env = DummyVecEnv([make_env(env_cfg=env_cfg, chunk_paths=[chunk_cfg])])
    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path, env=env)

    raw_env = cast(ExecutionEnv, base_env.envs[0])
    obs = env.reset()
    done = False
    total_reward = 0.0
    action_counts: dict[str, int] = {name: 0 for name in ACTION_NAME.values()}
    final_info: dict[str, Any] | None = None

    while not done:
        action, _ = model.predict(cast(Any, obs), deterministic=True)
        action_id = int(action[0]) if not isinstance(action, int) else int(action)
        action_counts[ACTION_NAME[action_id]] += 1
        obs, reward, dones, infos = env.step(np.array([action_id], dtype=np.int64))
        done = bool(dones[0])
        total_reward += float(reward[0])
        final_info = cast(dict[str, Any], infos[0])

    env.close()

    if final_info is None:
        raise RuntimeError("Evaluation finished without final info")

    return {
        "chunk": chunk_cfg["chunk"],
        "total_reward": total_reward,
        "filled_qty": float(final_info["filled_qty"]),
        "remaining_qty": float(final_info["remaining_qty"]),
        "equity": float(final_info["equity"]),
        "action_counts": action_counts,
        "last_chunk": final_info.get("chunk", raw_env.current_chunk),
    }


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    side = args.side.lower()
    train_cfg = load_yaml(args.train_config)

    chunk_paths_dc = build_chunk_paths(
        symbol=symbol,
        start_day=train_cfg["test_start_day"],
        end_day=train_cfg["test_end_day"],
        chunk_hours=int(train_cfg.get("chunk_hours", 6)),
    )
    chunk_paths = [
        {
            "chunk": x.chunk,
            "book_path": x.book_path,
            "trade_path": x.trade_path,
            "snapshot_path": x.snapshot_path,
        }
        for x in chunk_paths_dc
    ]

    run_name = f"{symbol.lower()}_{side}_1m"
    ckpt_dir = f"/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints_{run_name}"
    vecnorm_path = f"{ckpt_dir}/{run_name}_vecnormalize.pkl"
    model_path = f"{ckpt_dir}/{run_name}.zip"

    results = [
        run_single_chunk(
            symbol=symbol,
            side=side,
            chunk_cfg=chunk_cfg,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
        )
        for chunk_cfg in chunk_paths
    ]

    rewards = [x["total_reward"] for x in results]
    print("=== Evaluation Many Summary ===")
    print("symbol:", symbol)
    print("side:", side)
    print("num_chunks:", len(results))
    print("avg_reward:", round(sum(rewards) / len(rewards), 6))
    print("std_reward:", round(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0, 6))
    print()

    for row in results:
        print(
            f"chunk={row['chunk']} | reward={row['total_reward']:+.6f} | "
            f"filled={row['filled_qty']:.6f} | remaining={row['remaining_qty']:.6f} | equity={row['equity']:.6f}"
        )


if __name__ == "__main__":
    main()
