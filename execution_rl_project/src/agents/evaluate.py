from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.baselines.passive_best_bid import run_passive_best_bid
from src.baselines.passive_then_sweep import run_passive_then_sweep
from src.baselines.twap_market import run_twap_market
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO execution agent against baselines on one chunk")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-start", type=int, default=2000)
    parser.add_argument("--tail-buffer", type=int, default=5000)
    parser.add_argument("--train-config", default="configs/train_btc_long.yaml")
    parser.add_argument("--chunk-root", type=str, default=None)
    return parser.parse_args()


def build_env_cfg(symbol: str, side: str, train_cfg: dict | None = None) -> dict:
    train_cfg = train_cfg or {}
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
        "reward": {
            "lambda_terminal_remain": float(train_cfg.get("lambda_terminal_remain", 3.0)),
            "mode": str(train_cfg.get("reward_mode", "shortfall")),
            "lambda_lag": float(train_cfg.get("lambda_lag", 0.0)),
            "lambda_taker": float(train_cfg.get("lambda_taker", 0.0)),
            "lambda_excess": float(train_cfg.get("lambda_excess", 0.0)),
        },
    }


def summarize(name: str, results: list[dict]) -> None:
    n = len(results)
    avg_reward = sum(x["reward"] for x in results) / n
    avg_fill = sum(x["filled_qty"] for x in results) / n
    avg_remain = sum(x["remaining_qty"] for x in results) / n
    avg_equity = sum(x["equity"] for x in results) / n
    avg_agent_cost = sum(x.get("agent_total_cost", 0.0) for x in results) / n
    avg_benchmark_cost = sum(x.get("benchmark_total_cost", 0.0) for x in results) / n
    avg_excess_cost = sum(x.get("excess_cost", 0.0) for x in results) / n
    avg_taker_fill = sum(x.get("taker_fill_qty", 0.0) for x in results) / n

    reward_std = statistics.pstdev([x["reward"] for x in results]) if n > 1 else 0.0
    fill_std = statistics.pstdev([x["filled_qty"] for x in results]) if n > 1 else 0.0
    remain_std = statistics.pstdev([x["remaining_qty"] for x in results]) if n > 1 else 0.0
    equity_std = statistics.pstdev([x["equity"] for x in results]) if n > 1 else 0.0

    print(f"{name}:")
    print(f"  episodes={n}")
    print(f"  avg_reward={avg_reward:.6f} | std_reward={reward_std:.6f}")
    print(f"  avg_filled={avg_fill:.6f} | std_filled={fill_std:.6f}")
    print(f"  avg_remaining={avg_remain:.6f} | std_remaining={remain_std:.6f}")
    print(f"  avg_equity={avg_equity:.6f} | std_equity={equity_std:.6f}")
    print(f"  avg_agent_cost={avg_agent_cost:.6f}")
    print(f"  avg_benchmark_cost={avg_benchmark_cost:.6f}")
    print(f"  avg_excess_cost={avg_excess_cost:.6f}")
    print(f"  avg_taker_fill={avg_taker_fill:.6f}")
    print()


def sample_start_indices(
    env: ExecutionEnv,
    episodes: int,
    min_start: int = 2000,
    tail_buffer: int = 5000,
    seed: int = 42,
) -> list[int]:
    rng = random.Random(seed)
    wrapper = env._require_wrapper()
    usable = max(min_start + 1, wrapper.num_events() - tail_buffer)
    return [rng.randint(min_start, usable - 1) for _ in range(episodes)]


def run_ppo_model(
    env_cfg: dict,
    book_path: str,
    trade_path: str,
    snapshot_path: str | None,
    model_path: str,
    vecnorm_path: str,
    start_indices: list[int],
) -> list[dict]:
    def make_env():
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
        )

    base_env = DummyVecEnv([make_env])
    env_norm = VecNormalize.load(vecnorm_path, base_env)
    env_norm.training = False
    env_norm.norm_reward = False

    model = PPO.load(model_path)
    raw_env = base_env.envs[0]

    results: list[dict] = []

    for start_idx in start_indices:
        obs_raw, _ = raw_env.reset(options={"start_idx": int(start_idx)})
        obs = env_norm.normalize_obs(np.asarray([obs_raw], dtype=np.float32))

        done = False
        final_info = None
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_id = int(action[0]) if not isinstance(action, int) else int(action)
            obs_raw, reward, terminated, truncated, info = raw_env.step(action_id)
            obs = env_norm.normalize_obs(np.asarray([obs_raw], dtype=np.float32))
            done = bool(terminated or truncated)
            total_reward += float(reward)
            final_info = info

        if final_info is None:
            raise RuntimeError("PPO evaluation finished without final info")

        results.append(
            {
                "start_idx": int(start_idx),
                "reward": float(total_reward),
                "filled_qty": float(final_info["filled_qty"]),
                "remaining_qty": float(final_info["remaining_qty"]),
                "equity": float(final_info["equity"]),
                "agent_total_cost": float(final_info.get("agent_total_cost", 0.0)),
                "benchmark_total_cost": float(final_info.get("benchmark_total_cost", 0.0)),
                "excess_cost": float(final_info.get("excess_cost", 0.0)),
                "taker_fill_qty": float(final_info.get("taker_fill_qty", 0.0)),
            }
        )

    env_norm.close()
    return results


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
    chunk_index = max(0, min(int(args.chunk_index), len(chunk_paths) - 1))
    chunk_cfg = chunk_paths[chunk_index]

    env_cfg = build_env_cfg(symbol=symbol, side=side, train_cfg=train_cfg)

    run_name_suffix = str(train_cfg.get("run_name_suffix", ""))
    run_name = f"{symbol.lower()}_{side}_1m{run_name_suffix}"
    ckpt_dir = PROJECT_ROOT / "results" / f"checkpoints_{run_name}"
    vecnorm_path = str(ckpt_dir / f"{run_name}_vecnormalize.pkl")
    model_path = str(ckpt_dir / f"{run_name}.zip")

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=chunk_cfg["book_path"],
        trade_path=chunk_cfg["trade_path"],
        snapshot_path=chunk_cfg["snapshot_path"],
    )

    start_indices = sample_start_indices(
        env,
        episodes=int(args.episodes),
        min_start=int(args.min_start),
        tail_buffer=int(args.tail_buffer),
        seed=int(args.seed),
    )

    print("=== Evaluation Setup ===")
    print("symbol:", symbol)
    print("side:", side)
    print("chunk_index:", chunk_index)
    print("chunk:", chunk_cfg["chunk"])
    print("episodes:", len(start_indices))
    print("seed:", int(args.seed))
    print("model_path:", model_path)
    print("vecnorm_path:", vecnorm_path)
    print()

    twap_results = run_twap_market(env, episodes=len(start_indices), start_indices=start_indices)
    summarize("TWAP Market", twap_results)

    best_bid_results = run_passive_best_bid(env, episodes=len(start_indices), start_indices=start_indices)
    summarize("Passive Best Bid", best_bid_results)

    passive_sweep_results = run_passive_then_sweep(env, episodes=len(start_indices), start_indices=start_indices)
    summarize("Passive Then Sweep", passive_sweep_results)

    env.close()

    ppo_results = run_ppo_model(
        env_cfg=env_cfg,
        book_path=chunk_cfg["book_path"],
        trade_path=chunk_cfg["trade_path"],
        snapshot_path=chunk_cfg["snapshot_path"],
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        start_indices=start_indices,
    )
    summarize("PPO Trained Model", ppo_results)


if __name__ == "__main__":
    main()
