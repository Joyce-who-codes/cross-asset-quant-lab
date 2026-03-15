from __future__ import annotations

from pathlib import Path
import random
import statistics
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def summarize(name: str, results: list[dict]) -> None:
    n = len(results)
    avg_reward = sum(x["reward"] for x in results) / n
    avg_fill = sum(x["filled_qty"] for x in results) / n
    avg_remain = sum(x["remaining_qty"] for x in results) / n

    reward_std = statistics.pstdev([x["reward"] for x in results]) if n > 1 else 0.0
    fill_std = statistics.pstdev([x["filled_qty"] for x in results]) if n > 1 else 0.0
    remain_std = statistics.pstdev([x["remaining_qty"] for x in results]) if n > 1 else 0.0

    print(f"{name}:")
    print(f"  episodes={n}")
    print(f"  avg_reward={avg_reward:.6f} | std_reward={reward_std:.6f}")
    print(f"  avg_filled={avg_fill:.6f} | std_filled={fill_std:.6f}")
    print(f"  avg_remaining={avg_remain:.6f} | std_remaining={remain_std:.6f}")
    print()


def sample_start_indices(
    env,
    episodes: int,
    min_start: int = 2000,
    tail_buffer: int = 5000,
    seed: int = 42,
) -> list[int]:
    rng = random.Random(seed)
    usable = max(min_start + 1, env.wrapper.num_events() - tail_buffer)
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
    from src.env.execution_env import ExecutionEnv

    def make_env():
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
        )

    base_env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    results: list[dict] = []

    for start_idx in start_indices:
        obs = env.reset()
        env.env_method("reset", options={"start_idx": int(start_idx)})

        done = False
        final_info = None
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            final_info = infos[0]

        assert final_info is not None

        results.append(
            {
                "start_idx": int(start_idx),
                "reward": float(total_reward),
                "filled_qty": float(final_info["filled_qty"]),
                "remaining_qty": float(final_info["remaining_qty"]),
            }
        )

    env.close()
    return results


def main() -> None:
    print("[main] started")

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.baselines.passive_best_bid import run_passive_best_bid
    from src.baselines.passive_then_sweep import run_passive_then_sweep
    from src.baselines.twap_market import run_twap_market
    from src.env.execution_env import ExecutionEnv
    from src.utils.io import load_yaml

    print("[main] imports loaded")

    env_cfg_path = "/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/configs/env.yaml"
    env_cfg = load_yaml(str(env_cfg_path))
    print("[main] env config loaded")

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    model_path = "/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints/ppo_execution_agent_vecnorm_new.zip"
    vecnorm_path = "/home/joyce/projects/cross-asset-quant-lab/execution_rl_project/results/checkpoints/vecnormalize_execution.pkl"

    print(f"[main] book_path={book_path}")
    print(f"[main] trade_path={trade_path}")
    print(f"[main] snapshot_path={snapshot_path}")
    print(f"[main] model_path={model_path}")
    print(f"[main] vecnorm_path={vecnorm_path}")

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
    )
    print("[main] ExecutionEnv initialized")

    episodes = 50
    start_indices = sample_start_indices(env, episodes=episodes, seed=42)
    print(f"[main] sampled {len(start_indices)} random start indices")

    print("[main] running baseline: TWAP Market")
    twap_results = run_twap_market(env, episodes=episodes, start_indices=start_indices)
    summarize("TWAP Market", twap_results)

    print("[main] running baseline: Passive Best Bid")
    best_bid_results = run_passive_best_bid(env, episodes=episodes, start_indices=start_indices)
    summarize("Passive Best Bid", best_bid_results)

    print("[main] running baseline: Passive Then Sweep")
    passive_sweep_results = run_passive_then_sweep(env, episodes=episodes, start_indices=start_indices)
    summarize("Passive Then Sweep", passive_sweep_results)

    print("[main] running trained PPO model")
    ppo_results = run_ppo_model(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        start_indices=start_indices,
    )
    summarize("PPO Trained Model", ppo_results)

    print("[main] finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[main] failed: {type(e).__name__}: {e}")
        raise