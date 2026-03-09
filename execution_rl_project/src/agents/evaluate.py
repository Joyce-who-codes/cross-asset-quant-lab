from stable_baselines3 import PPO

from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")

    book_path = "data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT.csv"
    trade_path = "data/raw/tardis/BTCUSDT/trades/BTCUSDT.csv"

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
    )
    model = PPO.load("results/checkpoints/ppo_execution_agent")

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    print("=== Evaluation ===")
    print("total_reward:", total_reward)
    print("filled_qty:", step_info["filled_qty"])
    print("remaining_qty:", step_info["remaining_qty"])
    print("equity:", step_info["equity"])


if __name__ == "__main__":
    main()