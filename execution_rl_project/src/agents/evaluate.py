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

    book_path = "/home/joyce/test.csv"
    trade_path = "/home/joyce/test_trades.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/test_book.csv"

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
    )
    model = PPO.load("results/checkpoints/ppo_execution_agent")

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    action_counter = Counter()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counter[action] += 1

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print("=== Evaluation ===")
    print("total_reward:", total_reward)
    print("filled_qty:", step_info["filled_qty"])
    print("remaining_qty:", step_info["remaining_qty"])
    print("equity:", step_info["equity"])
    print("action counts:")
    for k, v in sorted(action_counter.items()):
        print(f"  {ACTION_NAME[k]}: {v}")


if __name__ == "__main__":
    main()