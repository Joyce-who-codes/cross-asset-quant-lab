from __future__ import annotations


def run_twap_market(env, episodes: int = 10) -> list[dict]:
    results = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = 3  # MARKET_BUY_SMALL
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append(
            {
                "reward": total_reward,
                "filled_qty": step_info["filled_qty"],
                "remaining_qty": step_info["remaining_qty"],
                "equity": step_info["equity"],
            }
        )
    return results