from __future__ import annotations


def run_passive_then_sweep(env, episodes: int = 10) -> list[dict]:
    results = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        threshold = int(env.max_steps * 0.8)
        placed = False

        while not done:
            if env.current_step < threshold:
                action = 1 if not placed else 0
                placed = True
            else:
                action = 4 if env.wrapper.has_active_order() else 3

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