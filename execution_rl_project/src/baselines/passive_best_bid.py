from __future__ import annotations


def run_passive_best_bid(
    env,
    episodes: int = 10,
    start_indices: list[int] | None = None,
) -> list[dict]:
    results = []

    for ep in range(episodes):
        options = None
        if start_indices is not None:
            options = {"start_idx": int(start_indices[ep])}

        obs, info = env.reset(options=options)
        done = False
        total_reward = 0.0
        first = True
        step_info = None

        while not done:
            action = 1 if first else 0  # PLACE_BID1 once, then HOLD
            first = False

            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        assert step_info is not None

        results.append(
            {
                "start_idx": int(start_indices[ep]) if start_indices is not None else -1,
                "reward": float(total_reward),
                "filled_qty": float(step_info["filled_qty"]),
                "remaining_qty": float(step_info["remaining_qty"]),
                "equity": float(step_info["equity"]),
            }
        )

    return results