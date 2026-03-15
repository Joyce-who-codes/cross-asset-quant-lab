from __future__ import annotations


def run_passive_then_sweep(
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
        threshold = int(env.max_steps * 0.8)
        placed = False
        step_info = None

        while not done:
            if env.current_step < threshold:
                action = 1 if not placed else 0
                placed = True
            else:
                action = 4 if env.wrapper.has_active_order() else 3

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