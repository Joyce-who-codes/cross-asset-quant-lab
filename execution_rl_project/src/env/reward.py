from __future__ import annotations


def compute_step_reward(
    arrival_price: float,
    filled_qty: float,
    avg_fill_price: float,
    target_qty: float,
    side: str,
) -> tuple[float, dict]:

    reward = 0.0
    shortfall = 0.0

    if filled_qty > 0:
        if side == "buy":
            shortfall = avg_fill_price - arrival_price
        elif side == "sell":
            shortfall = arrival_price - avg_fill_price
        else:
            raise ValueError(f"unsupported side: {side}")

        reward = -(shortfall * filled_qty) / max(target_qty, 1e-12)

    parts = {
        "shortfall": float(shortfall),
        "shortfall_reward": float(reward),
    }

    return float(reward), parts


def compute_terminal_penalty(
    remaining_qty: float,
    target_qty: float,
    lambda_terminal_remain: float,
) -> float:

    remaining_ratio = max(0.0, remaining_qty) / max(target_qty, 1e-12)
    return -lambda_terminal_remain * remaining_ratio