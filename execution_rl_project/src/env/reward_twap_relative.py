from __future__ import annotations


def compute_step_reward(
    arrival_price: float,
    filled_qty: float,
    avg_fill_price: float,
    target_qty: float,
    side: str,
    agent_cum_qty: float,
    benchmark_cum_qty: float,
    taker_fill_qty: float,
    lambda_lag: float,
    lambda_taker: float,
) -> tuple[float, dict]:

    shortfall = 0.0
    shortfall_reward = 0.0

    if filled_qty > 0:
        if side == "buy":
            shortfall = avg_fill_price - arrival_price
        elif side == "sell":
            shortfall = arrival_price - avg_fill_price
        else:
            raise ValueError(f"unsupported side: {side}")

        shortfall_reward = -(shortfall * filled_qty) / max(target_qty, 1e-12)

    lag_qty = max(0.0, benchmark_cum_qty - agent_cum_qty)
    lag_penalty = -lambda_lag * lag_qty / max(target_qty, 1e-12)

    taker_penalty = -lambda_taker * taker_fill_qty / max(target_qty, 1e-12)

    reward = shortfall_reward + lag_penalty + taker_penalty

    parts = {
        "shortfall": float(shortfall),
        "shortfall_reward": float(shortfall_reward),
        "lag_qty": float(lag_qty),
        "lag_penalty": float(lag_penalty),
        "taker_fill_qty": float(taker_fill_qty),
        "taker_penalty": float(taker_penalty),
        "reward": float(reward),
    }

    return float(reward), parts


def compute_terminal_reward(
    remaining_qty: float,
    target_qty: float,
    agent_total_cost: float,
    benchmark_total_cost: float,
    lambda_terminal_remain: float,
    lambda_excess: float,
) -> tuple[float, dict]:

    remaining_ratio = max(0.0, remaining_qty) / max(target_qty, 1e-12)
    remain_penalty = -lambda_terminal_remain * remaining_ratio

    excess_cost = benchmark_total_cost - agent_total_cost
    excess_reward = lambda_excess * excess_cost / max(target_qty, 1e-12)

    reward = remain_penalty + excess_reward

    parts = {
        "remaining_qty": float(remaining_qty),
        "remaining_ratio": float(remaining_ratio),
        "remain_penalty": float(remain_penalty),
        "agent_total_cost": float(agent_total_cost),
        "benchmark_total_cost": float(benchmark_total_cost),
        "excess_cost": float(excess_cost),
        "excess_reward": float(excess_reward),
        "reward": float(reward),
    }

    return float(reward), parts