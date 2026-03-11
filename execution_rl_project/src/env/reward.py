from __future__ import annotations


def compute_step_reward(
    decision_mid_price: float,
    filled_qty: float,
    avg_fill_price: float,
    remaining_qty: float,
    target_qty: float,
    urgency: float,
    lambda_wait: float,
    exec_cost_coef: float,
    taker_penalty_coef: float,
    is_taker_fill: bool,
    alpha_signal: float,
) -> tuple[float, dict]:
    qty_scale = max(target_qty, 1e-12)
    fill_ratio = max(0.0, filled_qty) / qty_scale
    remaining_ratio = max(0.0, remaining_qty) / qty_scale

    exec_reward = 0.0
    taker_penalty = 0.0
    wait_penalty = 0.0

    # 1) execution reward in bps
    if filled_qty > 0 and decision_mid_price > 0:
        exec_cost_bps = (
            (avg_fill_price - decision_mid_price) / decision_mid_price * 10000.0
        )
        exec_reward = -exec_cost_coef * exec_cost_bps * fill_ratio

    # 2) extra taker penalty
    if filled_qty > 0 and is_taker_fill:
        taker_penalty = -taker_penalty_coef * fill_ratio

    # 3) alpha-aware urgency waiting penalty
    # buy-side intuition:
    # alpha > 0 => short-term up move more likely => waiting is more dangerous
    # alpha < 0 => short-term down move more likely => waiting is less dangerous
    alpha_clipped = max(-1.0, min(1.0, alpha_signal))
    alpha_wait_multiplier = 1.0 + 0.5 * alpha_clipped
    wait_penalty = -lambda_wait * urgency * remaining_ratio * alpha_wait_multiplier

    total_reward = exec_reward + taker_penalty + wait_penalty

    parts = {
        "exec_reward": float(exec_reward),
        "taker_penalty": float(taker_penalty),
        "wait_penalty": float(wait_penalty),
        "alpha_wait_multiplier": float(alpha_wait_multiplier),
    }
    return float(total_reward), parts


def compute_terminal_penalty(
    remaining_qty: float,
    target_qty: float,
    lambda_terminal_remain: float,
) -> float:
    qty_scale = max(target_qty, 1e-12)
    remaining_ratio = max(0.0, remaining_qty) / qty_scale
    return -lambda_terminal_remain * remaining_ratio