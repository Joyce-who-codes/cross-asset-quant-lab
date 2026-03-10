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
) -> float:
    """
    Buy-side execution reward.

    Components:
    1) execution cost penalty
    2) extra taker penalty
    3) urgency-aware waiting penalty
    """
    qty_scale = max(target_qty, 1e-12)
    fill_ratio = max(0.0, filled_qty) / qty_scale
    remaining_ratio = max(0.0, remaining_qty) / qty_scale

    reward = 0.0

    # 1) execution cost penalty
    if filled_qty > 0 and decision_mid_price > 0:
        exec_cost_bps = (avg_fill_price - decision_mid_price) / decision_mid_price * 10000
        reward -= exec_cost_coef * exec_cost_bps

    # 2) extra taker penalty
    if filled_qty > 0 and is_taker_fill:
        reward -= taker_penalty_coef * fill_ratio

    # 3) urgency-aware waiting penalty
    # early stage: light penalty
    # late stage: stronger penalty
    reward -= lambda_wait * urgency * remaining_ratio

    return reward


def compute_terminal_penalty(
    remaining_qty: float,
    target_qty: float,
    lambda_terminal_remain: float,
) -> float:
    qty_scale = max(target_qty, 1e-12)
    remaining_ratio = max(0.0, remaining_qty) / qty_scale
    return -lambda_terminal_remain * remaining_ratio