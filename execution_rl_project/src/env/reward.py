from __future__ import annotations


def compute_step_reward(
    decision_mid_price: float,
    filled_qty: float,
    avg_fill_price: float,
    remaining_qty: float,
    target_qty: float,
    lambda_wait: float,
) -> float:
    reward = 0.0

    qty_scale = max(target_qty, 1e-12)
    fill_ratio = max(0.0, filled_qty) / qty_scale
    remaining_ratio = max(0.0, remaining_qty) / qty_scale

    # 只在成交时按“这一步买得贵不贵”打分
    if filled_qty > 0 and decision_mid_price > 0:
        exec_cost_ratio = (avg_fill_price - decision_mid_price) / decision_mid_price
        reward -= exec_cost_ratio * fill_ratio

    # 轻微等待惩罚
    reward -= lambda_wait * remaining_ratio

    return reward


def compute_terminal_penalty(
    remaining_qty: float,
    target_qty: float,
    lambda_terminal_remain: float,
) -> float:
    qty_scale = max(target_qty, 1e-12)
    remaining_ratio = max(0.0, remaining_qty) / qty_scale
    return -lambda_terminal_remain * remaining_ratio