from __future__ import annotations


def compute_step_reward(
    reference_price: float,
    filled_qty: float,
    avg_fill_price: float,
    remaining_qty: float,
    did_cancel: bool,
    lambda_wait: float,
    lambda_cancel: float,
) -> float:
    reward = 0.0

    if filled_qty > 0:
        exec_cost = (avg_fill_price - reference_price) * filled_qty
        reward -= exec_cost

    reward -= lambda_wait * remaining_qty

    if did_cancel:
        reward -= lambda_cancel

    return reward


def compute_terminal_penalty(
    remaining_qty: float,
    lambda_terminal_remain: float,
) -> float:
    return -lambda_terminal_remain * (remaining_qty ** 2)