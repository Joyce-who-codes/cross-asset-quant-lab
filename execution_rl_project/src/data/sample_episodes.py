from dataclasses import dataclass
import random


@dataclass
class EpisodeSpec:
    start_idx: int
    target_qty: float
    horizon_sec: int
    side: str


def sample_episode(
    max_start_idx: int,
    target_qty: float,
    horizon_sec: int,
    side: str = "buy",
) -> EpisodeSpec:
    start_idx = random.randint(0, max(0, max_start_idx - 1))
    return EpisodeSpec(
        start_idx=start_idx,
        target_qty=target_qty,
        horizon_sec=horizon_sec,
        side=side,
    )