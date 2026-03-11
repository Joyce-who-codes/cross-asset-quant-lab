def summarize(name: str, results: list[dict]) -> None:
    n = len(results)
    avg_reward = sum(x["reward"] for x in results) / n
    avg_fill = sum(x["filled_qty"] for x in results) / n
    avg_remain = sum(x["remaining_qty"] for x in results) / n
    print(f"{name}:")
    print(f"  avg_reward={avg_reward:.6f}")
    print(f"  avg_filled={avg_fill:.6f}")
    print(f"  avg_remaining={avg_remain:.6f}")


def main() -> None:
    from pathlib import Path
    import sys

    print("[main] started")
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.baselines.passive_best_bid import run_passive_best_bid
    from src.baselines.passive_then_sweep import run_passive_then_sweep
    from src.baselines.twap_market import run_twap_market
    from src.env.execution_env import ExecutionEnv
    from src.utils.io import load_yaml

    print("[main] imports loaded")
    env_cfg_path = project_root / "configs" / "env.yaml"
    env_cfg = load_yaml(str(env_cfg_path))
    print("[main] env config loaded")

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    print(f"[main] book_path={book_path}")
    print(f"[main] trade_path={trade_path}")
    print(f"[main] snapshot_path={snapshot_path}")

    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
    )
    print("[main] ExecutionEnv initialized")

    print("[main] running baseline: TWAP Market")
    summarize("TWAP Market", run_twap_market(env, episodes=5))
    print("[main] running baseline: Passive Best Bid")
    summarize("Passive Best Bid", run_passive_best_bid(env, episodes=5))
    print("[main] running baseline: Passive Then Sweep")
    summarize("Passive Then Sweep", run_passive_then_sweep(env, episodes=5))
    print("[main] finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[main] failed: {type(e).__name__}: {e}")
        raise