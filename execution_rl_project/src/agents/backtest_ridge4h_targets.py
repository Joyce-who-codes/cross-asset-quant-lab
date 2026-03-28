from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.evaluate import build_env_cfg
from src.env.execution_env import ExecutionEnv
from src.utils.io import load_yaml
from src.utils.project_paths import PROJECT_ROOT, resolve_data_root, results_root


DEFAULT_TARGET_CSV = (
    PROJECT_ROOT.parent / "cryptoAlpha" / "data" / "execution_4h" / "btc_execution_targets_4h_20251218_20251228.csv"
)
DEFAULT_BUY_CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints_btcusdt_buy_1m"
DEFAULT_SELL_CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints_btcusdt_sell_1m"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_btc_long.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay 4h Ridge BTC target changes with RL execution agents")
    parser.add_argument("--target-csv", type=str, default=str(DEFAULT_TARGET_CSV))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--signal-prefix", type=str, default="BTC")
    parser.add_argument("--start-day", type=str, default="2025-12-18")
    parser.add_argument("--end-day", type=str, default="2025-12-20")
    parser.add_argument("--chunk-hours", type=int, default=6)
    parser.add_argument("--train-config", type=str, default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--buy-model", type=str, default=str(DEFAULT_BUY_CKPT_DIR / "btcusdt_buy_1m.zip"))
    parser.add_argument(
        "--buy-vecnorm",
        type=str,
        default=str(DEFAULT_BUY_CKPT_DIR / "btcusdt_buy_1m_vecnormalize.pkl"),
    )
    parser.add_argument("--sell-model", type=str, default=str(DEFAULT_SELL_CKPT_DIR / "btcusdt_sell_1m.zip"))
    parser.add_argument(
        "--sell-vecnorm",
        type=str,
        default=str(DEFAULT_SELL_CKPT_DIR / "btcusdt_sell_1m_vecnormalize.pkl"),
    )
    parser.add_argument("--min-start-idx", type=int, default=2000)
    parser.add_argument("--tail-buffer-events", type=int, default=5000)
    parser.add_argument("--chunk-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def load_target_slice(
    target_csv: Path,
    start_day: str,
    end_day: str,
    signal_prefix: str,
) -> pd.DataFrame:
    df = pd.read_csv(target_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    start_ts = pd.Timestamp(start_day, tz="UTC")
    end_ts = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)

    qty_col = "target_qty"
    current_weight_col = f"{signal_prefix} current_weight"
    target_weight_col = f"{signal_prefix} target_weight"

    required_cols = ["datetime", qty_col, current_weight_col, target_weight_col, "delta_weight", "portfolio_value", "price"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"target csv missing columns: {missing}")

    out = df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)].copy()
    out = out[np.abs(out[qty_col]) > 1e-12].copy()
    out["side"] = np.where(out[qty_col] > 0, "buy", "sell")
    out["abs_target_qty"] = out[qty_col].abs().astype(float)
    out = out.reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("no non-zero target_qty rows found in the requested interval")

    return out


def chunk_name_for_timestamp(ts: pd.Timestamp, chunk_hours: int) -> str:
    ts_floor = ts.floor(f"{chunk_hours}h")
    return ts_floor.strftime("%Y-%m-%d_%H")


def load_chunk_meta(symbol: str, chunk_name: str, chunk_root: Path) -> dict:
    meta_path = chunk_root / symbol.upper() / chunk_name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing chunk meta: {meta_path}")
    return json.loads(meta_path.read_text())


def build_chunk_cfg(symbol: str, chunk_name: str, chunk_root: Path) -> dict[str, str]:
    chunk_dir = chunk_root / symbol.upper() / chunk_name
    return {
        "chunk": chunk_name,
        "book_path": str(chunk_dir / "book.parquet"),
        "trade_path": str(chunk_dir / "trades.parquet"),
        "snapshot_path": str(chunk_dir / "snapshot.parquet"),
        "meta_path": str(chunk_dir / "meta.json"),
    }


def chunk_exists(symbol: str, chunk_name: str, chunk_root: Path) -> bool:
    chunk_dir = chunk_root / symbol.upper() / chunk_name
    required = [
        chunk_dir / "book.parquet",
        chunk_dir / "trades.parquet",
        chunk_dir / "snapshot.parquet",
        chunk_dir / "meta.json",
    ]
    return all(path.exists() for path in required)


def estimate_start_idx(
    signal_ts: pd.Timestamp,
    chunk_meta: dict,
    min_start_idx: int,
    tail_buffer_events: int,
) -> tuple[int, float]:
    signal_us = int(signal_ts.value // 1_000)
    start_us = int(chunk_meta["start_us"])
    end_us = int(chunk_meta["end_us"])
    total_events = int(chunk_meta["book_rows"]) + int(chunk_meta["trade_rows"])

    if end_us <= start_us:
        return int(min_start_idx), 0.0

    fraction = (signal_us - start_us) / float(end_us - start_us)
    fraction = float(max(0.0, min(0.999999, fraction)))

    max_valid_idx = max(0, total_events - 1)
    usable_upper = max(min_start_idx, max_valid_idx - max(0, tail_buffer_events))
    start_idx = int(fraction * usable_upper)
    start_idx = max(min_start_idx, min(start_idx, usable_upper))
    return start_idx, fraction


def prepare_env_cfg(
    symbol: str,
    side: str,
    target_qty: float,
    train_cfg: dict,
) -> dict:
    env_cfg = build_env_cfg(symbol=symbol, side=side, train_cfg=train_cfg)
    env_cfg["execution"]["target_qty"] = float(target_qty)
    env_cfg["execution"]["market_clip_qty"] = max(0.001, float(target_qty) * 0.1)
    env_cfg["execution"]["random_start"] = False
    env_cfg["execution"]["random_chunk"] = False
    env_cfg["execution"]["start_idx"] = 2000
    return env_cfg


def run_ppo_episode(
    env_cfg: dict,
    chunk_cfg: dict[str, str],
    model_path: Path,
    vecnorm_path: Path,
    start_idx: int,
) -> dict[str, float | str | int]:
    def make_env() -> ExecutionEnv:
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=chunk_cfg["book_path"],
            trade_path=chunk_cfg["trade_path"],
            snapshot_path=chunk_cfg["snapshot_path"],
        )

    base_env = DummyVecEnv([make_env])
    env_norm = VecNormalize.load(str(vecnorm_path), base_env)
    env_norm.training = False
    env_norm.norm_reward = False
    raw_env = base_env.envs[0]
    model = PPO.load(str(model_path))

    obs_raw, info = raw_env.reset(options={"start_idx": int(start_idx)})
    obs = env_norm.normalize_obs(np.asarray([obs_raw], dtype=np.float32))

    done = False
    total_reward = 0.0
    final_info: dict | None = None
    action_trace: list[str] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_id = int(action[0]) if not isinstance(action, (int, np.integer)) else int(action)
        action_trace.append(str(action_id))
        obs_raw, reward, terminated, truncated, step_info = raw_env.step(action_id)
        obs = env_norm.normalize_obs(np.asarray([obs_raw], dtype=np.float32))
        total_reward += float(reward)
        done = bool(terminated or truncated)
        final_info = step_info

    env_norm.close()

    if final_info is None:
        raise RuntimeError("ppo episode finished without final info")

    return {
        "start_idx": int(start_idx),
        "arrival_price": float(info["arrival_price"]),
        "reward": float(total_reward),
        "filled_qty": float(final_info["filled_qty"]),
        "remaining_qty": float(final_info["remaining_qty"]),
        "equity": float(final_info["equity"]),
        "agent_total_cost": float(final_info.get("agent_total_cost", 0.0)),
        "benchmark_total_cost": float(final_info.get("benchmark_total_cost", 0.0)),
        "excess_cost": float(final_info.get("excess_cost", 0.0)),
        "taker_fill_qty": float(final_info.get("taker_fill_qty", 0.0)),
        "action_trace": " ".join(action_trace),
    }


def run_twap_episode(
    env_cfg: dict,
    chunk_cfg: dict[str, str],
    start_idx: int,
) -> dict[str, float | str | int]:
    env = ExecutionEnv(
        env_cfg=env_cfg,
        book_path=chunk_cfg["book_path"],
        trade_path=chunk_cfg["trade_path"],
        snapshot_path=chunk_cfg["snapshot_path"],
    )

    _, info = env.reset(options={"start_idx": int(start_idx)})
    done = False
    total_reward = 0.0
    final_info: dict | None = None
    action_trace: list[str] = []

    while not done:
        action_id = 3
        action_trace.append(str(action_id))
        _, reward, terminated, truncated, step_info = env.step(action_id)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        final_info = step_info

    if final_info is None:
        raise RuntimeError("twap episode finished without final info")

    return {
        "start_idx": int(start_idx),
        "arrival_price": float(info["arrival_price"]),
        "reward": float(total_reward),
        "filled_qty": float(final_info["filled_qty"]),
        "remaining_qty": float(final_info["remaining_qty"]),
        "equity": float(final_info["equity"]),
        "agent_total_cost": float(final_info.get("agent_total_cost", 0.0)),
        "benchmark_total_cost": float(final_info.get("benchmark_total_cost", 0.0)),
        "excess_cost": float(final_info.get("excess_cost", 0.0)),
        "taker_fill_qty": float(final_info.get("taker_fill_qty", 0.0)),
        "action_trace": " ".join(action_trace),
    }


def main() -> None:
    args = parse_args()
    date_tag = f"{args.start_day.replace('-', '')}_{args.end_day.replace('-', '')}"

    target_csv = Path(args.target_csv)
    if not target_csv.is_absolute():
        target_csv = (PROJECT_ROOT / target_csv).resolve()

    train_config_path = Path(args.train_config)
    if not train_config_path.is_absolute():
        train_config_path = (PROJECT_ROOT / train_config_path).resolve()
    train_cfg = load_yaml(str(train_config_path))

    output_dir = Path(args.output_dir) if args.output_dir else (
        results_root() / "ridge4h_rl_bridge" / date_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_root = Path(args.chunk_root) if args.chunk_root else resolve_data_root("TARDIS_CHUNK_ROOT", "tardis_chunks")
    signals = load_target_slice(
        target_csv=target_csv,
        start_day=args.start_day,
        end_day=args.end_day,
        signal_prefix=args.signal_prefix,
    )
    signal_export_path = output_dir / f"btc_ridge4h_signal_slice_{date_tag}.csv"
    signals.to_csv(signal_export_path, index=False)

    results: list[dict[str, object]] = []
    missing_signals: list[dict[str, object]] = []
    current_weight_col = f"{args.signal_prefix} current_weight"
    target_weight_col = f"{args.signal_prefix} target_weight"

    for row in signals.to_dict(orient="records"):
        signal_ts = pd.Timestamp(row["datetime"])
        chunk_name = chunk_name_for_timestamp(signal_ts, chunk_hours=args.chunk_hours)
        if not chunk_exists(args.symbol, chunk_name, chunk_root=chunk_root):
            missing_signals.append(
                {
                    "signal_datetime": signal_ts.isoformat(),
                    "chunk": chunk_name,
                    "side": str(row["side"]),
                    "signal_target_qty": float(row["target_qty"]),
                    "abs_target_qty": float(row["abs_target_qty"]),
                    "reason": "missing_chunk_data",
                }
            )
            continue

        chunk_meta = load_chunk_meta(args.symbol, chunk_name, chunk_root=chunk_root)
        chunk_cfg = build_chunk_cfg(args.symbol, chunk_name, chunk_root=chunk_root)
        start_idx, chunk_fraction = estimate_start_idx(
            signal_ts=signal_ts,
            chunk_meta=chunk_meta,
            min_start_idx=args.min_start_idx,
            tail_buffer_events=args.tail_buffer_events,
        )

        env_cfg = prepare_env_cfg(
            symbol=args.symbol,
            side=str(row["side"]),
            target_qty=float(row["abs_target_qty"]),
            train_cfg=train_cfg,
        )

        if str(row["side"]) == "buy":
            model_path = Path(args.buy_model)
            vecnorm_path = Path(args.buy_vecnorm)
        else:
            model_path = Path(args.sell_model)
            vecnorm_path = Path(args.sell_vecnorm)

        ppo_result = run_ppo_episode(
            env_cfg=env_cfg,
            chunk_cfg=chunk_cfg,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            start_idx=start_idx,
        )
        twap_result = run_twap_episode(
            env_cfg=env_cfg,
            chunk_cfg=chunk_cfg,
            start_idx=start_idx,
        )

        result_row = {
            "signal_datetime": signal_ts.isoformat(),
            "chunk": chunk_name,
            "chunk_fraction": chunk_fraction,
            "start_idx": int(start_idx),
            "side": str(row["side"]),
            "signal_target_qty": float(row["target_qty"]),
            "abs_target_qty": float(row["abs_target_qty"]),
            "current_weight": float(row[current_weight_col]),
            "target_weight": float(row[target_weight_col]),
            "delta_weight": float(row["delta_weight"]),
            "portfolio_value": float(row["portfolio_value"]),
            "signal_price": float(row["price"]),
            "ppo_reward": float(ppo_result["reward"]),
            "ppo_arrival_price": float(ppo_result["arrival_price"]),
            "ppo_filled_qty": float(ppo_result["filled_qty"]),
            "ppo_remaining_qty": float(ppo_result["remaining_qty"]),
            "ppo_fill_ratio": float(ppo_result["filled_qty"]) / max(float(row["abs_target_qty"]), 1e-12),
            "ppo_equity": float(ppo_result["equity"]),
            "ppo_agent_total_cost": float(ppo_result["agent_total_cost"]),
            "ppo_benchmark_total_cost": float(ppo_result["benchmark_total_cost"]),
            "ppo_excess_cost": float(ppo_result["excess_cost"]),
            "ppo_taker_fill_qty": float(ppo_result["taker_fill_qty"]),
            "ppo_action_trace": str(ppo_result["action_trace"]),
            "twap_reward": float(twap_result["reward"]),
            "twap_arrival_price": float(twap_result["arrival_price"]),
            "twap_filled_qty": float(twap_result["filled_qty"]),
            "twap_remaining_qty": float(twap_result["remaining_qty"]),
            "twap_fill_ratio": float(twap_result["filled_qty"]) / max(float(row["abs_target_qty"]), 1e-12),
            "twap_equity": float(twap_result["equity"]),
            "twap_agent_total_cost": float(twap_result["agent_total_cost"]),
            "twap_benchmark_total_cost": float(twap_result["benchmark_total_cost"]),
            "twap_excess_cost": float(twap_result["excess_cost"]),
            "twap_taker_fill_qty": float(twap_result["taker_fill_qty"]),
            "twap_action_trace": str(twap_result["action_trace"]),
        }
        results.append(result_row)

    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"btc_ridge4h_rl_replay_{date_tag}.csv"
    results_df.to_csv(csv_path, index=False)
    missing_path = output_dir / f"btc_ridge4h_rl_replay_{date_tag}_missing.csv"
    pd.DataFrame(missing_signals).to_csv(missing_path, index=False)

    summary = {
        "num_signals_requested": int(len(signals)),
        "num_signals_executed": int(len(results_df)),
        "num_signals_missing": int(len(missing_signals)),
        "start_day": args.start_day,
        "end_day": args.end_day,
        "signal_slice_csv": str(signal_export_path),
        "target_csv": str(target_csv),
        "output_csv": str(csv_path),
        "missing_csv": str(missing_path),
        "ppo_total_reward": float(results_df["ppo_reward"].sum()) if len(results_df) else 0.0,
        "twap_total_reward": float(results_df["twap_reward"].sum()) if len(results_df) else 0.0,
        "ppo_total_excess_cost": float(results_df["ppo_excess_cost"].sum()) if len(results_df) else 0.0,
        "twap_total_excess_cost": float(results_df["twap_excess_cost"].sum()) if len(results_df) else 0.0,
        "ppo_total_filled_qty": float(results_df["ppo_filled_qty"].sum()) if len(results_df) else 0.0,
        "twap_total_filled_qty": float(results_df["twap_filled_qty"].sum()) if len(results_df) else 0.0,
        "buy_signal_count": int((signals["side"] == "buy").sum()),
        "sell_signal_count": int((signals["side"] == "sell").sum()),
        "executed_buy_count": int((results_df["side"] == "buy").sum()) if len(results_df) else 0,
        "executed_sell_count": int((results_df["side"] == "sell").sum()) if len(results_df) else 0,
    }
    summary_path = output_dir / f"btc_ridge4h_rl_replay_{date_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved trade-level replay to {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
