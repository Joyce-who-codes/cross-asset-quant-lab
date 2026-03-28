from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.execution_env import ExecutionEnv
from src.utils.io import ensure_dir, load_yaml
from src.utils.project_paths import PROJECT_ROOT
from src.utils.seed import set_seed
from src.utils.tardis_chunk import build_chunk_paths


def build_env_cfg(symbol: str, side: str, train_cfg: dict | None = None) -> dict:
    train_cfg = train_cfg or {}
    return {
        "asset": {
            "symbol": symbol,
            "tick_size": 0.1,
            "lot_size": 0.001,
        },
        "backtest": {
            "maker_fee": 0.0,
            "taker_fee": 0.0002,
            "entry_latency_ns": 1000000,
            "response_latency_ns": 1000000,
            "queue_model_power": 2.0,
            "roi_lb": 10000,
            "roi_ub": 200000,
        },
        "execution": {
            "side": side,
            "target_qty": 0.05,
            "horizon_sec": 120,
            "step_sec": 2,
            "market_clip_qty": 0.005,
            "max_active_orders": 1,
            "start_idx": 2000,
            "random_start": False,
        },
        "reward": {
            "lambda_terminal_remain": float(train_cfg.get("lambda_terminal_remain", 3.0)),
            "mode": str(train_cfg.get("reward_mode", "shortfall")),
            "lambda_lag": float(train_cfg.get("lambda_lag", 0.0)),
            "lambda_taker": float(train_cfg.get("lambda_taker", 0.0)),
            "lambda_excess": float(train_cfg.get("lambda_excess", 0.0)),
        },
    }


def make_env(
    env_cfg: dict,
    chunk_paths: list[dict[str, str]],
):
    def _init():
        return ExecutionEnv(
            env_cfg=env_cfg,
            chunk_paths=chunk_paths,
        )

    return _init


def make_vec_env(
    env_cfg: dict,
    chunk_paths: list[dict[str, str]],
) -> DummyVecEnv:
    return DummyVecEnv(
        [
            make_env(
                env_cfg=env_cfg,
                chunk_paths=chunk_paths,
            )
        ]
    )


def build_model(train_cfg: dict, env: VecNormalize) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(train_cfg["learning_rate"]),
        n_steps=int(train_cfg["n_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
        clip_range=float(train_cfg["clip_range"]),
        ent_coef=float(train_cfg["ent_coef"]),
        verbose=1,
        device=train_cfg["device"],
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )


def max_episode_steps(env_cfg: dict) -> int:
    execution_cfg = env_cfg["execution"]
    return int(float(execution_cfg["horizon_sec"]) / float(execution_cfg["step_sec"]))


def timesteps_from_episodes(chunk_paths: list[dict[str, str]], episodes_per_chunk: int, env_cfg: dict) -> int:
    if episodes_per_chunk <= 0:
        return 0
    return len(chunk_paths) * episodes_per_chunk * max_episode_steps(env_cfg)


def load_or_create_model(
    *,
    train_cfg: dict,
    env: VecNormalize,
    model_path: Path,
    device: str,
) -> PPO:
    if model_path.exists():
        return PPO.load(
            str(model_path),
            env=env,
            device=device,
        )
    return build_model(train_cfg=train_cfg, env=env)


def save_checkpoint(model: PPO, env: VecNormalize, model_path: Path, vecnorm_path: Path, stage_name: str) -> None:
    model.save(str(model_path))
    env.save(str(vecnorm_path))
    print(f"[{stage_name}] saved model to {model_path}")
    print(f"[{stage_name}] saved vecnorm to {vecnorm_path}")


def train_stage(
    *,
    stage_name: str,
    stage_label: str,
    train_cfg: dict,
    env_cfg: dict,
    chunk_paths: list[dict[str, str]],
    total_timesteps: int,
    model_path: Path,
    vecnorm_path: Path,
    reset_num_timesteps: bool,
) -> None:
    if total_timesteps <= 0:
        print(f"[{stage_name}] skip: total_timesteps <= 0")
        return

    if not chunk_paths:
        raise ValueError(f"[{stage_name}] no chunk paths available")

    print("======================================")
    print(stage_label)
    print("======================================")
    print(f"[{stage_name}] num_chunks:", len(chunk_paths))
    print(f"[{stage_name}] total_timesteps:", total_timesteps)

    base_env = make_vec_env(
        env_cfg=env_cfg,
        chunk_paths=chunk_paths,
    )

    if model_path.exists() and vecnorm_path.exists():
        print(f"[{stage_name}] resuming from existing checkpoint")
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = True
        env.norm_reward = False
    else:
        print(f"[{stage_name}] starting from scratch")
        env = VecNormalize(
            base_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    model = load_or_create_model(
        train_cfg=train_cfg,
        env=env,
        model_path=model_path,
        device=train_cfg["device"],
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
    )

    save_checkpoint(model, env, model_path, vecnorm_path, stage_name)
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--side", type=str, choices=["buy", "sell"], required=True)
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase1", "phase2a", "phase2b", "phase2", "both"],
        default="both",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-config", type=str, default="configs/train_btc_long.yaml")
    parser.add_argument("--chunk-root", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper()
    side = args.side.lower()
    phase = args.phase
    split = args.split

    train_config_path = Path(args.train_config)
    if not train_config_path.is_absolute():
        train_config_path = PROJECT_ROOT / train_config_path
    train_cfg = load_yaml(str(train_config_path))
    set_seed(train_cfg["seed"])

    split_windows = {
        "train": (train_cfg["train_start_day"], train_cfg["train_end_day"]),
        "val": (train_cfg["val_start_day"], train_cfg["val_end_day"]),
        "test": (train_cfg["test_start_day"], train_cfg["test_end_day"]),
    }
    start_day, end_day = split_windows[split]
    chunk_hours = int(train_cfg.get("chunk_hours", 6))

    chunk_paths = build_chunk_paths(
        symbol=symbol,
        start_day=start_day,
        end_day=end_day,
        chunk_hours=chunk_hours,
        chunk_root=args.chunk_root,
    )

    if args.max_chunks is not None and args.max_chunks > 0:
        chunk_paths = chunk_paths[: args.max_chunks]

    run_name_suffix = str(train_cfg.get("run_name_suffix", ""))
    run_name = f"{symbol.lower()}_{side}_1m{run_name_suffix}"
    model_dir = ensure_dir(str(PROJECT_ROOT / "results" / f"checkpoints_{run_name}"))

    model_path = Path(model_dir) / f"{run_name}.zip"
    vecnorm_path = Path(model_dir) / f"{run_name}_vecnormalize.pkl"

    phase1_steps = int(train_cfg.get("phase1_timesteps", 0))
    phase2a_chunk_count = int(train_cfg.get("phase2a_chunk_count", 4))
    phase2a_episodes_per_chunk = int(train_cfg.get("phase2a_episodes_per_chunk", 20))
    phase2b_episodes_per_chunk = int(train_cfg.get("phase2b_episodes_per_chunk", 20))

    phase2a_chunk_paths = chunk_paths[: max(0, phase2a_chunk_count)]
    phase2b_chunk_paths = chunk_paths

    phase2a_env_cfg = build_env_cfg(symbol=symbol, side=side, train_cfg=train_cfg)
    phase2a_env_cfg["execution"]["random_start"] = True
    phase2a_env_cfg["execution"]["random_chunk"] = True
    phase2a_env_cfg["execution"]["episodes_per_chunk"] = phase2a_episodes_per_chunk
    phase2a_steps = timesteps_from_episodes(
        chunk_paths=phase2a_chunk_paths,
        episodes_per_chunk=phase2a_episodes_per_chunk,
        env_cfg=phase2a_env_cfg,
    )

    phase2b_env_cfg = build_env_cfg(symbol=symbol, side=side, train_cfg=train_cfg)
    phase2b_env_cfg["execution"]["random_start"] = True
    phase2b_env_cfg["execution"]["random_chunk"] = True
    phase2b_env_cfg["execution"]["episodes_per_chunk"] = phase2b_episodes_per_chunk
    phase2b_steps = timesteps_from_episodes(
        chunk_paths=phase2b_chunk_paths,
        episodes_per_chunk=phase2b_episodes_per_chunk,
        env_cfg=phase2b_env_cfg,
    )

    print("======================================")
    print("[train] symbol:", symbol)
    print("[train] side:", side)
    print("[train] phase:", phase)
    print("[train] split:", split)
    print("[train] start_day:", start_day)
    print("[train] end_day:", end_day)
    print("[train] chunk_hours:", chunk_hours)
    print("[train] max_chunks:", args.max_chunks)
    print("[train] num_chunks:", len(chunk_paths))
    print("[train] phase1_timesteps:", phase1_steps)
    print("[train] phase2a_chunk_count:", len(phase2a_chunk_paths))
    print("[train] phase2a_episodes_per_chunk:", phase2a_episodes_per_chunk)
    print("[train] phase2a_timesteps:", phase2a_steps)
    print("[train] phase2b_episodes_per_chunk:", phase2b_episodes_per_chunk)
    print("[train] phase2b_timesteps:", phase2b_steps)
    print("[train] save_dir:", model_dir)
    print("======================================")

    if phase in {"phase1", "both"} and phase1_steps > 0:
        env_cfg_phase1 = build_env_cfg(symbol=symbol, side=side, train_cfg=train_cfg)
        env_cfg_phase1["execution"]["random_start"] = True
        env_cfg_phase1["execution"]["random_chunk"] = False
        env_cfg_phase1["execution"]["fixed_chunk_index"] = 0

        train_stage(
            stage_name="phase1",
            stage_label="PHASE 1 : fixed chunk warmup",
            train_cfg=train_cfg,
            env_cfg=env_cfg_phase1,
            chunk_paths=chunk_paths,
            total_timesteps=phase1_steps,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            reset_num_timesteps=True,
        )

        if phase == "phase1":
            return

    if phase in {"phase2a", "phase2", "both"} and phase2a_steps > 0:
        train_stage(
            stage_name="phase2a",
            stage_label="PHASE 2A : first chunks random training",
            train_cfg=train_cfg,
            env_cfg=phase2a_env_cfg,
            chunk_paths=phase2a_chunk_paths,
            total_timesteps=phase2a_steps,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            reset_num_timesteps=False,
        )

        if phase == "phase2a":
            return

    if phase in {"phase2b", "phase2", "both"} and phase2b_steps > 0:
        train_stage(
            stage_name="phase2b",
            stage_label="PHASE 2B : full chunk pool random training",
            train_cfg=train_cfg,
            env_cfg=phase2b_env_cfg,
            chunk_paths=phase2b_chunk_paths,
            total_timesteps=phase2b_steps,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            reset_num_timesteps=False,
        )


if __name__ == "__main__":
    main()
