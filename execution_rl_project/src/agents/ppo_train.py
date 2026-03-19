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


def build_env_cfg(symbol: str, side: str) -> dict:
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
            "lambda_terminal_remain": 3.0,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--side", type=str, choices=["buy", "sell"], required=True)
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2", "both"], default="both")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-config", type=str, default="configs/train_btc_long.yaml")
    parser.add_argument("--chunk-root", type=str, default=None)
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

    run_name = f"{symbol.lower()}_{side}_1m"
    model_dir = ensure_dir(str(PROJECT_ROOT / "results" / f"checkpoints_{run_name}"))

    model_path = Path(model_dir) / f"{run_name}.zip"
    vecnorm_path = Path(model_dir) / f"{run_name}_vecnormalize.pkl"

    total_steps = int(train_cfg["total_timesteps"])
    phase1_steps = int(train_cfg.get("phase1_timesteps", 0))
    phase2_steps = max(0, total_steps - phase1_steps)

    print("======================================")
    print("[train] symbol:", symbol)
    print("[train] side:", side)
    print("[train] phase:", phase)
    print("[train] split:", split)
    print("[train] start_day:", start_day)
    print("[train] end_day:", end_day)
    print("[train] chunk_hours:", chunk_hours)
    print("[train] num_chunks:", len(chunk_paths))
    print("[train] save_dir:", model_dir)
    print("======================================")

    if phase in {"phase1", "both"} and phase1_steps > 0:
        print("======================================")
        print("PHASE 1 : fixed chunk warmup")
        print("======================================")

        env_cfg_phase1 = build_env_cfg(symbol=symbol, side=side)
        env_cfg_phase1["execution"]["random_start"] = True
        env_cfg_phase1["execution"]["random_chunk"] = False
        env_cfg_phase1["execution"]["fixed_chunk_index"] = 0

        base_env_1 = make_vec_env(
            env_cfg=env_cfg_phase1,
            chunk_paths=chunk_paths,
        )
        env_1 = VecNormalize(
            base_env_1,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        model_1 = build_model(train_cfg=train_cfg, env=env_1)

        model_1.learn(
            total_timesteps=phase1_steps,
            reset_num_timesteps=True,
        )

        model_1.save(str(model_path))
        env_1.save(str(vecnorm_path))

        print(f"[phase1] saved model to {model_path}")
        print(f"[phase1] saved vecnorm to {vecnorm_path}")

        env_1.close()

        if phase == "phase1":
            return

    if phase in {"phase2", "both"} and phase2_steps > 0:
        print("======================================")
        print("PHASE 2 : random chunk training")
        print("======================================")

        env_cfg_phase2 = build_env_cfg(symbol=symbol, side=side)
        env_cfg_phase2["execution"]["random_start"] = True
        env_cfg_phase2["execution"]["random_chunk"] = True

        base_env_2 = make_vec_env(
            env_cfg=env_cfg_phase2,
            chunk_paths=chunk_paths,
        )

        if model_path.exists() and vecnorm_path.exists():
            print("[phase2] resuming from phase1 checkpoint")
            env_2 = VecNormalize.load(str(vecnorm_path), base_env_2)
            env_2.training = True
            env_2.norm_reward = False
            model_2 = PPO.load(
                str(model_path),
                env=env_2,
                device=train_cfg["device"],
            )
        else:
            print("[phase2] no phase1 checkpoint found, starting from scratch")
            env_2 = VecNormalize(
                base_env_2,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
            model_2 = build_model(train_cfg=train_cfg, env=env_2)

        model_2.learn(
            total_timesteps=phase2_steps,
            reset_num_timesteps=False,
        )

        model_2.save(str(model_path))
        env_2.save(str(vecnorm_path))

        print(f"[phase2] saved model to {model_path}")
        print(f"[phase2] saved vecnorm to {vecnorm_path}")

        env_2.close()


if __name__ == "__main__":
    main()
