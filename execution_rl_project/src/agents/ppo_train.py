from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.execution_env import ExecutionEnv
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_seed


def make_env(
    env_cfg: dict,
    book_path: str,
    trade_path: str,
    snapshot_path: str | None,
):
    def _init():
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
        )
    return _init


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train.yaml")
    set_seed(train_cfg["seed"])

    model_dir = ensure_dir(train_cfg["model_dir"])

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    base_env = DummyVecEnv(
        [
            make_env(
                env_cfg=env_cfg,
                book_path=book_path,
                trade_path=trade_path,
                snapshot_path=snapshot_path,
            )
        ]
    )

    model_path = model_dir / "ppo_execution_agent_vecnorm"
    vecnorm_path = model_dir / "vecnormalize_execution.pkl"

    resume = bool(train_cfg.get("resume", False))

    if resume and model_path.with_suffix(".zip").exists() and vecnorm_path.exists():
        print("[train] resuming from checkpoint")
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = True
        env.norm_reward = False

        model = PPO.load(str(model_path.with_suffix(".zip")), env=env, device=train_cfg["device"])
    else:
        print("[train] starting new training")
        env = VecNormalize(
            base_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            verbose=1,
            device=train_cfg["device"],
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
        )

    print(f"[train] book_path={book_path}")
    print(f"[train] trade_path={trade_path}")
    print(f"[train] snapshot_path={snapshot_path}")
    print(f"[train] device={train_cfg['device']}")
    print(f"[train] resume={resume}")
    print(f"[train] random_start={env_cfg['execution'].get('random_start', False)}")

    model.learn(total_timesteps=train_cfg["total_timesteps"], reset_num_timesteps=not resume)
    model_path_new = model_dir / "ppo_execution_agent_vecnorm_new"
    model.save(str(model_path_new))
    env.save(str(vecnorm_path))

    print(f"[train] saved model to {model_path_new}.zip")
    print(f"[train] saved vecnormalize stats to {vecnorm_path}")

    env.close()


if __name__ == "__main__":
    main()