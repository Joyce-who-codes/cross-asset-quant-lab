from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.env.execution_env import ExecutionEnv
from src.utils.io import ensure_dir, load_yaml
from src.utils.seed import set_seed


def make_env(env_cfg: dict, book_path: str, trade_path: str, snapshot_path: str | None):
    def _fn():
        return ExecutionEnv(
            env_cfg=env_cfg,
            book_path=book_path,
            trade_path=trade_path,
            snapshot_path=snapshot_path,
        )
    return _fn


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")
    train_cfg = load_yaml("configs/train.yaml")
    set_seed(train_cfg["seed"])

    model_dir = ensure_dir(train_cfg["model_dir"])

    book_path = "/home/joyce/test.csv"
    trade_path = "/home/joyce/test_trades.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/test_book.csv"

    vec_env = make_vec_env(
        make_env(env_cfg, book_path, trade_path, snapshot_path),
        n_envs=1,
        seed=train_cfg["seed"],
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg["learning_rate"],
        n_steps=train_cfg["n_steps"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_range=train_cfg["clip_range"],
        ent_coef=train_cfg["ent_coef"],
        verbose=1,
        device=train_cfg["device"],
    )

    model.learn(total_timesteps=train_cfg["total_timesteps"])
    model.save(str(model_dir / "ppo_execution_agent"))


if __name__ == "__main__":
    main()