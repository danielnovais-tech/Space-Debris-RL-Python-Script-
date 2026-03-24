from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._deps import require
import numpy as np


@dataclass(frozen=True)
class TrainHierarchicalArgs:
    nodes: int = 4
    strategies: int = 5
    timesteps: int = 50_000
    seed: int = 0
    robust: bool = False
    obs_bitflip_p: float = 0.001
    ltl: Sequence[str] = ()
    manager_model: str = "manager_ppo"
    train_worker: bool = False
    worker_model: str = "worker_ppo"


def _ensure_zip_suffix(path: str | Path) -> str:
    p = str(path)
    return p if p.endswith(".zip") else (p + ".zip")


def train_hierarchical(args: TrainHierarchicalArgs) -> None:
    """Set up distributed env (+ optional RobustEnv) and train manager/worker.

    This is a small orchestration helper intended for CLI use.
    Training logic lives in `space_debris_rl.hierarchical_rl`.
    """

    env = _make_distributed_env(args)

    manager_path = _ensure_zip_suffix(args.manager_model)
    train_manager(
        env,
        total_timesteps=int(args.timesteps),
        model_path=manager_path,
        strategies=int(args.strategies),
        ltl_formulas=list(args.ltl) if args.ltl else None,
        seed=int(args.seed),
    )

    if bool(args.train_worker):
        worker_path = _ensure_zip_suffix(args.worker_model)
        train_worker(
            env,
            total_timesteps=int(args.timesteps),
            model_path=worker_path,
            manager_path=manager_path,
            strategies=int(args.strategies),
            num_strategies=int(args.strategies),
            seed=int(args.seed),
        )

    print("Hierarchical training completed.")


def _make_distributed_env(args: TrainHierarchicalArgs):
    from .distributed_env import DistributedServiceEnv

    base_env = DistributedServiceEnv(num_nodes=int(args.nodes), max_steps=200, seed=int(args.seed))
    if not args.robust:
        return base_env

    from .corruption import CorruptionConfig
    from .robust_env import RobustEnv, RobustEnvConfig
    from .safety import SafetyMonitor

    safety = SafetyMonitor(
        ltl_formulas=list(args.ltl) if args.ltl else None,
        strategy_space_n=int(args.strategies),
    )
    return RobustEnv(
        base_env,
        cfg=RobustEnvConfig(corruption=CorruptionConfig(obs_bitflip_p=float(args.obs_bitflip_p))),
        safety=safety,
        seed=int(args.seed),
    )


def train_manager(
    env,
    *,
    total_timesteps: int = 100_000,
    model_path: str = "manager_ppo.zip",
    strategies: int = 5,
    ltl_formulas: list[str] | None = None,
    seed: int = 0,
):
    """Train the high-level manager policy.

    The manager action space is a single Discrete of size `strategies`.
    """
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
    require("gymnasium", extra="rl")

    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    class ManagerEnv(gym.Wrapper):
        def __init__(self, env: gym.Env):
            super().__init__(env)
            self.action_space = spaces.Discrete(int(strategies))

        def step(self, action):  # type: ignore[override]
            # Manager doesn't actuate directly; step underlying env with a sampled low-level action.
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            info = dict(info)
            info["strategy"] = int(action)
            return obs, float(reward), terminated, truncated, info

    vec_env = DummyVecEnv([lambda: ManagerEnv(env)])
    manager = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.0,
    )
    manager.learn(total_timesteps=int(total_timesteps))
    manager.save(str(model_path))
    print(f"Manager saved to {model_path}")
    return manager


def train_worker(
    env,
    *,
    total_timesteps: int = 100_000,
    model_path: str = "worker_ppo.zip",
    manager_path: str | None = None,
    strategies: int = 5,
    num_strategies: int | None = None,
    seed: int = 0,
):
    """Train the low-level worker policy conditioned on strategy.

    Observation = original state + strategy one-hot (size=num_strategies).
    """
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    from .envs import StrategyConditionedEnv

    _manager = PPO.load(str(manager_path)) if manager_path else None

    n_strat = int(num_strategies) if num_strategies is not None else int(strategies)

    # Vary strategies across episodes so the worker learns to act under all strategies.
    def make_env():
        wrapped = StrategyConditionedEnv(env, num_strategies=int(n_strat))

        class _RandomizeStrategyOnReset(type(wrapped)):
            def reset(self, **kwargs):  # type: ignore[override]
                s = int(np.random.randint(0, int(n_strat)))
                self.set_strategy(s)
                return super().reset(**kwargs)

        wrapped.__class__ = _RandomizeStrategyOnReset  # type: ignore[misc]
        wrapped.set_strategy(0)
        return wrapped

    vec_env = DummyVecEnv([make_env])

    # Stub: train a single worker on a fixed strategy. Later we can add curricula
    # or a manager-in-the-loop roll-in.
    worker = PPO("MlpPolicy", vec_env, verbose=1, seed=int(seed))
    worker.learn(total_timesteps=int(total_timesteps))
    worker.save(str(model_path))
    print(f"Worker saved to {model_path}")
    return worker
