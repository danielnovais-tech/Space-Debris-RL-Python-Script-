from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._deps import require
from .distributed_env import DistributedServiceEnv
from .robust_env import RobustEnv, RobustEnvConfig
from .corruption import CorruptionConfig
from .strategy_conditioning import StrategyConditionedObs
from .strategy_worker import StrategyWorker
from .safety import SafetyMonitor


@dataclass(frozen=True)
class HierarchicalConfig:
    num_nodes: int = 4
    max_steps: int = 200
    strategy_n: int = 3
    seed: int = 0
    obs_bitflip_p: float = 0.0
    robust: bool = False


def _make_base_env(cfg: HierarchicalConfig):
    env = DistributedServiceEnv(num_nodes=int(cfg.num_nodes), max_steps=int(cfg.max_steps), seed=int(cfg.seed))
    if cfg.robust:
        env = RobustEnv(
            env,
            cfg=RobustEnvConfig(corruption=CorruptionConfig(obs_bitflip_p=float(cfg.obs_bitflip_p))),
            seed=int(cfg.seed),
        )
    return env


def train_manager(*, total_timesteps: int, cfg: HierarchicalConfig, model_path: str | Path) -> Any:
    """Train a manager that selects a discrete recovery strategy."""
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
    require("gymnasium", extra="rl")

    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    base = _make_base_env(cfg)

    class ManagerEnv(gym.Wrapper):
        def __init__(self, env: gym.Env):
            super().__init__(env)
            self.action_space = spaces.Discrete(int(cfg.strategy_n))

        def step(self, action):  # type: ignore[override]
            # Manager doesn't actuate directly in this minimal demo.
            # Reward shaping: small penalty for choosing "aggressive" strategies.
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            reward = float(reward) - 0.001 * float(int(action))
            info = dict(info)
            info["strategy"] = int(action)
            return obs, reward, terminated, truncated, info

    env = DummyVecEnv([lambda: ManagerEnv(base)])
    model = PPO("MlpPolicy", env, verbose=1, seed=int(cfg.seed))
    model.learn(total_timesteps=int(total_timesteps))
    model_path = Path(model_path)
    model.save(str(model_path))
    return model


def train_worker(*, total_timesteps: int, cfg: HierarchicalConfig, model_path: str | Path) -> Any:
    """Train a worker conditioned on a strategy one-hot appended to observation."""
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    base = _make_base_env(cfg)
    conditioned = StrategyConditionedObs(base, strategy_n=int(cfg.strategy_n))
    conditioned.set_strategy(0)
    env = DummyVecEnv([lambda: conditioned])

    model = PPO("MlpPolicy", env, verbose=1, seed=int(cfg.seed))
    model.learn(total_timesteps=int(total_timesteps))
    model_path = Path(model_path)
    model.save(str(model_path))
    return model


def evaluate_hierarchical(
    *,
    manager_model_path: str | Path,
    worker_model_path: str | Path,
    episodes: int,
    cfg: HierarchicalConfig,
) -> None:
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore

    base = _make_base_env(cfg)
    env = StrategyConditionedObs(base, strategy_n=int(cfg.strategy_n))

    manager = PPO.load(str(manager_model_path))
    worker = PPO.load(str(worker_model_path))

    for ep in range(int(episodes)):
        obs, _info = env.reset()
        done = False
        total = 0.0

        while not done:
            # Manager chooses strategy from *unconditioned* part of observation.
            raw_obs = np.asarray(obs, dtype=np.float32)
            raw_dim = int(raw_obs.shape[0] - cfg.strategy_n)
            manager_obs = raw_obs[:raw_dim]
            strategy, _ = manager.predict(manager_obs, deterministic=True)
            env.set_strategy(int(strategy))

            # Worker chooses concrete action given conditioned obs.
            conditioned_obs = env.observation(manager_obs)
            action, _ = worker.predict(conditioned_obs, deterministic=True)

            obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            total += float(reward)

        print(f"Episode {ep + 1} total reward: {total:.2f}")


def evaluate_strategy_manager_worker(
    *,
    episodes: int,
    cfg: HierarchicalConfig,
    manager_model_path: str | Path | None = None,
    fixed_strategy: int | None = None,
    ltl_formulas: list[str] | None = None,
) -> None:
    """Evaluate hierarchical interface: manager(strategy) -> worker(actions).

    - Manager outputs discrete strategy in [0, strategy_n)
    - Worker deterministically maps strategy to low-level actions
    - SafetyMonitor validates/vetoes both strategy and final actions (incl LTL)
    """
    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")

    from stable_baselines3 import PPO  # type: ignore

    env = _make_base_env(cfg)
    worker = StrategyWorker(num_nodes=int(cfg.num_nodes))
    safety = SafetyMonitor(
        action_space_n=None,
        strategy_space_n=int(cfg.strategy_n),
        ltl_formulas=ltl_formulas,
    )

    manager = None
    if manager_model_path is not None:
        manager = PPO.load(str(manager_model_path))

    for ep in range(int(episodes)):
        obs, _info = env.reset()
        safety.reset()
        done = False
        total = 0.0
        vetoed_strategies = 0
        vetoed_actions = 0

        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)

            # Derive a small system_state summary for strategy checks.
            # For distributed env, treat cpu as avg cpu across nodes.
            cpu = float(np.mean(obs_arr.reshape((cfg.num_nodes, 3))[:, 0])) if obs_arr.size >= cfg.num_nodes * 3 else float(obs_arr.flat[0])
            system_state = {"cpu": cpu}

            if fixed_strategy is not None:
                strategy = int(fixed_strategy)
            elif manager is not None:
                strategy, _ = manager.predict(obs_arr, deterministic=True)
                strategy = int(strategy)
            else:
                strategy = 0

            sdec = safety.check_strategy(strategy, system_state=system_state)
            if not sdec.ok:
                strategy = 0
                vetoed_strategies += 1

            wout = worker.act(obs_arr, strategy=strategy)
            actions = wout.actions

            # LTL veto applies across nodes.
            sys_state = {
                "actions": actions.tolist(),
                "action_taken": "restart" if any(a == 1 for a in actions.tolist()) else "other",
            }
            ltl = safety.check_ltl(sys_state)
            if not ltl.ok:
                actions = np.zeros_like(actions)
                vetoed_actions += 1

            obs, reward, terminated, truncated, _info = env.step(actions)
            done = bool(terminated or truncated)
            total += float(reward)

        print(
            f"Episode {ep + 1} total reward: {total:.2f} | "
            f"strategy_vetoes: {vetoed_strategies} | action_vetoes: {vetoed_actions}"
        )
