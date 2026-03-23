from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._deps import require
from .distributed_env import DistributedServiceEnv


def evaluate_hierarchical(
    *,
    manager_path: str | Path,
    worker_path: str | Path,
    episodes: int = 5,
    robust: bool = False,
    obs_bitflip_p: float = 0.0,
    num_nodes: int = 4,
    seed: int = 0,
) -> tuple[list[float], list[dict[str, Any]] | None]:
    """Evaluate a HierarchicalAgent on the distributed service environment.

    Returns (episode_rewards, decision_log_or_none).
    """

    require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
    from stable_baselines3 import PPO  # type: ignore

    base_env = DistributedServiceEnv(num_nodes=int(num_nodes), seed=int(seed))

    env: Any = base_env
    if robust:
        from .corruption import CorruptionConfig
        from .robust_env import RobustEnv, RobustEnvConfig

        env = RobustEnv(
            base_env,
            cfg=RobustEnvConfig(corruption=CorruptionConfig(obs_bitflip_p=float(obs_bitflip_p))),
            seed=int(seed),
        )

    manager = PPO.load(str(manager_path))
    worker = PPO.load(str(worker_path))

    from .hrl import HierarchicalAgent

    agent = HierarchicalAgent(manager=manager, worker=worker)

    episode_totals: list[float] = []
    for ep in range(int(episodes)):
        obs, _info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, agent_info = agent.act(np.asarray(obs, dtype=np.float32), deterministic=True)

            # Gymnasium API: step(action) -> (obs, reward, terminated, truncated, info)
            if hasattr(env, "step_with_context"):
                obs, reward, terminated, truncated, info = env.step_with_context(
                    action,
                    agent_info=agent_info,
                )
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            # Best-effort: carry agent info into info for auditing.
            if isinstance(info, dict):
                info.setdefault("agent_info", agent_info)

            done = bool(terminated or truncated)
            total_reward += float(reward)

        episode_totals.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    decision_log = None
    if hasattr(env, "get_decision_log"):
        try:
            decision_log = list(env.get_decision_log())
        except Exception:
            decision_log = None

    return episode_totals, decision_log
