from __future__ import annotations

import numpy as np

from space_debris_rl.distributed_env import DistributedServiceEnv
from space_debris_rl.envs import StrategyConditionedEnv


def test_strategy_one_hot_can_change_across_resets() -> None:
    base = DistributedServiceEnv(num_nodes=2, max_steps=2, seed=0)
    env = StrategyConditionedEnv(base, num_strategies=3)

    # Simulate the worker-training behavior: sample a strategy each episode.
    tails: list[tuple[float, float, float]] = []
    for s in [0, 1, 2]:
        env.set_strategy(s)
        obs, _info = env.reset()
        tails.append(tuple(np.asarray(obs, dtype=np.float32)[-3:].tolist()))

    assert len(set(tails)) == 3
