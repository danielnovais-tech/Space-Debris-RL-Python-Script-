from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class StrategySpace:
    n: int


class StrategyConditionedObs(gym.ObservationWrapper):
    """Append a one-hot strategy vector to the observation.

    This lets a worker policy be conditioned on the high-level manager's selected strategy
    without requiring multi-input policies.
    """

    def __init__(self, env: gym.Env, *, strategy_n: int):
        super().__init__(env)
        self.strategy_n = int(strategy_n)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("StrategyConditionedObs requires Box observation_space")

        low = np.concatenate(
            [env.observation_space.low.flatten(), np.zeros((self.strategy_n,), dtype=np.float32)]
        ).astype(np.float32)
        high = np.concatenate(
            [
                env.observation_space.high.flatten(),
                np.ones((self.strategy_n,), dtype=np.float32),
            ]
        ).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(int(low.shape[0]),),
            dtype=np.float32,
        )

        self._strategy: int = 0

    def set_strategy(self, strategy: int) -> None:
        s = int(strategy)
        if s < 0 or s >= self.strategy_n:
            s = 0
        self._strategy = s

    def observation(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).flatten()
        one_hot = np.zeros((self.strategy_n,), dtype=np.float32)
        one_hot[self._strategy] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32, copy=False)
