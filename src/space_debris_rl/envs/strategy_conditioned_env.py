from __future__ import annotations

from typing import Any

import gymnasium as gym

from ..strategy_conditioning import StrategyConditionedObs


class StrategyConditionedEnv(gym.Wrapper):
    """Augment observations with a one-hot encoding of the current strategy.

    This wrapper exists for worker-policy training: the worker receives an observation
    conditioned on the manager's chosen strategy.

    Notes:
    - This is a thin compatibility shim around `StrategyConditionedObs`.
    - The strategy must be set externally via `set_strategy()`.
    """

    def __init__(self, env: gym.Env, num_strategies: int):
        conditioned = StrategyConditionedObs(env, strategy_n=int(num_strategies))
        super().__init__(conditioned)
        self.num_strategies = int(num_strategies)

    def set_strategy(self, strategy: int) -> None:
        # Delegate to underlying StrategyConditionedObs
        casted = int(strategy)
        self.env.set_strategy(casted)  # type: ignore[attr-defined]

    def reset(self, **kwargs: Any):  # type: ignore[override]
        return self.env.reset(**kwargs)

    def step(self, action: Any):  # type: ignore[override]
        return self.env.step(action)
