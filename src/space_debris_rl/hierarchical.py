from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def predict(self, obs: np.ndarray) -> int: ...


@dataclass(frozen=True)
class HierarchicalDecision:
    strategy: int
    action: object
    reason: str | None = None


class HierarchicalController:
    """Manager/worker controller interface.

    This is an integration seam: you can keep the manager small and verifiable
    (discrete strategies) while letting workers handle time-critical details.

    For now this is policy-agnostic and can wrap either learned or rule-based components.
    """

    def __init__(self, *, manager: Policy, workers: dict[int, Policy], default_strategy: int = 0):
        self.manager = manager
        self.workers = dict(workers)
        self.default_strategy = int(default_strategy)

    def act(self, obs: np.ndarray) -> HierarchicalDecision:
        try:
            strategy = int(self.manager.predict(obs))
        except Exception:
            strategy = self.default_strategy

        worker = self.workers.get(strategy)
        if worker is None:
            worker = self.workers.get(self.default_strategy)
            strategy = self.default_strategy

        action = worker.predict(obs) if worker is not None else 0
        return HierarchicalDecision(strategy=strategy, action=action, reason=None)


class ConstantManager:
    def __init__(self, strategy: int = 0):
        self.strategy = int(strategy)

    def predict(self, obs: np.ndarray) -> int:  # noqa: ARG002
        return self.strategy
