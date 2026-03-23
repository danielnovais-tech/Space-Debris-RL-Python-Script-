from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class HierarchicalDecision:
    strategy: int
    strategy_name: str
    action: Any
    action_name: str


class HierarchicalAgent:
    """Manager selects a high-level strategy; worker translates to low-level actions.

    - Manager: typically a Stable-Baselines3 model with `.predict(obs)` returning a discrete strategy.
    - Worker: optional learned policy. If absent, we use deterministic mapping.

    This agent supports two kinds of low-level actions:
    - scalar int actions (Discrete environments, e.g. `SpaceDebrisAvoidanceEnv`)
    - per-node action vectors (MultiDiscrete environments, e.g. `DistributedServiceEnv`)
    """

    def __init__(
        self,
        *,
        manager: Optional[Any] = None,
        worker: Optional[Any] = None,
        strategy_map: Optional[dict[int, Any]] = None,
        num_nodes: int | None = None,
    ):
        self.manager = manager
        self.worker = worker
        self.num_nodes = int(num_nodes) if num_nodes is not None else None

        if strategy_map is not None:
            self.strategy_map = dict(strategy_map)
        else:
            if self.num_nodes is not None:
                self.strategy_map = {
                    0: self._noop_vector(self.num_nodes),
                    1: self._single_node_action(self.num_nodes, action=1),
                    2: self._single_node_action(self.num_nodes, action=2),
                    3: self._single_node_action(self.num_nodes, action=3),
                }
            else:
                self.strategy_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    def act(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, dict[str, Any]]:
        obs = np.asarray(observation, dtype=np.float32)

        if self.manager is not None:
            strategy, _ = self.manager.predict(obs, deterministic=deterministic)
            strategy = int(strategy)
        else:
            if obs.size >= 1 and float(obs.flat[0]) > 90:
                strategy = 2
            elif obs.size >= 1 and float(obs.flat[0]) > 70:
                strategy = 1
            else:
                strategy = 0

        if self.worker is not None:
            action, _ = self.worker.predict(obs, deterministic=deterministic)
        else:
            action = self.strategy_map.get(strategy, self.strategy_map.get(0, 0))

        decision = HierarchicalDecision(
            strategy=strategy,
            strategy_name=self._strategy_name(strategy),
            action=action,
            action_name=self._action_name(action),
        )
        info = {
            "strategy": decision.strategy,
            "strategy_name": decision.strategy_name,
            "action": decision.action,
            "action_name": decision.action_name,
        }
        return decision.action, info

    def _strategy_name(self, strategy: int) -> str:
        names = {
            0: "noop",
            1: "restart",
            2: "scale_up",
            3: "clear_cache",
            4: "reset_node",
        }
        return names.get(int(strategy), "unknown")

    def _action_name(self, action: Any) -> str:
        if isinstance(action, (int, np.integer)):
            names = {
                0: "noop",
                1: "restart_service",
                2: "scale_up",
                3: "clear_cache",
                4: "reset_node",
            }
            return names.get(int(action), "unknown")

        try:
            arr = np.asarray(action)
            if arr.ndim == 1:
                return "per_node_actions"
        except Exception:
            pass

        return "unknown"

    @staticmethod
    def _noop_vector(num_nodes: int) -> np.ndarray:
        return np.zeros((int(num_nodes),), dtype=np.int64)

    @staticmethod
    def _single_node_action(num_nodes: int, *, action: int) -> np.ndarray:
        acts = np.zeros((int(num_nodes),), dtype=np.int64)
        acts[0] = int(action)
        return acts
