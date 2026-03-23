from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkerOutput:
    actions: np.ndarray
    reason: str | None = None


class StrategyWorker:
    """Maps a high-level discrete strategy to low-level per-node actions.

    Strategy mapping (example):
      0 = noop
      1 = restart one node (worst error)
      2 = scale up one node (highest CPU)
      3 = clear cache one node (highest mem)

    Output is a vector of length num_nodes with values in {0,1,2,3}.
    """

    def __init__(self, *, num_nodes: int):
        self.num_nodes = int(num_nodes)

    def act(self, obs_flat: np.ndarray, *, strategy: int) -> WorkerOutput:
        obs = np.asarray(obs_flat, dtype=np.float32).reshape((self.num_nodes, 3))
        actions = np.zeros((self.num_nodes,), dtype=np.int64)

        s = int(strategy)
        if s == 0:
            return WorkerOutput(actions, "strategy_noop")

        if s == 1:
            idx = int(np.argmax(obs[:, 1]))  # highest error_rate
            actions[idx] = 1
            return WorkerOutput(actions, "strategy_restart")

        if s == 2:
            idx = int(np.argmax(obs[:, 0]))  # highest cpu
            actions[idx] = 2
            return WorkerOutput(actions, "strategy_scale_up")

        if s == 3:
            idx = int(np.argmax(obs[:, 2]))  # highest mem
            actions[idx] = 3
            return WorkerOutput(actions, "strategy_clear_cache")

        # Unknown strategy => noop
        return WorkerOutput(actions, "strategy_unknown")
