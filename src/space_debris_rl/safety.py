from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyDecision:
    ok: bool
    reason: str | None = None


class SafetyMonitor:
    """Runtime safety guard for telemetry + actions.

    This is intentionally lightweight and deterministic so it can be
    validated/tested separately from the RL policy.
    """

    def __init__(
        self,
        *,
        obs_min: float = -1e6,
        obs_max: float = 1e6,
        max_abs_delta: float = 50.0,
        action_space_n: int | None = None,
    ):
        self.obs_min = float(obs_min)
        self.obs_max = float(obs_max)
        self.max_abs_delta = float(max_abs_delta)
        self.action_space_n = action_space_n

        self._prev_obs: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_obs = None

    def validate_observation(self, obs: np.ndarray) -> SafetyDecision:
        if not isinstance(obs, np.ndarray):
            return SafetyDecision(False, "obs_not_numpy")

        if obs.size == 0:
            return SafetyDecision(False, "obs_empty")

        if not np.isfinite(obs).all():
            return SafetyDecision(False, "obs_non_finite")

        if (obs < self.obs_min).any() or (obs > self.obs_max).any():
            return SafetyDecision(False, "obs_out_of_range")

        if self._prev_obs is not None:
            if obs.shape != self._prev_obs.shape:
                return SafetyDecision(False, "obs_shape_changed")
            # Simple temporal consistency check: large instant jumps likely indicate corruption.
            if np.max(np.abs(obs - self._prev_obs)) > self.max_abs_delta:
                return SafetyDecision(False, "obs_temporal_inconsistent")

        self._prev_obs = obs.astype(np.float32, copy=True)
        return SafetyDecision(True, None)

    def validate_action(self, action: int) -> SafetyDecision:
        try:
            action_i = int(action)
        except Exception:
            return SafetyDecision(False, "action_not_int")

        if self.action_space_n is not None:
            if action_i < 0 or action_i >= int(self.action_space_n):
                return SafetyDecision(False, "action_out_of_bounds")

        return SafetyDecision(True, None)
