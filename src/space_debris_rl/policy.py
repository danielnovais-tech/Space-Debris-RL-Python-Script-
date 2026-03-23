from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    reason: str | None = None


class RuleBasedFallbackPolicy:
    """Very small safe-mode policy.

    Action mapping (matches env.Discrete(5)):
    0: no-op
    1: +x thrust
    2: -x thrust
    3: +y thrust
    4: -y thrust

    Heuristic: head toward goal using the dominant axis.
    """

    def __init__(self, *, goal_weight: float = 1.0):
        self.goal_weight = float(goal_weight)

    def predict(self, obs: np.ndarray) -> PolicyDecision:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.size < 6:
            return PolicyDecision(0, "obs_too_small")

        x, y, vx, vy, gx, gy = map(float, obs[:6])
        dx = (gx - x) * self.goal_weight
        dy = (gy - y) * self.goal_weight

        # If we're close enough or already moving well, do nothing.
        if abs(dx) < 0.2 and abs(dy) < 0.2 and abs(vx) < 0.05 and abs(vy) < 0.05:
            return PolicyDecision(0, "near_goal")

        if abs(dx) >= abs(dy):
            return PolicyDecision(1 if dx > 0 else 2, "fallback_toward_goal_x")
        return PolicyDecision(3 if dy > 0 else 4, "fallback_toward_goal_y")
