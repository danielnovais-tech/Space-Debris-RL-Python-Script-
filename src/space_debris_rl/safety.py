from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ltl import LTLMonitor


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
        ltl_formulas: list[str] | None = None,
        strategy_space_n: int | None = None,
    ):
        self.obs_min = float(obs_min)
        self.obs_max = float(obs_max)
        self.max_abs_delta = float(max_abs_delta)
        self.action_space_n = action_space_n
        self.strategy_space_n = strategy_space_n

        self.ltl = LTLMonitor(ltl_formulas) if ltl_formulas else None

        self._prev_obs: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_obs = None
        if self.ltl is not None:
            self.ltl.reset()

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

    def validate_strategy(self, strategy: int) -> SafetyDecision:
        try:
            s = int(strategy)
        except Exception:
            return SafetyDecision(False, "strategy_not_int")

        if self.strategy_space_n is not None:
            if s < 0 or s >= int(self.strategy_space_n):
                return SafetyDecision(False, "strategy_out_of_bounds")
        return SafetyDecision(True, None)

    def check_strategy(self, strategy: int, *, system_state: dict) -> SafetyDecision:
        """Strategy-level guard (pre-action) with optional LTL delegation."""
        sdec = self.validate_strategy(strategy)
        if not sdec.ok:
            return sdec

        # Example heuristic veto: don't restart when CPU already low.
        # This mirrors the kind of policy you'd certify separately.
        cpu = float(system_state.get("cpu", 0.0) or 0.0)
        if int(strategy) == 1 and cpu < 20.0:
            return SafetyDecision(False, "strategy_restart_unnecessary_low_cpu")

        # Delegate to LTL monitor if configured.
        ltl_dec = self.check_ltl(system_state)
        if not ltl_dec.ok:
            return SafetyDecision(False, f"strategy_ltl_veto:{ltl_dec.reason}")

        return SafetyDecision(True, None)

    def check_ltl(self, system_state: dict, *, strategy: int | None = None) -> SafetyDecision:
        if self.ltl is None:
            return SafetyDecision(True, None)
        res = self.ltl.check(system_state, strategy=strategy)
        return SafetyDecision(bool(res.ok), res.reason)

    def veto_action(
        self,
        action: int,
        *,
        system_state: dict,
        strategy: int | None = None,
        fallback_action: int = 0,
    ) -> tuple[int, SafetyDecision]:
        """Return (possibly overridden action, decision).

        If an LTL-like constraint is violated, returns fallback_action.
        """
        ltl_dec = self.check_ltl(system_state, strategy=strategy)
        if not ltl_dec.ok:
            return int(fallback_action), SafetyDecision(False, f"ltl_veto:{ltl_dec.reason}")
        return int(action), SafetyDecision(True, None)
