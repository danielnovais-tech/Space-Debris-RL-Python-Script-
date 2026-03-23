from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from .corruption import CorruptionConfig, ObservationCorruptor
from .safety import SafetyMonitor


logger = logging.getLogger(__name__)


class RobustEnvConfig:
    def __init__(
        self,
        *,
        corruption: CorruptionConfig | None = None,
        reset_on_bad_obs: bool = True,
        fallback_action: int = 0,
    ) -> None:
        self.corruption = corruption or CorruptionConfig()
        self.reset_on_bad_obs = bool(reset_on_bad_obs)
        self.fallback_action = int(fallback_action)


class RobustEnv(gym.Wrapper):
    """Gym wrapper adding input sanity checks and optional corruption injection."""

    def __init__(
        self,
        env: gym.Env,
        *,
        cfg: RobustEnvConfig | None = None,
        safety: SafetyMonitor | None = None,
        seed: int = 0,
    ):
        super().__init__(env)
        self.cfg = cfg or RobustEnvConfig()
        self.safety = safety or SafetyMonitor(action_space_n=getattr(env.action_space, "n", None))
        self.corruptor = ObservationCorruptor(self.cfg.corruption, seed=seed)
        self._last_obs: np.ndarray | None = None
        self.decision_log: list[dict[str, Any]] = []
        self._step_index: int = 0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.safety.reset()
        self._step_index = 0
        obs2 = self.corruptor.corrupt(np.asarray(obs, dtype=np.float32))
        decision = self.safety.validate_observation(obs2)
        if not decision.ok and self.cfg.reset_on_bad_obs:
            obs, info = self.env.reset(**kwargs)
            self.safety.reset()
            obs2 = np.asarray(obs, dtype=np.float32)
        self._last_obs = obs2
        return obs2, info

    def step(self, action):  # type: ignore[override]
        entry: dict[str, Any] = {
            "step": self._step_index,
            "action_received": self._to_jsonable(action),
            "strategy": None,
            "agent_info": None,
            "strategy_vetoed": False,
            "strategy_veto_reason": None,
            "vetoed": False,
            "veto_reason": None,
            "fallback_used": False,
            "watchdog_triggered": False,
            "obs_rejected": False,
            "obs_reject_reason": None,
        }

        action, sanitize_meta = self._sanitize_action(action, strategy=None)
        entry.update(sanitize_meta)

        obs, reward, terminated, truncated, info = self.env.step(action)
        entry["action_executed"] = self._to_jsonable(action)
        obs2 = self.corruptor.corrupt(np.asarray(obs, dtype=np.float32))
        obs_decision = self.safety.validate_observation(obs2)
        if not obs_decision.ok:
            info = dict(info)
            info["safety_obs_rejected"] = True
            info["safety_reason"] = obs_decision.reason
            entry["obs_rejected"] = True
            entry["obs_reject_reason"] = obs_decision.reason
            if self.cfg.reset_on_bad_obs:
                # Watchdog-like recovery: reset environment and continue.
                obs_reset, _info2 = self.env.reset()
                self.safety.reset()
                obs2 = np.asarray(obs_reset, dtype=np.float32)
                terminated = False
                truncated = False
                reward = float(reward)  # keep reward as-is
                entry["watchdog_triggered"] = True

        self._last_obs = obs2
        self.decision_log.append(entry)
        self._step_index += 1
        return obs2, reward, terminated, truncated, info

    def step_with_context(
        self,
        action,
        *,
        strategy: int | None = None,
        agent_info: dict[str, Any] | None = None,
        strategy_veto_reason: str | None = None,
    ):
        """Optional contextual step that carries a high-level strategy for logging.

        Gym APIs only accept the low-level action. This helper lets hierarchical
        callers provide the strategy/agent_info so we can record it alongside
        action-level vetoes.
        """
        # Run sanitize + env.step with strategy-aware LTL.
        entry: dict[str, Any] = {
            "step": self._step_index,
            "action_received": self._to_jsonable(action),
            "strategy": None,
            "agent_info": None,
            "strategy_vetoed": False,
            "strategy_veto_reason": None,
            "vetoed": False,
            "veto_reason": None,
            "fallback_used": False,
            "watchdog_triggered": False,
            "obs_rejected": False,
            "obs_reject_reason": None,
        }

        if strategy is None and agent_info is not None:
            try:
                strategy_val = agent_info.get("strategy")
                if strategy_val is not None:
                    strategy = int(strategy_val)
            except Exception:
                pass

        action_s, sanitize_meta = self._sanitize_action(action, strategy=strategy)
        entry.update(sanitize_meta)

        obs, reward, terminated, truncated, info = self.env.step(action_s)
        entry["action_executed"] = self._to_jsonable(action_s)
        obs2 = self.corruptor.corrupt(np.asarray(obs, dtype=np.float32))
        obs_decision = self.safety.validate_observation(obs2)
        if not obs_decision.ok:
            info = dict(info)
            info["safety_obs_rejected"] = True
            info["safety_reason"] = obs_decision.reason
            entry["obs_rejected"] = True
            entry["obs_reject_reason"] = obs_decision.reason
            if self.cfg.reset_on_bad_obs:
                obs_reset, _info2 = self.env.reset()
                self.safety.reset()
                obs2 = np.asarray(obs_reset, dtype=np.float32)
                terminated = False
                truncated = False
                reward = float(reward)
                entry["watchdog_triggered"] = True

        if strategy is not None:
            info = dict(info)
            info["strategy"] = int(strategy)

        if agent_info is not None:
            info = dict(info)
            info["agent_info"] = dict(agent_info)

        if strategy_veto_reason is not None:
            info = dict(info)
            info["strategy_veto_reason"] = str(strategy_veto_reason)

        self._last_obs = obs2
        self.decision_log.append(entry)
        self._step_index += 1

        if strategy is not None or agent_info is not None or strategy_veto_reason is not None:
            info = dict(info)
            if strategy is not None:
                info["strategy"] = int(strategy)
            if agent_info is not None:
                info["agent_info"] = dict(agent_info)
            if strategy_veto_reason is not None:
                info["strategy_veto_reason"] = str(strategy_veto_reason)

            # backfill into last entry
            if self.decision_log:
                last = self.decision_log[-1]
                if strategy is not None:
                    last["strategy"] = int(strategy)
                if agent_info is not None:
                    last["agent_info"] = self._to_jsonable(agent_info)
                if strategy_veto_reason is not None:
                    last["strategy_vetoed"] = True
                    last["strategy_veto_reason"] = str(strategy_veto_reason)
        return obs, reward, terminated, truncated, info

    def get_decision_log(self) -> list[dict[str, Any]]:
        return list(self.decision_log)

    def clear_decision_log(self) -> None:
        self.decision_log.clear()

    @staticmethod
    def _to_jsonable(x: Any):
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        try:
            arr = np.asarray(x)
            if arr.ndim == 0:
                return int(arr)
            return arr.tolist()
        except Exception:
            return str(x)

    def _sanitize_action(self, action, *, strategy: int | None):
        meta: dict[str, Any] = {}
        # Discrete action
        if hasattr(self.action_space, "n"):
            act = int(action)
            act_decision = self.safety.validate_action(act)
            if not act_decision.ok:
                act = int(self.cfg.fallback_action)
                meta["vetoed"] = True
                meta["veto_reason"] = act_decision.reason
                meta["fallback_used"] = True

            # Optional LTL veto.
            act, _dec = self.safety.veto_action(
                act,
                system_state={"actions": [act], "action_taken": "restart" if act == 1 else "other"},
                strategy=strategy,
                fallback_action=int(self.cfg.fallback_action),
            )
            return int(act), meta

        # MultiDiscrete per-component validation
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            acts = np.asarray(action, dtype=np.int64).reshape(self.action_space.nvec.shape)
            sanitized = acts.copy()
            # Bounds-check each component against its nvec.
            for idx, max_n in np.ndenumerate(self.action_space.nvec):
                a = int(acts[idx])
                if a < 0 or a >= int(max_n):
                    sanitized[idx] = int(self.cfg.fallback_action)
                    meta["vetoed"] = True
                    meta["veto_reason"] = "action_out_of_bounds"
                    meta["fallback_used"] = True

            # Optional LTL veto across nodes.
            flat_actions = np.asarray(sanitized, dtype=np.int64).ravel().tolist()
            sys_state = {
                "actions": sanitized.tolist(),
                "action_taken": "restart" if any(int(a) == 1 for a in flat_actions) else "other",
            }
            ok = self.safety.check_ltl(sys_state, strategy=strategy)
            if not ok.ok:
                sanitized[:] = int(self.cfg.fallback_action)
                meta["vetoed"] = True
                meta["veto_reason"] = f"ltl_veto:{ok.reason}"
                meta["fallback_used"] = True
            return sanitized, meta

        # Unknown action space, pass through.
        return action, meta
