from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .corruption import CorruptionConfig, ObservationCorruptor
from .safety import SafetyMonitor


@dataclass
class RobustEnvConfig:
    corruption: CorruptionConfig = CorruptionConfig()
    # If telemetry is rejected, we reset the env (watchdog-style) to a known-good state.
    reset_on_bad_obs: bool = True
    # If action rejected, override with safe fallback action.
    fallback_action: int = 0


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

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self.safety.reset()
        obs2 = self.corruptor.corrupt(np.asarray(obs, dtype=np.float32))
        decision = self.safety.validate_observation(obs2)
        if not decision.ok and self.cfg.reset_on_bad_obs:
            obs, info = self.env.reset(**kwargs)
            self.safety.reset()
            obs2 = np.asarray(obs, dtype=np.float32)
        return obs2, info

    def step(self, action):  # type: ignore[override]
        act_decision = self.safety.validate_action(int(action))
        if not act_decision.ok:
            action = self.cfg.fallback_action

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs2 = self.corruptor.corrupt(np.asarray(obs, dtype=np.float32))
        obs_decision = self.safety.validate_observation(obs2)
        if not obs_decision.ok:
            info = dict(info)
            info["safety_obs_rejected"] = True
            info["safety_reason"] = obs_decision.reason
            if self.cfg.reset_on_bad_obs:
                # Watchdog-like recovery: reset environment and continue.
                obs_reset, _info2 = self.env.reset()
                self.safety.reset()
                obs2 = np.asarray(obs_reset, dtype=np.float32)
                terminated = False
                truncated = False
                reward = float(reward)  # keep reward as-is

        return obs2, reward, terminated, truncated, info
