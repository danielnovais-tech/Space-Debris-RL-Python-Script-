import numpy as np

from space_debris_rl.safety import SafetyMonitor


def test_safety_monitor_rejects_non_finite_obs():
    sm = SafetyMonitor(action_space_n=5)
    obs = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    decision = sm.validate_observation(obs)
    assert decision.ok is False
    assert decision.reason == "obs_non_finite"


def test_safety_monitor_temporal_inconsistency():
    sm = SafetyMonitor(max_abs_delta=1.0)
    obs1 = np.zeros((10,), dtype=np.float32)
    assert sm.validate_observation(obs1).ok

    obs2 = np.zeros((10,), dtype=np.float32)
    obs2[0] = 10.0
    d = sm.validate_observation(obs2)
    assert d.ok is False
    assert d.reason == "obs_temporal_inconsistent"
