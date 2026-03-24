import json

import numpy as np

from space_debris_rl.env import SpaceDebrisAvoidanceEnv
from space_debris_rl.robust_env import RobustEnv


def test_decision_log_includes_strategy_from_agent_info_and_is_jsonable() -> None:
    env = RobustEnv(SpaceDebrisAvoidanceEnv(), seed=0)
    obs, _info = env.reset()

    _obs2, _r, _terminated, _truncated, _info2 = env.step_with_context(
        0,
        agent_info={"strategy": 2, "worker_used": True, "other": "x"},
    )

    log = env.get_decision_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["strategy"] == 2
    assert entry["worker_used"] is True
    assert entry["agent_info"] is not None

    # Ensure JSON export can succeed.
    json.dumps(log)
