from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from space_debris_rl.distributed_env import DistributedServiceEnv
from space_debris_rl.hierarchical_training import train_worker
from space_debris_rl._deps import MissingDependencyError, require


def test_worker_training_smoke_writes_model_zip() -> None:
    try:
        require("stable_baselines3", extra="rl", pip_name="stable-baselines3")
    except MissingDependencyError as exc:
        pytest.skip(str(exc))

    env = DistributedServiceEnv(num_nodes=2, max_steps=5, seed=0)
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "test_worker.zip"
        train_worker(
            env,
            total_timesteps=100,
            model_path=str(out_path),
            num_strategies=3,
            seed=0,
        )
        assert out_path.exists()
