import numpy as np

from space_debris_rl.hrl import HierarchicalAgent


def test_hierarchical_agent_scalar_action_default_mapping():
    agent = HierarchicalAgent()
    action, info = agent.act(np.array([0.0], dtype=np.float32))
    assert isinstance(action, (int, np.integer))
    assert "strategy" in info and "action" in info
    assert info["worker_used"] is False


def test_hierarchical_agent_vector_action_for_distributed():
    agent = HierarchicalAgent(num_nodes=3)
    action, info = agent.act(np.array([95.0], dtype=np.float32))
    arr = np.asarray(action)
    assert arr.shape == (3,)
    assert set(arr.tolist()).issubset({0, 1, 2, 3})
    assert info["strategy_name"] in {"scale_up", "restart", "noop"}
    assert info["worker_used"] is False


def test_hierarchical_agent_learned_worker_augments_obs_with_one_hot():
    class DummyManager:
        def predict(self, obs, deterministic=True):  # noqa: ARG002
            return 2, None

    class DummyWorker:
        def __init__(self):
            self.last_obs = None

        def predict(self, obs, deterministic=True):  # noqa: ARG002
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return 123, None

    worker = DummyWorker()
    agent = HierarchicalAgent(
        manager=DummyManager(),
        worker=worker,
        use_learned_worker=True,
        worker_strategies=5,
    )
    action, info = agent.act(np.array([1.0, 2.0], dtype=np.float32))

    assert action == 123
    assert info["strategy"] == 2
    assert info["worker_used"] is True
    assert worker.last_obs is not None
    assert worker.last_obs.shape == (2 + 5,)
    assert np.allclose(worker.last_obs[-5:], np.array([0, 0, 1, 0, 0], dtype=np.float32))
