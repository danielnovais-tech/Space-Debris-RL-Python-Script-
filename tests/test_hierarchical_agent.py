import numpy as np

from space_debris_rl.hrl import HierarchicalAgent


def test_hierarchical_agent_scalar_action_default_mapping():
    agent = HierarchicalAgent()
    action, info = agent.act(np.array([0.0], dtype=np.float32))
    assert isinstance(action, (int, np.integer))
    assert "strategy" in info and "action" in info


def test_hierarchical_agent_vector_action_for_distributed():
    agent = HierarchicalAgent(num_nodes=3)
    action, info = agent.act(np.array([95.0], dtype=np.float32))
    arr = np.asarray(action)
    assert arr.shape == (3,)
    assert set(arr.tolist()).issubset({0, 1, 2, 3})
    assert info["strategy_name"] in {"scale_up", "restart", "noop"}
