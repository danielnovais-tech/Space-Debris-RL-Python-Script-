from space_debris_rl.safety import SafetyMonitor


def test_strategy_out_of_bounds_rejected():
    sm = SafetyMonitor(strategy_space_n=3)
    d = sm.validate_strategy(5)
    assert d.ok is False
    assert d.reason == "strategy_out_of_bounds"


def test_ltl_veto_restarts():
    sm = SafetyMonitor(ltl_formulas=["always_restart_less_than_3_per_hour"])
    # Two ok
    assert sm.check_ltl({"action_taken": "restart"}).ok
    assert sm.check_ltl({"action_taken": "restart"}).ok
    # Third violates
    assert sm.check_ltl({"action_taken": "restart"}).ok is False
