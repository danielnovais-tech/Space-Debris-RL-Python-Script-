from space_debris_rl.safety import SafetyMonitor


def test_check_strategy_restart_veto_low_cpu():
    sm = SafetyMonitor(strategy_space_n=4)
    d = sm.check_strategy(1, system_state={"cpu": 10.0})
    assert d.ok is False
    assert d.reason == "strategy_restart_unnecessary_low_cpu"
