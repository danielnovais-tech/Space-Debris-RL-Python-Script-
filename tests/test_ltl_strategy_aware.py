from space_debris_rl.safety import SafetyMonitor


def test_strategy_aware_restart_rate_limit_only_applies_to_strategy_1() -> None:
    safety = SafetyMonitor(ltl_formulas=["strategy_1_restart_less_than_3_per_hour"], strategy_space_n=4)

    # Feed 3 restarts under strategy 1 -> should veto on 3rd (>=3).
    for i in range(3):
        dec = safety.check_ltl({"action_taken": "restart"}, strategy=1)
    assert not dec.ok
    assert dec.reason is not None

    # Same restarts under a different strategy should not trigger this formula.
    safety.reset()
    for _ in range(10):
        dec2 = safety.check_ltl({"action_taken": "restart"}, strategy=2)
        assert dec2.ok
