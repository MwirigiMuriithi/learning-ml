from agriprice.validate import run_acceptance


def test_all_acceptance_criteria_pass(cfg, world, panels):
    table, ev = run_acceptance(cfg, panels, world)
    failed = table[~table.passed]
    assert failed.empty, failed[["check", "detail"]].to_string()
    assert ev.passed.all()
