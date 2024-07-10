from optexp.runner.exp_state import ExperimentState, IterationCounter


def test_iteration_counter():
    ic = IterationCounter()
    assert ic.epoch == 0
    assert ic.step == 0
    assert ic.step_within_epoch == 0

    ic.start()
    assert ic.epoch == 1
    assert ic.step == 1
    assert ic.step_within_epoch == 1

    ic.next_iter()
    assert ic.epoch == 1
    assert ic.step == 2
    assert ic.step_within_epoch == 2

    ic.next_iter()
    assert ic.epoch == 1
    assert ic.step == 3
    assert ic.step_within_epoch == 3

    ic.next_epoch()
    assert ic.epoch == 2
    assert ic.step == 3
    assert ic.step_within_epoch == 1

    ic.next_iter()
    assert ic.epoch == 2
    assert ic.step == 4
    assert ic.step_within_epoch == 2


def test_interaction_counters_are_independent():
    ic1 = IterationCounter()
    ic2 = IterationCounter()

    ic1.start()
    assert ic1.epoch == 1
    assert ic1.step == 1
    assert ic1.step_within_epoch == 1

    ic1.next_iter()
    assert ic1.epoch == 1
    assert ic1.step == 2
    assert ic1.step_within_epoch == 2

    assert ic2.epoch == 0
    assert ic2.step == 0
    assert ic2.step_within_epoch == 0
