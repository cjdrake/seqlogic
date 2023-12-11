"""Common code.

For now this is just used for testing.
It might be useful to add to seqlogic library.
"""

from collections import defaultdict
from collections.abc import Callable

from seqlogic.hier import HierVar, Module
from seqlogic.logic import logic
from seqlogic.logicvec import logicvec, xes
from seqlogic.sim import SimVar, notify, sleep

# [Time][Var] = Val
waves = defaultdict(dict)


class TraceVar(SimVar, HierVar):
    """Variable that supports dumping to memory."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        SimVar.__init__(self, value=logic.X)
        HierVar.__init__(self, name, parent)
        waves[self._sim.time()][self] = self._value

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self.dirty():
            waves[self._sim.time()][self] = self._next_value
        SimVar.update(self)

    def negedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return (self._value is logic.T) and (self._next_value is logic.F)

    def posedge(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return (self._value is logic.F) and (self._next_value is logic.T)


class TraceVec(SimVar, HierVar):
    """Variable that supports dumping to memory."""

    def __init__(self, name: str, n: int, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        SimVar.__init__(self, value=xes((n,)))
        HierVar.__init__(self, name, parent)
        waves[self._sim.time()][self] = self._value

    def update(self):
        """TODO(cjdrake): Write docstring."""
        if self.dirty():
            waves[self._sim.time()][self] = self._next_value
        SimVar.update(self)


async def reset_drv(
    reset: TraceVar, init: logic = logic.T, phase1_ticks: int = 1, phase2_ticks: int = 1
):
    r"""
    Simulate a reset signal.

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾

    <phase1> <phase2>

    Args:
        phase1_ticks: Number of sim ticks in ...
        phase2_ticks: Number of sim ticks in ...

    Raises:
        ValueError: An argument has the correct type, but an incorrect value
    """
    if phase1_ticks < 0:
        raise ValueError(f"Expected phase1_ticks ≥ 0, got {phase1_ticks}")
    if phase2_ticks < 0:
        raise ValueError(f"Expected phase2_ticks ≥ 0, got {phase2_ticks}")

    # T = 0
    reset.next = init
    await sleep(phase1_ticks)

    reset.next = ~reset.value
    await sleep(phase2_ticks)

    reset.next = ~reset.value


async def clock_drv(
    clock: TraceVar,
    init: logic = logic.F,
    shift_ticks: int = 0,
    phase1_ticks: int = 1,
    phase2_ticks: int = 1,
):
    r"""
    Simulate a clock signal.

    ________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________

    <shift> <phase1> <phase2>

    Args:
        init: Initial clock signal value
        shift_ticks: Number of sim ticks to shift the first clock transition
        phase1_ticks: Number of sim ticks in the clock's active phase
        phase2_ticks: Number of sim ticks in the clock's passive phase

    The period is phase1_ticks + phase2_ticks
    The duty cycle is phase1_ticks / (phase1_ticks + phase2_ticks)

    Raises:
        ValueError: An argument has the correct type, but an incorrect value
    """
    if shift_ticks < 0:
        raise ValueError(f"Expected shift_ticks ≥ 0, got {shift_ticks}")
    if phase1_ticks < 0:
        raise ValueError(f"Expected phase1_ticks ≥ 0, got {phase1_ticks}")
    if phase2_ticks < 0:
        raise ValueError(f"Expected phase2_ticks ≥ 0, got {phase2_ticks}")

    # T = 0
    clock.next = init
    await sleep(shift_ticks)

    while True:
        clock.next = ~clock.value
        await sleep(phase1_ticks)

        clock.next = ~clock.value
        await sleep(phase2_ticks)


async def dff_arn_drv(
    q: SimVar,
    d: Callable[[], logicvec],
    reset_n: TraceVar,
    reset_value: logicvec,
    clock: TraceVar,
):
    """D Flop Flop with asynchronous, negedge-triggered reset."""
    while True:
        var = await notify(reset_n.negedge, clock.posedge)
        assert var in {reset_n, clock}
        if var is reset_n:
            q.next = reset_value
        elif var is clock and reset_n.value is logic.T:
            q.next = d()
