"""
Example LFSR implementation
"""

from collections import defaultdict

from seqlogic.logic import logic
from seqlogic.logicvec import vec, xes
from seqlogic.sim import SimVar, get_loop, notify, sleep

loop = get_loop()
waves = defaultdict(dict)


def waves_add(time, var, val):
    waves[time][var] = val


class TraceVar(SimVar):
    """
    Variable that supports dumping to memory.
    """

    def __init__(self):
        super().__init__(value=logic.X)
        waves_add(self._sim.time(), self, self._value)

    def update(self):
        if self.dirty():
            waves_add(self._sim.time(), self, self._next)
        super().update()

    def negedge(self) -> bool:
        return (self._value is logic.T) and (self._next is logic.F)

    def posedge(self) -> bool:
        return (self._value is logic.F) and (self._next is logic.T)


class TraceVec(SimVar):
    """
    Variable that supports dumping to memory.
    """

    def __init__(self, n: int):
        super().__init__(value=xes((n,)))
        waves_add(self._sim.time(), self, self._value)

    def update(self):
        if self.dirty():
            waves_add(self._sim.time(), self, self._next)
        super().update()


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
        raise ValueError(f"Expected phase1_ticks >= 0, got {phase1_ticks}")
    if phase2_ticks < 0:
        raise ValueError(f"Expected phase2_ticks >= 0, got {phase2_ticks}")

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
        raise ValueError(f"Expected shift_ticks >= 0, got {shift_ticks}")
    if phase1_ticks < 0:
        raise ValueError(f"Expected phase1_ticks >= 0, got {phase1_ticks}")
    if phase2_ticks < 0:
        raise ValueError(f"Expected phase2_ticks >= 0, got {phase2_ticks}")

    # T = 0
    clock.next = init
    await sleep(shift_ticks)

    while True:
        clock.next = ~clock.value
        await sleep(phase1_ticks)

        clock.next = ~clock.value
        await sleep(phase2_ticks)


async def lfsr_drv(lfsr: TraceVec, reset_n: TraceVar, clock: TraceVar):
    while True:
        var = await notify(reset_n.negedge, clock.posedge)
        assert var in {reset_n, clock}
        if var is reset_n:
            lfsr.next = vec("3'b100")
        elif var is clock and reset_n.value is logic.T:
            lfsr.next = vec([lfsr.value[2] ^ lfsr.value[0], lfsr.value[0], lfsr.value[1]])


def test_lfsr():
    """Test a 3-bit LFSR"""
    loop.reset()
    waves.clear()

    reset_n = TraceVar()
    clock = TraceVar()
    lfsr = TraceVec(3)

    # Schedule LFSR
    loop.add_proc(lfsr_drv, 0, lfsr, reset_n, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, 1, reset_n, init=logic.T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(clock_drv, 1, clock, init=logic.F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5)

    loop.run(until=100)

    exp = {
        -1: {
            reset_n: logic.X,
            clock: logic.X,
            lfsr: vec("3'bxxx"),
        },
        0: {
            reset_n: logic.T,
            clock: logic.F,
        },
        5: {
            clock: logic.T,
        },
        # reset_n.negedge
        6: {
            reset_n: logic.F,
            lfsr: vec("3'b100"),
        },
        10: {
            clock: logic.F,
        },
        15: {
            clock: logic.T,
        },
        # reset_n.posedge
        16: {
            reset_n: logic.T,
        },
        20: {
            clock: logic.F,
        },
        # clock.posedge
        25: {
            clock: logic.T,
            lfsr: vec("3'b001"),
        },
        30: {
            clock: logic.F,
        },
        35: {
            clock: logic.T,
            lfsr: vec("3'b011"),
        },
        40: {
            clock: logic.F,
        },
        45: {
            clock: logic.T,
            lfsr: vec("3'b111"),
        },
        50: {
            clock: logic.F,
        },
        55: {
            clock: logic.T,
            lfsr: vec("3'b110"),
        },
        60: {
            clock: logic.F,
        },
        65: {
            clock: logic.T,
            lfsr: vec("3'b101"),
        },
        70: {
            clock: logic.F,
        },
        75: {
            clock: logic.T,
            lfsr: vec("3'b010"),
        },
        80: {
            clock: logic.F,
        },
        # Repeat cycle
        85: {
            clock: logic.T,
            lfsr: vec("3'b100"),
        },
        90: {
            clock: logic.F,
        },
        95: {
            clock: logic.T,
            lfsr: vec("3'b001"),
        },
    }

    assert waves == exp
