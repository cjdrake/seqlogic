"""
Example FSM implementation

Demonstrate usage of an enum
"""

from seqlogic.enum import Enum
from seqlogic.logic import logic
from seqlogic.sim import SimVar, get_loop, notify

from .common import TraceVar, clock_drv, dff_arn_drv, reset_drv, waves

loop = get_loop()


class SeqDetect(Enum):
    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"

    XX = "2bxx"


class TraceSeqDetect(SimVar):
    """
    Variable that supports dumping to memory.
    """

    def __init__(self):
        super().__init__(value=SeqDetect.XX)
        waves[self._sim.time()][self] = self._value

    def update(self):
        if self.dirty():
            waves[self._sim.time()][self] = self._next
        super().update()


async def input_drv(
    x: TraceVar,
    reset_n: TraceVar,
    clock: TraceVar,
):
    await notify(reset_n.negedge)
    x.next = logic.F

    await notify(reset_n.posedge)
    await notify(clock.posedge)
    await notify(clock.posedge)
    x.next = logic.T  # A => B

    await notify(clock.posedge)
    x.next = logic.T  # B => C

    await notify(clock.posedge)
    x.next = logic.T  # C => D

    await notify(clock.posedge)
    x.next = logic.T  # D => D

    await notify(clock.posedge)
    x.next = logic.F  # D => A


def test_fsm():
    """Test a 3-bit LFSR"""
    loop.reset()
    waves.clear()

    # State Variables
    ps = TraceSeqDetect()

    # Inputs
    x = TraceVar()

    def ns() -> SeqDetect:
        match (ps.value, x.value):
            case (SeqDetect.A, logic.F):
                return SeqDetect.A
            case (SeqDetect.A, logic.T):
                return SeqDetect.B

            case (SeqDetect.B, logic.F):
                return SeqDetect.A
            case (SeqDetect.B, logic.T):
                return SeqDetect.C

            case (SeqDetect.C, logic.F):
                return SeqDetect.A
            case (SeqDetect.C, logic.T):
                return SeqDetect.D

            case (SeqDetect.D, logic.F):
                return SeqDetect.A
            case (SeqDetect.D, logic.T):
                return SeqDetect.D

            case _:
                return SeqDetect.XX

    # Control Variables
    reset_n = TraceVar()
    reset_value = SeqDetect.A
    clock = TraceVar()

    # Schedule input
    loop.add_proc(input_drv, 0, x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(dff_arn_drv, 0, ps, ns, reset_n, reset_value, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, 1, reset_n, init=logic.T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(clock_drv, 1, clock, init=logic.F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: logic.X,
            clock: logic.X,
            x: logic.X,
            ps: SeqDetect.XX,
        },
        0: {
            reset_n: logic.T,
            clock: logic.F,
        },
        # clock.posedge; reset_n = 1
        5: {
            clock: logic.T,
        },
        # reset_n.negedge
        6: {
            reset_n: logic.F,
            x: logic.F,
            ps: SeqDetect.A,
        },
        10: {
            clock: logic.F,
        },
        # clock.posedge; reset_n = 0
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
        # clock.posedge; reset_n = 1
        25: {
            clock: logic.T,
        },
        30: {
            clock: logic.F,
        },
        # clock.posedge
        35: {
            clock: logic.T,
            x: logic.T,
        },
        40: {
            clock: logic.F,
        },
        # clock.posedge
        45: {
            clock: logic.T,
            ps: SeqDetect.B,
        },
        50: {
            clock: logic.F,
        },
        # clock.posedge
        55: {
            clock: logic.T,
            ps: SeqDetect.C,
        },
        60: {
            clock: logic.F,
        },
        # clock.posedge
        65: {
            clock: logic.T,
            ps: SeqDetect.D,
        },
        70: {
            clock: logic.F,
        },
        # clock.posedge
        75: {
            clock: logic.T,
            x: logic.F,
            # ps: D => D
        },
        80: {
            clock: logic.F,
        },
        # clock.posedge
        85: {
            clock: logic.T,
            ps: SeqDetect.A,
        },
        90: {
            clock: logic.F,
        },
        # clock.posedge
        95: {
            clock: logic.T,
        },
    }

    assert waves == exp
