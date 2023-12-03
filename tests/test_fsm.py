"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from seqlogic.enum import Enum
from seqlogic.logic import logic
from seqlogic.sim import Region, SimVar, get_loop, notify

from .common import _TraceVar, clock_drv, dff_arn_drv, reset_drv, waves

loop = get_loop()


class SeqDetect(Enum):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"

    XX = "2bxx"


class _TraceSeqDetect(SimVar):
    """Variable that supports dumping to memory."""

    def __init__(self):
        super().__init__(value=SeqDetect.XX)
        waves[self._sim.time()][self] = self._value

    def update(self):
        if self.dirty():
            waves[self._sim.time()][self] = self._next
        super().update()


async def _input_drv(
    x: _TraceVar,
    reset_n: _TraceVar,
    clock: _TraceVar,
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
    """Test a 3-bit LFSR."""
    loop.reset()
    waves.clear()

    # State Variables
    ps = _TraceSeqDetect()

    # Inputs
    x = _TraceVar()

    def ns() -> SeqDetect:
        match (ps.value, x.value):
            case (SeqDetect.A, logic.F):
                return SeqDetect.A  # pyright: ignore[reportGeneralTypeIssues]
            case (SeqDetect.A, logic.T):
                return SeqDetect.B  # pyright: ignore[reportGeneralTypeIssues]

            case (SeqDetect.B, logic.F):
                return SeqDetect.A  # pyright: ignore[reportGeneralTypeIssues]
            case (SeqDetect.B, logic.T):
                return SeqDetect.C  # pyright: ignore[reportGeneralTypeIssues]

            case (SeqDetect.C, logic.F):
                return SeqDetect.A  # pyright: ignore[reportGeneralTypeIssues]
            case (SeqDetect.C, logic.T):
                return SeqDetect.D  # pyright: ignore[reportGeneralTypeIssues]

            case (SeqDetect.D, logic.F):
                return SeqDetect.A  # pyright: ignore[reportGeneralTypeIssues]
            case (SeqDetect.D, logic.T):
                return SeqDetect.D  # pyright: ignore[reportGeneralTypeIssues]

            case _:
                return SeqDetect.XX  # pyright: ignore[reportGeneralTypeIssues]

    # Control Variables
    reset_n = _TraceVar()
    reset_value = SeqDetect.A
    clock = _TraceVar()

    # Schedule input
    loop.add_proc(_input_drv, Region(0), x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(dff_arn_drv, Region(0), ps, ns, reset_n, reset_value, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, Region(1), reset_n, init=logic.T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(
        clock_drv, Region(1), clock, init=logic.F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5
    )

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
