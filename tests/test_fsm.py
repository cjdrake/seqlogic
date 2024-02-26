"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Module
from seqlogic.enum import Enum
from seqlogic.logicvec import F, T, X
from seqlogic.sim import Region, get_loop, notify
from seqlogic.var import Logic, TraceVar

from .common import clock_drv, dff_arn_drv, reset_drv

loop = get_loop()


class SeqDetect(Enum):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"

    XX = "2bxx"


class EnumVar(TraceVar):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent, SeqDetect.XX)


async def _input_drv(
    x: Logic,
    reset_n: Logic,
    clock: Logic,
):
    await notify(reset_n.negedge)
    x.next = F

    await notify(reset_n.posedge)
    await notify(clock.posedge)
    await notify(clock.posedge)
    x.next = T  # A => B

    await notify(clock.posedge)
    x.next = T  # B => C

    await notify(clock.posedge)
    x.next = T  # C => D

    await notify(clock.posedge)
    x.next = T  # D => D

    await notify(clock.posedge)
    x.next = F  # D => A


def test_fsm():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Module(name="top")
    clock = Logic(name="clock", parent=top)
    reset_n = Logic(name="reset_n", parent=top)
    ps = EnumVar(name="ps", parent=top)
    x = Logic(name="x", parent=top)

    waves = defaultdict(dict)
    top.dump_waves(waves, r".*")

    def ns() -> SeqDetect:
        match ps.value:
            case SeqDetect.A:
                if x.value == T:
                    return SeqDetect.B  # pyright: ignore[reportReturnType]
                else:
                    return SeqDetect.A  # pyright: ignore[reportReturnType]
            case SeqDetect.B:
                if x.value == T:
                    return SeqDetect.C  # pyright: ignore[reportReturnType]
                else:
                    return SeqDetect.A  # pyright: ignore[reportReturnType]
            case SeqDetect.C:
                if x.value == T:
                    return SeqDetect.D  # pyright: ignore[reportReturnType]
                else:
                    return SeqDetect.A  # pyright: ignore[reportReturnType]
            case SeqDetect.D:
                if x.value == T:
                    return SeqDetect.D  # pyright: ignore[reportReturnType]
                else:
                    return SeqDetect.A  # pyright: ignore[reportReturnType]

            case _:
                return SeqDetect.XX  # pyright: ignore[reportReturnType]

    # Schedule input
    loop.add_proc(_input_drv, Region(0), x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(dff_arn_drv, Region(0), ps, ns, reset_n, SeqDetect.A, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, Region(1), reset_n, init=T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(
        clock_drv, Region(1), clock, init=F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5
    )

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: X,
            clock: X,
            x: X,
            ps: SeqDetect.XX,
        },
        0: {
            reset_n: T,
            clock: F,
        },
        # clock.posedge; reset_n = 1
        5: {
            clock: T,
        },
        # reset_n.negedge
        6: {
            reset_n: F,
            x: F,
            ps: SeqDetect.A,
        },
        10: {
            clock: F,
        },
        # clock.posedge; reset_n = 0
        15: {
            clock: T,
        },
        # reset_n.posedge
        16: {
            reset_n: T,
        },
        20: {
            clock: F,
        },
        # clock.posedge; reset_n = 1
        25: {
            clock: T,
        },
        30: {
            clock: F,
        },
        # clock.posedge
        35: {
            clock: T,
            x: T,
        },
        40: {
            clock: F,
        },
        # clock.posedge
        45: {
            clock: T,
            ps: SeqDetect.B,
        },
        50: {
            clock: F,
        },
        # clock.posedge
        55: {
            clock: T,
            ps: SeqDetect.C,
        },
        60: {
            clock: F,
        },
        # clock.posedge
        65: {
            clock: T,
            ps: SeqDetect.D,
        },
        70: {
            clock: F,
        },
        # clock.posedge
        75: {
            clock: T,
            x: F,
            # ps: D => D
        },
        80: {
            clock: F,
        },
        # clock.posedge
        85: {
            clock: T,
            ps: SeqDetect.A,
        },
        90: {
            clock: F,
        },
        # clock.posedge
        95: {
            clock: T,
        },
    }

    assert waves == exp
