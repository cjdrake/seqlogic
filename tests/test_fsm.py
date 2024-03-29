"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Bit, Enum, Module, get_loop, notify
from seqlogic.bits import F, T, X
from seqlogic.enum import Enum as EnumVal
from seqlogic.sim import Region

from .common import p_clk, p_dff, p_rst

# pyright: reportAttributeAccessIssue=false
# pyright: reportReturnType=false


loop = get_loop()


class SeqDetect(EnumVal):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"


async def p_input(
    x: Bit,
    reset_n: Bit,
    clock: Bit,
):
    """TODO(cjdrake): Write docstring."""
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
    clock = Bit(name="clock", parent=top)
    reset_n = Bit(name="reset_n", parent=top)
    ps = Enum(name="ps", parent=top, cls=SeqDetect)
    x = Bit(name="x", parent=top)

    waves = defaultdict(dict)
    top.dump_waves(waves, r".*")

    def ns() -> SeqDetect:
        match ps.value:
            case SeqDetect.A:
                if x.value:
                    return SeqDetect.B
                else:
                    return SeqDetect.A
            case SeqDetect.B:
                if x.value:
                    return SeqDetect.C
                else:
                    return SeqDetect.A
            case SeqDetect.C:
                if x.value:
                    return SeqDetect.D
                else:
                    return SeqDetect.A
            case SeqDetect.D:
                if x.value:
                    return SeqDetect.D
                else:
                    return SeqDetect.A
            case _:
                return SeqDetect.X

    # Schedule input
    loop.add_proc(Region(1), p_input, x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(Region(1), p_dff, ps, ns, reset_n, SeqDetect.A, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(Region(2), p_rst, reset_n, init=T, phase1=6, phase2=10)
    loop.add_proc(Region(2), p_clk, clock, init=F, shift=5, phase1=5, phase2=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: X,
            clock: X,
            x: X,
            ps: SeqDetect.X,
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
