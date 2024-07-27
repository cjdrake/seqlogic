"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Enum, Module, Packed, Vector, get_loop
from seqlogic.sim import Region

from .common import p_clk, p_dff, p_rst

loop = get_loop()


class SeqDetect(Enum):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"


async def p_input(x, reset_n, clock):
    await reset_n.negedge()
    x.next = "1b0"

    await reset_n.posedge()
    await clock.posedge()
    await clock.posedge()
    x.next = "1b1"  # A => B

    await clock.posedge()
    x.next = "1b1"  # B => C

    await clock.posedge()
    x.next = "1b1"  # C => D

    await clock.posedge()
    x.next = "1b1"  # D => D

    await clock.posedge()
    x.next = "1b0"  # D => A


def test_fsm():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Module(name="top", parent=None)
    clock = Packed(name="clock", parent=top, dtype=Vector[1])
    reset_n = Packed(name="reset_n", parent=top, dtype=Vector[1])
    ps = Packed(name="ps", parent=top, dtype=SeqDetect)
    x = Packed(name="x", parent=top, dtype=Vector[1])

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
    loop.add_proc(Region.ACTIVE, p_input, x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(Region.ACTIVE, p_dff, ps, ns, clock, reset_n, SeqDetect.A)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(Region.ACTIVE, p_rst, reset_n, init="1b1", phase1=6, phase2=10)
    loop.add_proc(Region.ACTIVE, p_clk, clock, init="1b0", shift=5, phase1=5, phase2=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: "1bX",
            clock: "1bX",
            x: "1bX",
            ps: SeqDetect.X,
        },
        0: {
            reset_n: "1b1",
            clock: "1b0",
        },
        # clock.posedge; reset_n = 1
        5: {
            clock: "1b1",
        },
        # reset_n.negedge
        6: {
            reset_n: "1b0",
            x: "1b0",
            ps: SeqDetect.A,
        },
        10: {
            clock: "1b0",
        },
        # clock.posedge; reset_n = 0
        15: {
            clock: "1b1",
        },
        # reset_n.posedge
        16: {
            reset_n: "1b1",
        },
        20: {
            clock: "1b0",
        },
        # clock.posedge; reset_n = 1
        25: {
            clock: "1b1",
        },
        30: {
            clock: "1b0",
        },
        # clock.posedge
        35: {
            clock: "1b1",
            x: "1b1",
        },
        40: {
            clock: "1b0",
        },
        # clock.posedge
        45: {
            clock: "1b1",
            ps: SeqDetect.B,
        },
        50: {
            clock: "1b0",
        },
        # clock.posedge
        55: {
            clock: "1b1",
            ps: SeqDetect.C,
        },
        60: {
            clock: "1b0",
        },
        # clock.posedge
        65: {
            clock: "1b1",
            ps: SeqDetect.D,
        },
        70: {
            clock: "1b0",
        },
        # clock.posedge
        75: {
            clock: "1b1",
            x: "1b0",
            # ps: D => D
        },
        80: {
            clock: "1b0",
        },
        # clock.posedge
        85: {
            clock: "1b1",
            ps: SeqDetect.A,
        },
        90: {
            clock: "1b0",
        },
        # clock.posedge
        95: {
            clock: "1b1",
        },
    }

    assert waves == exp
