"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Enum, Module, Packed, Vec, get_loop
from seqlogic.control.globals import drive_clock, drive_reset

loop = get_loop()


class SeqDetect(Enum):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"


async def drive_input(x, reset_n, clock):
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


def f(ps: SeqDetect, x: Vec[1]) -> SeqDetect:
    match ps:
        case SeqDetect.A:
            if x:
                return SeqDetect.B
            else:
                return SeqDetect.A
        case SeqDetect.B:
            if x:
                return SeqDetect.C
            else:
                return SeqDetect.A
        case SeqDetect.C:
            if x:
                return SeqDetect.D
            else:
                return SeqDetect.A
        case SeqDetect.D:
            if x:
                return SeqDetect.D
            else:
                return SeqDetect.A
        case _:
            return SeqDetect.xprop(ps)


def test_fsm():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Module(name="top", parent=None)

    clock = Packed(name="clock", parent=top, dtype=Vec[1])
    reset_n = Packed(name="reset_n", parent=top, dtype=Vec[1])

    ps = Packed(name="ps", parent=top, dtype=SeqDetect)
    ns = Packed(name="ns", parent=top, dtype=SeqDetect)
    x = Packed(name="x", parent=top, dtype=Vec[1])

    reset = Packed(name="reset", parent=top, dtype=Vec[1])
    top.expr(reset, ~reset_n)

    top.combi(ns, f, ps, x)
    top.dff_ar(ps, ns, clock, reset, SeqDetect.A)

    top.elab()

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/x")
    top.dump_waves(waves, r"/top/ps")

    # Schedule input
    loop.add_active(drive_input(x, reset_n, clock))

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_active(drive_reset(reset_n, offticks=6, onticks=10))
    loop.add_active(drive_clock(clock, shiftticks=5, onticks=5, offticks=5))

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {x: "1bX", ps: SeqDetect.X},
        # reset_n.negedge
        6: {x: "1b0", ps: SeqDetect.A},
        35: {x: "1b1"},
        45: {ps: SeqDetect.B},
        55: {ps: SeqDetect.C},
        65: {ps: SeqDetect.D},
        75: {x: "1b0"},
        85: {ps: SeqDetect.A},
    }

    assert waves == exp
