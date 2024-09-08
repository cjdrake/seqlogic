"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Enum, Module, Packed, Vec, get_loop
from seqlogic.control.globals import drv_clock, drv_reset

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


def g(x: Vec[1], ns: SeqDetect) -> SeqDetect:
    match x:
        case "1b0":
            return SeqDetect.A
        case "1b1":
            return ns
        case _:
            return SeqDetect.xprop(x)


def f(ps: SeqDetect, x: Vec[1]) -> SeqDetect:
    match ps:
        case SeqDetect.A:
            return g(x, SeqDetect.B)
        case SeqDetect.B:
            return g(x, SeqDetect.C)
        case SeqDetect.C:
            return g(x, SeqDetect.D)
        case SeqDetect.D:
            return g(x, SeqDetect.D)
        case _:
            return SeqDetect.xprop(ps)


class MyFsm(Module):
    def build(self):
        pass


def test_fsm():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = MyFsm(name="top")

    clock = Packed(name="clock", parent=top, dtype=Vec[1])
    reset_n = Packed(name="reset_n", parent=top, dtype=Vec[1])

    ps = Packed(name="ps", parent=top, dtype=SeqDetect)
    ns = Packed(name="ns", parent=top, dtype=SeqDetect)
    x = Packed(name="x", parent=top, dtype=Vec[1])

    top.combi(ns, f, ps, x)
    top.dff_r(ps, ns, clock, reset_n, rval=SeqDetect.A, rneg=True)

    top.elab()

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/x")
    top.dump_waves(waves, r"/top/ps")

    # Schedule input
    loop.add_initial(drive_input(x, reset_n, clock))

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_initial(drv_reset(reset_n, shiftticks=6, onticks=10, neg=True))
    loop.add_initial(drv_clock(clock, shiftticks=5, onticks=5, offticks=5))

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
