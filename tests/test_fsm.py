"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Enum, Module, Packed, Vec, create_task, ite, run
from seqlogic.control.sync import drv_clock, drv_reset


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
            return ite(x, SeqDetect.B, SeqDetect.A)
        case SeqDetect.B:
            return ite(x, SeqDetect.C, SeqDetect.A)
        case SeqDetect.C:
            return ite(x, SeqDetect.D, SeqDetect.A)
        case SeqDetect.D:
            return ite(x, SeqDetect.D, SeqDetect.A)
        case _:
            return SeqDetect.xprop(ps)


class MyFsm(Module):
    def build(self):
        pass


def test_fsm():
    """Test a 3-bit LFSR."""
    top = MyFsm(name="top")

    clock = Packed(name="clock", parent=top, dtype=Vec[1])
    reset_n = Packed(name="reset_n", parent=top, dtype=Vec[1])

    ps = Packed(name="ps", parent=top, dtype=SeqDetect)
    ns = Packed(name="ns", parent=top, dtype=SeqDetect)
    x = Packed(name="x", parent=top, dtype=Vec[1])

    top.combi(ns, f, ps, x)
    top.dff(ps, ns, clock, rst=reset_n, rval=SeqDetect.A, rneg=True)

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/x")
    top.dump_waves(waves, r"/top/ps")

    async def main():
        await top.main()

        # Schedule input
        create_task(drive_input(x, reset_n, clock))

        # Schedule reset and clock
        # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
        create_task(drv_reset(reset_n, shiftticks=6, onticks=10, neg=True))
        create_task(drv_clock(clock, shiftticks=5, onticks=5, offticks=5))

    run(main(), until=100)

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
