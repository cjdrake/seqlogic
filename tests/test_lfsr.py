"""Example LFSR implementation."""

from collections import defaultdict

from bvwx import Vec, cat
from deltacycle import create_task, run

from seqlogic import Module
from seqlogic.control.sync import drv_clock, drv_reset


def lfsr(x: Vec[3]) -> Vec[3]:
    return cat(x[0] ^ x[2], x[:2])


class Top(Module):
    """Top level module."""

    def build(self):
        # Control
        clock = self.input(name="clock", dtype=Vec[1])
        reset_n = self.input(name="reset_n", dtype=Vec[1])

        # State
        q = self.logic(name="q", dtype=Vec[3])
        d = self.logic(name="d", dtype=Vec[3])

        self.combi(d, lfsr, q)
        self.dff(q, d, clock, rst=reset_n, rval="3b100", rneg=True)


def test_lfsr():
    """Test a 3-bit LFSR."""
    top = Top(name="top")

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/q")

    assert top.q.name == "q"
    assert top.q.qualname == "/top/q"

    async def main():
        create_task(top.main())

        # Schedule reset and clock
        # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
        create_task(drv_reset(top.reset_n, shiftticks=6, onticks=10, neg=True))
        create_task(drv_clock(top.clock, shiftticks=5, onticks=5, offticks=5))

    run(main(), until=100)

    exp = {
        # Initialize everything to X'es
        -1: {top.q: "3bXXX"},
        # reset_n.negedge
        6: {top.q: "3b100"},
        # clock.posedge; reset_n = 1
        25: {top.q: "3b001"},
        35: {top.q: "3b011"},
        45: {top.q: "3b111"},
        55: {top.q: "3b110"},
        65: {top.q: "3b101"},
        75: {top.q: "3b010"},
        # Repeat cycle
        85: {top.q: "3b100"},
        95: {top.q: "3b001"},
    }

    assert waves == exp
