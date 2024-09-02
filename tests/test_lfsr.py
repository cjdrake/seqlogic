"""Example LFSR implementation."""

# This tracing method requires cross module references to _protected logic
# pylint: disable=protected-access

from collections import defaultdict

from seqlogic import Module, Vec, cat, get_loop
from seqlogic.control.globals import drive_clock, drive_reset

loop = get_loop()


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

        # Invert the reset
        reset = self.logic(name="reset", dtype=Vec[1])
        self.expr(reset, ~reset_n)

        self.combi(d, lfsr, q)
        self.dff_ar(q, d, clock, reset, "3b100")


def test_lfsr():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Top(name="top")
    top.elab()

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/q")

    assert top._q.name == "q"
    assert top._q.qualname == "/top/q"

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_initial(drive_reset(top.reset_n, offticks=6, onticks=10))
    loop.add_initial(drive_clock(top.clock, shiftticks=5, onticks=5, offticks=5))

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {top._q: "3bXXX"},
        # reset_n.negedge
        6: {top._q: "3b100"},
        # clock.posedge; reset_n = 1
        25: {top._q: "3b001"},
        35: {top._q: "3b011"},
        45: {top._q: "3b111"},
        55: {top._q: "3b110"},
        65: {top._q: "3b101"},
        75: {top._q: "3b010"},
        # Repeat cycle
        85: {top._q: "3b100"},
        95: {top._q: "3b001"},
    }

    assert waves == exp
