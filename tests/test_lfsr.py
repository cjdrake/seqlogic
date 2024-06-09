"""Example LFSR implementation."""

from collections import defaultdict

from seqlogic import Bit, Bits, Module, get_loop
from seqlogic.lbool import Vec, cat
from seqlogic.sim import Region

from .common import p_clk, p_dff, p_rst

loop = get_loop()


class Top(Module):
    """Top level module."""

    def __init__(self):
        super().__init__(name="top", parent=None)
        # Control
        self.reset_n = Bit(name="reset_n", parent=self)
        self.clock = Bit(name="clock", parent=self)
        # State
        self.q = Bits(name="q", parent=self, dtype=Vec[3])


def test_lfsr():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Top()
    waves = defaultdict(dict)
    top.dump_waves(waves, r".*")

    assert top.q.name == "q"
    assert top.q.qualname == "/top/q"

    def d():
        v = top.q.value
        return cat(v[0] ^ v[2], v[:2])

    # Schedule LFSR
    loop.add_proc(Region.ACTIVE, p_dff, top.q, d, top.reset_n, "3b100", top.clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(Region.ACTIVE, p_rst, top.reset_n, init="1b1", phase1=6, phase2=10)
    loop.add_proc(Region.ACTIVE, p_clk, top.clock, init="1b0", shift=5, phase1=5, phase2=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            top.reset_n: "1bX",
            top.clock: "1bX",
            top.q: "3bXXX",
        },
        0: {
            top.reset_n: "1b1",
            top.clock: "1b0",
        },
        # clock.posedge; reset_n = 1
        # q = xxx
        5: {
            top.clock: "1b1",
        },
        # reset_n.negedge
        # q = reset_value
        6: {
            top.reset_n: "1b0",
            top.q: "3b100",
        },
        10: {
            top.clock: "1b0",
        },
        # clock.posedge; reset_n = 0
        15: {
            top.clock: "1b1",
        },
        # reset_n.posedge
        16: {
            top.reset_n: "1b1",
        },
        20: {
            top.clock: "1b0",
        },
        # clock.posedge; reset_n = 1
        # q = 001
        25: {
            top.clock: "1b1",
            top.q: "3b001",
        },
        30: {
            top.clock: "1b0",
        },
        35: {
            top.clock: "1b1",
            top.q: "3b011",
        },
        40: {
            top.clock: "1b0",
        },
        45: {
            top.clock: "1b1",
            top.q: "3b111",
        },
        50: {
            top.clock: "1b0",
        },
        55: {
            top.clock: "1b1",
            top.q: "3b110",
        },
        60: {
            top.clock: "1b0",
        },
        65: {
            top.clock: "1b1",
            top.q: "3b101",
        },
        70: {
            top.clock: "1b0",
        },
        75: {
            top.clock: "1b1",
            top.q: "3b010",
        },
        80: {
            top.clock: "1b0",
        },
        # Repeat cycle
        85: {
            top.clock: "1b1",
            top.q: "3b100",
        },
        90: {
            top.clock: "1b0",
        },
        95: {
            top.clock: "1b1",
            top.q: "3b001",
        },
    }

    assert waves == exp
