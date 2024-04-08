"""Example LFSR implementation."""

from collections import defaultdict

from seqlogic import Bit, Bits, Module, get_loop
from seqlogic.lbool import cat, ones, vec, xes, zeros
from seqlogic.sim import Region

from .common import p_clk, p_dff, p_rst

loop = get_loop()


class Top(Module):
    """Top level module."""

    def __init__(self):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name="top", parent=None)
        # Control
        self.reset_n = Bit(name="reset_n", parent=self)
        self.clock = Bit(name="clock", parent=self)
        # State
        self.q = Bits(name="q", parent=self, shape=(3,))


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

    reset_value = vec("3b100")

    # Schedule LFSR
    loop.add_proc(Region(1), p_dff, top.q, d, top.reset_n, reset_value, top.clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(Region(2), p_rst, top.reset_n, init=ones(1), phase1=6, phase2=10)
    loop.add_proc(Region(2), p_clk, top.clock, init=zeros(1), shift=5, phase1=5, phase2=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            top.reset_n: xes(1),
            top.clock: xes(1),
            top.q: xes(3),
        },
        0: {
            top.reset_n: ones(1),
            top.clock: zeros(1),
        },
        # clock.posedge; reset_n = 1
        # q = xxx
        5: {
            top.clock: ones(1),
        },
        # reset_n.negedge
        # q = reset_value
        6: {
            top.reset_n: zeros(1),
            top.q: vec("3b100"),
        },
        10: {
            top.clock: zeros(1),
        },
        # clock.posedge; reset_n = 0
        15: {
            top.clock: ones(1),
        },
        # reset_n.posedge
        16: {
            top.reset_n: ones(1),
        },
        20: {
            top.clock: zeros(1),
        },
        # clock.posedge; reset_n = 1
        # q = 001
        25: {
            top.clock: ones(1),
            top.q: vec("3b001"),
        },
        30: {
            top.clock: zeros(1),
        },
        35: {
            top.clock: ones(1),
            top.q: vec("3b011"),
        },
        40: {
            top.clock: zeros(1),
        },
        45: {
            top.clock: ones(1),
            top.q: vec("3b111"),
        },
        50: {
            top.clock: zeros(1),
        },
        55: {
            top.clock: ones(1),
            top.q: vec("3b110"),
        },
        60: {
            top.clock: zeros(1),
        },
        65: {
            top.clock: ones(1),
            top.q: vec("3b101"),
        },
        70: {
            top.clock: zeros(1),
        },
        75: {
            top.clock: ones(1),
            top.q: vec("3b010"),
        },
        80: {
            top.clock: zeros(1),
        },
        # Repeat cycle
        85: {
            top.clock: ones(1),
            top.q: vec("3b100"),
        },
        90: {
            top.clock: zeros(1),
        },
        95: {
            top.clock: ones(1),
            top.q: vec("3b001"),
        },
    }

    assert waves == exp
