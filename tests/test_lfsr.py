"""Example LFSR implementation."""

from collections import defaultdict

from seqlogic import Bit, Bits, Module
from seqlogic.logicvec import F, T, X, cat, logicvec, vec
from seqlogic.sim import Region, get_loop

from .common import clock_drv, dff_arn_drv, reset_drv

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

    def d() -> logicvec:
        v: logicvec = top.q.value
        return cat([v[0] ^ v[2], v[:2]])

    reset_value = vec("3b100")

    # Schedule LFSR
    loop.add_proc(dff_arn_drv, Region(0), top.q, d, top.reset_n, reset_value, top.clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, Region(1), top.reset_n, init=T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(
        clock_drv, Region(1), top.clock, init=F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5
    )

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            top.reset_n: X,
            top.clock: X,
            top.q: vec("3bxxx"),
        },
        0: {
            top.reset_n: T,
            top.clock: F,
        },
        # clock.posedge; reset_n = 1
        # q = xxx
        5: {
            top.clock: T,
        },
        # reset_n.negedge
        # q = reset_value
        6: {
            top.reset_n: F,
            top.q: vec("3b100"),
        },
        10: {
            top.clock: F,
        },
        # clock.posedge; reset_n = 0
        15: {
            top.clock: T,
        },
        # reset_n.posedge
        16: {
            top.reset_n: T,
        },
        20: {
            top.clock: F,
        },
        # clock.posedge; reset_n = 1
        # q = 001
        25: {
            top.clock: T,
            top.q: vec("3b001"),
        },
        30: {
            top.clock: F,
        },
        35: {
            top.clock: T,
            top.q: vec("3b011"),
        },
        40: {
            top.clock: F,
        },
        45: {
            top.clock: T,
            top.q: vec("3b111"),
        },
        50: {
            top.clock: F,
        },
        55: {
            top.clock: T,
            top.q: vec("3b110"),
        },
        60: {
            top.clock: F,
        },
        65: {
            top.clock: T,
            top.q: vec("3b101"),
        },
        70: {
            top.clock: F,
        },
        75: {
            top.clock: T,
            top.q: vec("3b010"),
        },
        80: {
            top.clock: F,
        },
        # Repeat cycle
        85: {
            top.clock: T,
            top.q: vec("3b100"),
        },
        90: {
            top.clock: F,
        },
        95: {
            top.clock: T,
            top.q: vec("3b001"),
        },
    }

    assert waves == exp
