"""Example LFSR implementation."""


from seqlogic.logic import logic
from seqlogic.logicvec import cat, logicvec, vec
from seqlogic.sim import Region, get_loop

from .common import _TraceVar, _TraceVec, clock_drv, dff_arn_drv, reset_drv, waves

loop = get_loop()


def test_lfsr():
    """Test a 3-bit LFSR."""
    loop.reset()
    waves.clear()

    # State Variables
    q = _TraceVec(3)

    def d() -> logicvec:
        v: logicvec = q.value
        return cat([v[0] ^ v[2], v[:2]])

    # Control Variables
    reset_n = _TraceVar()
    reset_value = vec("3b100")
    clock = _TraceVar()

    # Schedule LFSR
    loop.add_proc(dff_arn_drv, Region(0), q, d, reset_n, reset_value, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(reset_drv, Region(1), reset_n, init=logic.T, phase1_ticks=6, phase2_ticks=10)
    loop.add_proc(
        clock_drv, Region(1), clock, init=logic.F, shift_ticks=5, phase1_ticks=5, phase2_ticks=5
    )

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: logic.X,
            clock: logic.X,
            q: vec("3bxxx"),
        },
        0: {
            reset_n: logic.T,
            clock: logic.F,
        },
        # clock.posedge; reset_n = 1
        # q = xxx
        5: {
            clock: logic.T,
        },
        # reset_n.negedge
        # q = reset_value
        6: {
            reset_n: logic.F,
            q: vec("3b100"),
        },
        10: {
            clock: logic.F,
        },
        # clock.posedge; reset_n = 0
        15: {
            clock: logic.T,
        },
        # reset_n.posedge
        16: {
            reset_n: logic.T,
        },
        20: {
            clock: logic.F,
        },
        # clock.posedge; reset_n = 1
        # q = 001
        25: {
            clock: logic.T,
            q: vec("3b001"),
        },
        30: {
            clock: logic.F,
        },
        35: {
            clock: logic.T,
            q: vec("3b011"),
        },
        40: {
            clock: logic.F,
        },
        45: {
            clock: logic.T,
            q: vec("3b111"),
        },
        50: {
            clock: logic.F,
        },
        55: {
            clock: logic.T,
            q: vec("3b110"),
        },
        60: {
            clock: logic.F,
        },
        65: {
            clock: logic.T,
            q: vec("3b101"),
        },
        70: {
            clock: logic.F,
        },
        75: {
            clock: logic.T,
            q: vec("3b010"),
        },
        80: {
            clock: logic.F,
        },
        # Repeat cycle
        85: {
            clock: logic.T,
            q: vec("3b100"),
        },
        90: {
            clock: logic.F,
        },
        95: {
            clock: logic.T,
            q: vec("3b001"),
        },
    }

    assert waves == exp
