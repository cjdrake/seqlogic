"""Example FSM implementation.

Demonstrate usage of an enum.
"""

from collections import defaultdict

from seqlogic import Bit, Enum, Module, get_loop
from seqlogic.lbool import VecEnum, ones, xes, zeros
from seqlogic.sim import Region

from .common import p_clk, p_dff, p_rst

# pyright: reportAttributeAccessIssue=false
# pyright: reportReturnType=false


loop = get_loop()


class SeqDetect(VecEnum):
    """Sequence Detector state."""

    A = "2b00"
    B = "2b01"
    C = "2b10"
    D = "2b11"


async def p_input(
    x: Bit,
    reset_n: Bit,
    clock: Bit,
):
    """TODO(cjdrake): Write docstring."""
    await reset_n.negedge()
    x.next = zeros(1)

    await reset_n.posedge()
    await clock.posedge()
    await clock.posedge()
    x.next = ones(1)  # A => B

    await clock.posedge()
    x.next = ones(1)  # B => C

    await clock.posedge()
    x.next = ones(1)  # C => D

    await clock.posedge()
    x.next = ones(1)  # D => D

    await clock.posedge()
    x.next = zeros(1)  # D => A


def test_fsm():
    """Test a 3-bit LFSR."""
    loop.reset()

    top = Module(name="top")
    clock = Bit(name="clock", parent=top)
    reset_n = Bit(name="reset_n", parent=top)
    ps = Enum(name="ps", parent=top, cls=SeqDetect)
    x = Bit(name="x", parent=top)

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
    loop.add_proc(Region(1), p_input, x, reset_n, clock)

    # Schedule LFSR
    loop.add_proc(Region(1), p_dff, ps, ns, reset_n, SeqDetect.A, clock)

    # Schedule reset and clock
    # Note: Avoiding simultaneous reset/clock negedge/posedge on purpose
    loop.add_proc(Region(2), p_rst, reset_n, init=ones(1), phase1=6, phase2=10)
    loop.add_proc(Region(2), p_clk, clock, init=zeros(1), shift=5, phase1=5, phase2=5)

    loop.run(until=100)

    exp = {
        # Initialize everything to X'es
        -1: {
            reset_n: xes(1),
            clock: xes(1),
            x: xes(1),
            ps: SeqDetect.X,
        },
        0: {
            reset_n: ones(1),
            clock: zeros(1),
        },
        # clock.posedge; reset_n = 1
        5: {
            clock: ones(1),
        },
        # reset_n.negedge
        6: {
            reset_n: zeros(1),
            x: zeros(1),
            ps: SeqDetect.A,
        },
        10: {
            clock: zeros(1),
        },
        # clock.posedge; reset_n = 0
        15: {
            clock: ones(1),
        },
        # reset_n.posedge
        16: {
            reset_n: ones(1),
        },
        20: {
            clock: zeros(1),
        },
        # clock.posedge; reset_n = 1
        25: {
            clock: ones(1),
        },
        30: {
            clock: zeros(1),
        },
        # clock.posedge
        35: {
            clock: ones(1),
            x: ones(1),
        },
        40: {
            clock: zeros(1),
        },
        # clock.posedge
        45: {
            clock: ones(1),
            ps: SeqDetect.B,
        },
        50: {
            clock: zeros(1),
        },
        # clock.posedge
        55: {
            clock: ones(1),
            ps: SeqDetect.C,
        },
        60: {
            clock: zeros(1),
        },
        # clock.posedge
        65: {
            clock: ones(1),
            ps: SeqDetect.D,
        },
        70: {
            clock: zeros(1),
        },
        # clock.posedge
        75: {
            clock: ones(1),
            x: zeros(1),
            # ps: D => D
        },
        80: {
            clock: zeros(1),
        },
        # clock.posedge
        85: {
            clock: ones(1),
            ps: SeqDetect.A,
        },
        90: {
            clock: zeros(1),
        },
        # clock.posedge
        95: {
            clock: ones(1),
        },
    }

    assert waves == exp
