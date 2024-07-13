"""Common code.

For now this is just used for testing.
It might be useful to add to seqlogic library.
"""

from collections import defaultdict

from seqlogic import resume, sleep

# [Time][Var] = Val
waves = defaultdict(dict)


async def p_rst(reset, init, phase1: int = 1, phase2: int = 1):
    r"""
    Simulate a reset signal.

    ‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾

    <phase1> <phase2>

    Args:
        phase1: Number of sim ticks in ...
        phase2: Number of sim ticks in ...

    Raises:
        ValueError: An argument has the correct type, but an incorrect value
    """
    if phase1 < 0:
        raise ValueError(f"Expected phase1 ≥ 0, got {phase1}")
    if phase2 < 0:
        raise ValueError(f"Expected phase2 ≥ 0, got {phase2}")

    # T = 0
    reset.next = init
    await sleep(phase1)

    reset.next = ~reset.value
    await sleep(phase2)

    reset.next = ~reset.value


async def p_clk(clock, init, shift: int = 0, phase1: int = 1, phase2: int = 1):
    r"""
    Simulate a clock signal.

    ________/‾‾‾‾‾‾‾‾\\________/‾‾‾‾‾‾‾‾\\________

    <shift> <phase1> <phase2>

    Args:
        init: Initial clock signal value
        shift: Number of sim ticks to shift the first clock transition
        phase1: Number of sim ticks in the clock's active phase
        phase2: Number of sim ticks in the clock's passive phase

    The period is phase1 + phase2
    The duty cycle is phase1 / (phase1 + phase2)

    Raises:
        ValueError: An argument has the correct type, but an incorrect value
    """
    if shift < 0:
        raise ValueError(f"Expected shift ≥ 0, got {shift}")
    if phase1 < 0:
        raise ValueError(f"Expected phase1 ≥ 0, got {phase1}")
    if phase2 < 0:
        raise ValueError(f"Expected phase2 ≥ 0, got {phase2}")

    # T = 0
    clock.next = init
    await sleep(shift)

    while True:
        clock.next = ~clock.value
        await sleep(phase1)

        clock.next = ~clock.value
        await sleep(phase2)


async def p_dff(q, d, clock, reset_n, reset_value):
    """D Flop Flop with asynchronous, negedge-triggered reset."""
    while True:
        state = await resume(
            (reset_n, reset_n.is_negedge),
            (clock, lambda: clock.is_posedge() and reset_n.is_pos()),
        )
        if state is reset_n:
            q.next = reset_value
        elif state is clock:
            q.next = d()
        else:
            assert False
