"""Common code.

For now this is just used for testing.
It might be useful to add to seqlogic library.
"""

from collections import defaultdict

from seqlogic import resume

# [Time][Var] = Val
waves = defaultdict(dict)


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
