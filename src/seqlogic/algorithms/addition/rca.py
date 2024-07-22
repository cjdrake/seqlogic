"""Ripple Carry Addition (RCA)."""

from ...bits import Vector as Vec
from ...bits import cat


def add(a: Vec, b: Vec, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
    """Ripple Carry Addition."""
    assert len(a) > 0 and len(a) == len(b)

    gen = zip(a, b)

    a_0, b_0 = next(gen)
    s = [a_0 ^ b_0 ^ ci]
    c = [a_0 & b_0 | a_0 & ci | b_0 & ci]

    for i, (a_i, b_i) in enumerate(gen, start=1):
        s.append(a_i ^ b_i ^ c[i - 1])
        c.append(a_i & b_i | a_i & c[i - 1] | b_i & c[i - 1])

    return cat(*s), c[-1]
