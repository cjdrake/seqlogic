"""Ripple Carry Addition (RCA)."""

from ...bits import Vector, cat


def add(a: Vector, b: Vector, ci: Vector[1]) -> tuple[Vector, Vector[1]]:
    """Ripple Carry Addition."""
    assert len(a) > 0 and len(a) == len(b)

    gen = zip(a, b)

    a_0, b_0 = next(gen)
    s = [a_0 ^ b_0 ^ ci]
    c = [a_0 & b_0 | a_0 & ci | b_0 & ci]

    for i, (a_i, b_i) in enumerate(gen, start=1):
        s.append(a_i ^ b_i ^ c[i - 1])
        c.append(a_i & b_i | c[i - 1] & (a_i | b_i))

    return cat(*s), c[-1]
