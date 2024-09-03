"""Kogge Stone Addition (KSA)."""

from ...bits import AddResult
from ...bits import Vector as Vec
from ...bits import cat
from ...util import clog2


def adc(a: Vec, b: Vec, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
    """Kogge Stone Addition."""
    n = len(a)
    assert n > 0 and n == len(b)

    # Generate / Propagate
    g = [a_i & b_i for a_i, b_i in zip(a, b)]
    p = [a_i | b_i for a_i, b_i in zip(a, b)]
    for i in range(clog2(n)):
        start = 1 << i
        for j in range(start, n):
            g[j] = g[j] | p[j] & g[j - start]
            p[j] = p[j] & p[j - start]

    # Carries
    c = [ci]
    for i in range(n):
        c.append(g[i] | c[i] & p[i])
    c, co = cat(*c[:n]), c[n]

    return AddResult(a ^ b ^ c, co)
