"""Utility functions."""

from functools import cache


@cache
def clog2(x: int) -> int:
    """Return the ceiling log base two of an integer ≥ 1.

    This function tells you the minimum dimension of a Boolean space with at
    least N points.

    For example, here are the values of `clog2(N)` for `1 ≤ N < 18`:

    >>> [clog2(n) for n in range(1, 18)]
    [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5]

    This function is undefined for non-positive integers:

    >>> clog2(0)
    Traceback (most recent call last):
        ...
    ValueError: Expected x ≥ 1, got 0
    """
    if x < 1:
        raise ValueError(f"Expected x ≥ 1, got {x}")

    y = 0
    shifter = 1
    while x > shifter:
        shifter <<= 1
        y += 1
    return y


class classproperty:
    def __init__(self, func):
        self._f = func

    def __get__(self, unused_obj, cls):
        return self._f(cls)
