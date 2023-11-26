"""Utility Functions."""


def get_bit(x: int, n: int) -> bool:
    """Return the nth bit of x as a bool.

    >>> get_bit(2, 0)
    False
    >>> get_bit(2, 1)
    True
    """
    return bool((x >> n) & 1)


def bools2int(*xs: bool) -> int:
    """Convert a tuple of bools to an int.

    >>> bools2int()
    0
    >>> bools2int(False, True, False, True, False, True)
    42
    """
    y = 0
    for i, x in enumerate(xs):
        y |= x << i
    return y


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
