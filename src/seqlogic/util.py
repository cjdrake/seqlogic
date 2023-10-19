"""
Utility Functions
"""


def get_bit(x: int, n: int) -> bool:
    """Return the nth bit of x as a bool."""
    return bool((x >> n) & 1)


def bools2int(*xs: bool) -> int:
    """Convert a tuple of bools to an int."""
    y = 0
    for i, x in enumerate(xs):
        y |= x << i
    return y
