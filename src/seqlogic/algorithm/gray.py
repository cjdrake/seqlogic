"""Gray Code."""

from bvwx import Array, cat, uxor


def bin2gray(b: Array) -> Array:
    """Convert binary to gray."""
    return cat(b[:-1] ^ b[1:], b[-1])


def gray2bin(g: Array) -> Array:
    """Convert gray to binary."""
    return cat(*[uxor(g[i:]) for i, _ in enumerate(g)])
