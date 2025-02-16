"""Gray Code."""

from bvwx import Vec, cat, uxor


def bin2gray(b: Vec) -> Vec:
    """Convert binary to gray."""
    return cat(b[:-1] ^ b[1:], b[-1])


def gray2bin(g: Vec) -> Vec:
    """Convert gray to binary."""
    return cat(*[uxor(g[i:]) for i, _ in enumerate(g)])
