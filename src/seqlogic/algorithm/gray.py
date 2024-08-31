"""Gray Code."""

from ..bits import Vector, cat, uxor


def bin2gray(b: Vector) -> Vector:
    """Convert binary to gray."""
    return cat(b[:-1] ^ b[1:], b[-1])


def gray2bin(g: Vector) -> Vector:
    """Convert gray to binary."""
    return cat(*[uxor(g[i:]) for i, _ in enumerate(g)])
