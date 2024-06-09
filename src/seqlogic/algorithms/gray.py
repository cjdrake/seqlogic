"""Gray Code."""

from ..vec import Vec, cat


def bin2gray(b: Vec) -> Vec:
    """Convert binary to gray."""
    return cat(b[:-1] ^ b[1:], b[-1])


def gray2bin(g: Vec) -> Vec:
    """Convert gray to binary."""
    return cat(*[g[i:].uxor() for i, _ in enumerate(g)])
