"""Gray Code."""

from ..bits import bits, cat, foo


def bin2gray(b: bits) -> bits:
    """Convert binary to gray."""
    b = b.flatten()
    return cat([b[:-1] ^ b[1:]] + [b[-1]])


def gray2bin(g: bits) -> bits:
    """Convert gray to binary."""
    g = g.flatten()
    return foo([g[i:].ulxor() for i, _ in enumerate(g)])
