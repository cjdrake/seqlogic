"""Gray Code."""

from ..bits import Bits, bits, cat


def bin2gray(b: Bits) -> Bits:
    """Convert binary to gray."""
    b = b.flatten()
    return cat([b[:-1] ^ b[1:]] + [b[-1]])


def gray2bin(g: Bits) -> Bits:
    """Convert gray to binary."""
    g = g.flatten()
    return bits([g[i:].ulxor() for i, _ in enumerate(g)])
