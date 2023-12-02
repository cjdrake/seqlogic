"""Gray Code."""

from ..logicvec import cat, logicvec, vec


def bin2gray(b: logicvec) -> logicvec:
    """Convert binary to gray."""
    b = b.flatten()
    return cat([b[:-1] ^ b[1:]] + [b[-1]])


def gray2bin(g: logicvec) -> logicvec:
    """Convert gray to binary."""
    g = g.flatten()
    return vec([g[i:].ulxor() for i, _ in enumerate(g)])
