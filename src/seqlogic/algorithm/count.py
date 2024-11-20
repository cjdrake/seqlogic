"""Bit Manipulation / Count."""

from ..bits import Bits, Empty, Scalar, Vector, and_, cat, or_, pack, rep, u2bv
from ..util import clog2


def clz(x: Bits) -> Empty | Scalar | Vector:
    """Count leading zeros."""
    xr = pack(x)
    # Decode: {0000: 10000, 0001: 01000, ..., 01--: 00010, 1---: 00001}
    d = cat(xr, True) & -xr
    # Encode {10000: 100, 01000: 011, 00100: 010, 00010: 001, 00001: 000}
    n = clog2(d.size)
    xs = [and_(rep(d[i], n), u2bv(i, n)) for i in range(d.size)]
    return or_(*xs)
