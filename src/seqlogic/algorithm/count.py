"""Bit Manipulation / Count."""

from seqlogic import Bits, Vec, cat, encode_onehot, pack


def clz(x: Bits) -> Vec:
    """Count leading zeros."""
    xr = pack(x)
    # Decode: {0000: 10000, 0001: 01000, ..., 01--: 00010, 1---: 00001}
    d = cat(xr, True) & -xr
    # Encode {10000: 100, 01000: 011, 00100: 010, 00010: 001, 00001: 000}
    return encode_onehot(d)
