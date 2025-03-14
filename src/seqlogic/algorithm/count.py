"""Bit Manipulation / Count."""

from bvwx import Bits, Vec, cat, encode_onehot, ngc, pack


def clz(x: Bits) -> Vec:
    """Count leading zeros."""
    return ctz(pack(x))


def ctz(x: Bits) -> Vec:
    """Count trailing zeros."""
    # Decode: {0000: 10000, 0001: 01000, ..., 01--: 00010, 1---: 00001}
    d = cat(x, True) & ngc(x)
    # Encode {10000: 100, 01000: 011, 00100: 010, 00010: 001, 00001: 000}
    return encode_onehot(d)
