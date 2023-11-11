"""
Positional Cube (PC) Notation
"""

ZERO = 0b01
ONE = 0b10

from_int = (ZERO, ONE)


def getx(data: int, n: int) -> int:
    return (data >> (n << 1)) & 0b11


def setx(n: int, x: int) -> int:
    return x << (n << 1)
