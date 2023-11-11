"""
Positional Cube (PC) Notation
"""

ZERO = 0b01
ONE = 0b10

from_int = (ZERO, ONE)

from_char = {
    "X": ZERO & ONE,
    "0": ZERO,
    "1": ONE,
    "x": ZERO | ONE,
}

to_char = {
    ZERO & ONE: "X",
    ZERO: "0",
    ONE: "1",
    ZERO | ONE: "x",
}


def getx(data: int, n: int) -> int:
    return (data >> (n << 1)) & 0b11


def setx(n: int, x: int) -> int:
    return x << (n << 1)
