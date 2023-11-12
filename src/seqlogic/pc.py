"""
Positional Cube (PC) Notation
"""

ZERO = 0b01
ONE = 0b10
NULL = ZERO & ONE
DC = ZERO | ONE  # "Don't Care"


from_int = (ZERO, ONE)

from_char = {
    "X": NULL,
    "0": ZERO,
    "1": ONE,
    "x": DC,
}

to_char = {
    NULL: "X",
    ZERO: "0",
    ONE: "1",
    DC: "x",
}


def getx(data: int, n: int) -> int:
    return (data >> (n << 1)) & 0b11


def setx(n: int, x: int) -> int:
    return x << (n << 1)
