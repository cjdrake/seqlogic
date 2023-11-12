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

from_hexchar = {
    "0": 0x55,
    "1": 0x56,
    "2": 0x59,
    "3": 0x5A,
    "4": 0x65,
    "5": 0x66,
    "6": 0x69,
    "7": 0x6A,
    "8": 0x95,
    "9": 0x96,
    "a": 0x99,
    "b": 0x9A,
    "c": 0xA5,
    "d": 0xA6,
    "e": 0xA9,
    "f": 0xAA,
    "A": 0x99,
    "B": 0x9A,
    "C": 0xA5,
    "D": 0xA6,
    "E": 0xA9,
    "F": 0xAA,
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
