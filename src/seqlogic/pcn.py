"""
Positional Cube (PC) Notation
"""

from typing import NewType

PcItem = NewType("PcItem", int)
PcList = NewType("PcList", int)

ZERO = PcItem(0b01)
ONE = PcItem(0b10)
NULL = PcItem(ZERO & ONE)
DC = PcItem(ZERO | ONE)  # "Don't Care"


from_int = (ZERO, ONE)

from_char = {
    "X": NULL,
    "0": ZERO,
    "1": ONE,
    "x": DC,
}

from_hexchar = {
    "0": PcList(0b01_01_01_01),
    "1": PcList(0b01_01_01_10),
    "2": PcList(0b01_01_10_01),
    "3": PcList(0b01_01_10_10),
    "4": PcList(0b01_10_01_01),
    "5": PcList(0b01_10_01_10),
    "6": PcList(0b01_10_10_01),
    "7": PcList(0b01_10_10_10),
    "8": PcList(0b10_01_01_01),
    "9": PcList(0b10_01_01_10),
    "a": PcList(0b10_01_10_01),
    "A": PcList(0b10_01_10_01),
    "b": PcList(0b10_01_10_10),
    "B": PcList(0b10_01_10_10),
    "c": PcList(0b10_10_01_01),
    "C": PcList(0b10_10_01_01),
    "d": PcList(0b10_10_01_10),
    "D": PcList(0b10_10_01_10),
    "e": PcList(0b10_10_10_01),
    "E": PcList(0b10_10_10_01),
    "f": PcList(0b10_10_10_10),
    "F": PcList(0b10_10_10_10),
}

to_char = {
    NULL: "X",
    ZERO: "0",
    ONE: "1",
    DC: "x",
}


def get_item(data: PcList, n: int) -> PcItem:
    return PcItem((data >> (n << 1)) & 0b11)


def set_item(n: int, x: PcItem) -> PcList:
    return PcList(x << (n << 1))
