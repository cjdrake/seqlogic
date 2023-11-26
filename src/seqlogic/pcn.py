"""Positional Cube (PC) Notation."""

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


def lnot(x: PcItem) -> PcItem:
    """Return output of "lifted" NOT function.

    f(x) -> y:
        N => N | 00 => 00
        0 => 1 | 01 => 10
        1 => 0 | 10 => 01
        X => X | 11 => 11
    """
    x_0 = x & 1
    x_1 = (x >> 1) & 1

    y_0, y_1 = x_1, x_0
    y = (y_1 << 1) | y_0

    return PcItem(y)


def lnor(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" NOR function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 1
        N X => N
        X 0 => X
        1 X => 1

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+
       01 |00|01|11|10|
          +--+--+--+--+
       11 |00|11|11|10|  y0 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       10 |00|10|10|10|     | x0[1] & x1[1]
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_0 & x1_1 | x0_1 & x1_0 | x0_1 & x1_1
    y_1 = x0_0 & x1_0
    y = (y_1 << 1) | y_0

    return PcItem(y)


def lor(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" OR function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 1
        N X => N
        X 0 => X
        1 X => 1

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       01 |00|01|11|10|     | x0[1] & x1[1]
          +--+--+--+--+
       11 |00|11|11|10|  y0 = x0[0] & x1[0]
          +--+--+--+--+
       10 |00|10|10|10|
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_0 & x1_0
    y_1 = x0_0 & x1_1 | x0_1 & x1_0 | x0_1 & x1_1
    y = (y_1 << 1) | y_0

    return PcItem(y)


def lnand(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" NAND function.

    f(x0, x1) -> y:
        0 0 => 1
        0 1 => 1
        1 0 => 1
        1 1 => 0
        N X => N
        0 X => 1
        X 1 => X

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+     | x0[0] & x1[1]
       01 |00|01|01|01|     | x0[1] & x1[0]
          +--+--+--+--+
       11 |00|01|11|11|  y0 = x0[1] & x1[1]
          +--+--+--+--+
       10 |00|01|11|10|
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_1 & x1_1
    y_1 = x0_0 & x1_0 | x0_0 & x1_1 | x0_1 & x1_0
    y = (y_1 << 1) | y_0

    return PcItem(y)


def land(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" AND function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 0
        1 0 => 0
        1 1 => 1
        N X => N
        0 X => 0
        X 1 => X

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[1] & x1[1]
          +--+--+--+--+
       01 |00|01|01|01|
          +--+--+--+--+
       11 |00|01|11|11|  y0 = x0[0] & x1[0]
          +--+--+--+--+     | x0[0] & x1[1]
       10 |00|01|11|10|     | x0[1] & x1[0]
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_0 & x1_0 | x0_0 & x1_1 | x0_1 & x1_0
    y_1 = x0_1 & x1_1
    y = (y_1 << 1) | y_0

    return PcItem(y)


def lxnor(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" XNOR function.

    f(x0, x1) -> y:
        0 0 => 1
        0 1 => 0
        1 0 => 0
        1 1 => 1
        N X => N
        X 0 => X
        X 1 => X

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+     | x0[1] & x1[1]
       01 |00|10|11|01|
          +--+--+--+--+
       11 |00|11|11|11|  y0 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       10 |00|01|11|10|
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_0 & x1_1 | x0_1 & x1_0
    y_1 = x0_0 & x1_0 | x0_1 & x1_1
    y = (y_1 << 1) | y_0

    return PcItem(y)


def lxor(x0: PcItem, x1: PcItem) -> PcItem:
    """Return output of "lifted" XOR function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 0
        N X => N
        X 0 => X
        X 1 => X

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       01 |00|01|11|10|
          +--+--+--+--+
       11 |00|11|11|11|  y0 = x0[0] & x1[0]
          +--+--+--+--+     | x0[1] & x1[1]
       10 |00|10|11|01|
          +--+--+--+--+
    """
    x0_0 = x0 & 1
    x0_1 = (x0 >> 1) & 1
    x1_0 = x1 & 1
    x1_1 = (x1 >> 1) & 1

    y_0 = x0_0 & x1_0 | x0_1 & x1_1
    y_1 = x0_0 & x1_1 | x0_1 & x1_0
    y = (y_1 << 1) | y_0

    return PcItem(y)


def limplies(p: PcItem, q: PcItem) -> PcItem:
    """Return output of "lifted" IMPLIES function.

    f(p, q) -> y:
        0 0 => 1
        0 1 => 1
        1 0 => 0
        1 1 => 1
        N X => N
        0 X => 1
        1 X => X
        X 0 => X
        X 1 => 1
        X X => X

           q
           00 01 11 10
          +--+--+--+--+
     p 00 |00|00|00|00|  y1 = p[0] & q[0]
          +--+--+--+--+     | p[0] & q[1]
       01 |00|10|10|10|     | p[1] & q[1]
          +--+--+--+--+
       11 |00|11|11|10|  y0 = p[1] & q[0]
          +--+--+--+--+
       10 |00|01|11|10|
          +--+--+--+--+
    """
    p_0 = p & 1
    p_1 = (p >> 1) & 1
    q_0 = q & 1
    q_1 = (q >> 1) & 1

    y_0 = p_1 & q_0
    y_1 = p_0 & q_0 | p_0 & q_1 | p_1 & q_1
    y = (y_1 << 1) | y_0

    return PcItem(y)


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


def set_item(data: PcList, n: int, x: PcItem) -> PcList:
    return PcList(data | (x << (n << 1)))


def set_byte(data: PcList, n: int, x: PcList) -> PcList:
    return PcList(data | (x << (8 * n)))


def zeros(n: int) -> PcList:
    data = PcList(0)
    for i in range(n):
        data = set_item(data, i, ZERO)
    return data


def ones(n: int) -> PcList:
    data = PcList(0)
    for i in range(n):
        data = set_item(data, i, ONE)
    return data
