"""Positional Cube (PC) Notation."""

# pylint: disable = protected-access

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import NewType

PcItem = NewType("PcItem", int)

_ITEM_BITS = 2
_ITEM_MASK = 0b11

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

to_char = {
    NULL: "X",
    ZERO: "0",
    ONE: "1",
    DC: "x",
}

from_hexchar = {
    "0": 0b01_01_01_01,
    "1": 0b01_01_01_10,
    "2": 0b01_01_10_01,
    "3": 0b01_01_10_10,
    "4": 0b01_10_01_01,
    "5": 0b01_10_01_10,
    "6": 0b01_10_10_01,
    "7": 0b01_10_10_10,
    "8": 0b10_01_01_01,
    "9": 0b10_01_01_10,
    "a": 0b10_01_10_01,
    "A": 0b10_01_10_01,
    "b": 0b10_01_10_10,
    "B": 0b10_01_10_10,
    "c": 0b10_10_01_01,
    "C": 0b10_10_01_01,
    "d": 0b10_10_01_10,
    "D": 0b10_10_01_10,
    "e": 0b10_10_10_01,
    "E": 0b10_10_10_01,
    "f": 0b10_10_10_10,
    "F": 0b10_10_10_10,
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
    x_1 = x >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    x0_1 = x0 >> 1
    x1_0 = x1 & 1
    x1_1 = x1 >> 1

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
    p_1 = p >> 1
    q_0 = q & 1
    q_1 = q >> 1

    y_0 = p_1 & q_0
    y_1 = p_0 & q_0 | p_0 & q_1 | p_1 & q_1
    y = (y_1 << 1) | y_0

    return PcItem(y)


class Cube:
    """N-dimensional cube."""

    def __init__(self, n: int, data: int):
        """TODO(cjdrake): Write docstring."""
        self._n = n
        assert 0 <= data < (1 << self.nbits)
        self._data = data

    def __getitem__(self, key: int) -> Cube:
        match key:
            case int() as i:
                i = self._norm_index(i)
                d = (self._data >> (_ITEM_BITS * i)) & _ITEM_MASK
                return Cube(1, d)
            case _:
                raise TypeError("Expected key to be an int")

    def __len__(self) -> int:
        return self._n

    @property
    def data(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._data

    @cached_property
    def nbits(self) -> int:
        """Number of bits of data."""
        return _ITEM_BITS * self._n

    def lnot(self) -> Cube:
        """Return output of "lifted" NOT function."""
        x_0 = self._bit_mask[0]
        x_01 = x_0 << 1
        x_1 = self._bit_mask[1]
        x_10 = x_1 >> 1

        y0 = x_10
        y1 = x_01
        y = y1 | y0

        return Cube(self._n, y)

    def lnor(self, other: Cube) -> Cube:
        """Return output of "lifted" NOR function.

        y1 = x0[0] & x1[0]
        y0 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_01
        y = y1 | y0

        return Cube(self._n, y)

    def lor(self, other: Cube) -> Cube:
        """Return output of "lifted" OR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[0]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]

        y0 = x0_0 & x1_0
        y1 = x0_01 & x1_1 | x0_1 & x1_01 | x0_1 & x1_1
        y = y1 | y0

        return Cube(self._n, y)

    def ulor(self) -> Cube:
        """Return unary "lifted" OR of bits."""
        y = Cube(1, 0b01)
        for x in self:
            y = y.lor(x)
        return y

    def lnand(self, other: Cube) -> Cube:
        """Return output of "lifted" NAND function.

        y1 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[1] & x1[1]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_10 & x1_10
        y1 = x0_01 & x1_01 | x0_01 & x1_1 | x0_1 & x1_01
        y = y1 | y0

        return Cube(self._n, y)

    def land(self, other: Cube) -> Cube:
        """Return output of "lifted" AND function.

        y1 = x0[1] & x1[1]
        y0 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_1 & x1_1
        y = y1 | y0

        return Cube(self._n, y)

    def uland(self) -> Cube:
        """Return unary "lifted" AND of bits."""
        y = Cube(1, 0b10)
        for x in self:
            y = y.land(x)
        return y

    def lxnor(self, other: Cube) -> Cube:
        """Return output of "lifted" XNOR function.

        y1 = x0[0] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[1] | x0[1] & x1[0]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_01 & x1_01 | x0_1 & x1_1
        y = y1 | y0

        return Cube(self._n, y)

    def lxor(self, other: Cube) -> Cube:
        """Return output of "lifted" XOR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[0] & x1[0] | x0[1] & x1[1]
        """
        assert self._n == len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_1 | x0_1 & x1_01
        y = y1 | y0

        return Cube(self._n, y)

    def ulxor(self) -> Cube:
        """Return unary "lifted" XOR of bits."""
        y = Cube(1, 0b01)
        for x in self:
            y = y.lxor(x)
        return y

    def _norm_index(self, index: int) -> int:
        lo, hi = -self._n, self._n
        if not lo <= index < hi:
            s = f"Expected index in [{lo}, {hi}), got {index}"
            raise IndexError(s)
        # Normalize negative start index
        if index < 0:
            index += hi
        return index

    @cached_property
    def _mask(self) -> tuple[int, int]:
        """Return PC zero/one mask.

        The zero mask is: 0b01010101...
        The one mask is:  0b10101010...

        N
        1          01 =  1 = (  4-1)/3
        2        0101 =  5 = ( 16-1)/3
        3      010101 = 21 = ( 64-1)/3
        4    01010101 = 85 = (256-1)/3
        ...

        F(0) = 1
        F(n+1) = 4*F(n) + 1
        F(n) = (4^n - 1) / 3
        """
        zero_mask = 0
        for i in range(self._n):
            zero_mask |= ZERO << (_ITEM_BITS * i)
        one_mask = zero_mask << 1
        return zero_mask, one_mask

    @cached_property
    def _bit_mask(self) -> tuple[int, int]:
        return self._data & self._mask[0], self._data & self._mask[1]


def from_pcitems(xs: Iterable[PcItem] = ()) -> Cube:
    """Convert an iterable of PcItems to a PcList."""
    size = 0
    data = 0
    for i, x in enumerate(xs):
        size += 1
        data |= x << (_ITEM_BITS * i)
    return Cube(size, data)


def from_quads(xs: Iterable[int] = ()) -> Cube:
    """Convert an iterable of bytes (four PcItems each) to a PcList."""
    size = 0
    data = 0
    for i, x in enumerate(xs):
        size += 4
        data |= x << (4 * _ITEM_BITS * i)
    return Cube(size, data)
