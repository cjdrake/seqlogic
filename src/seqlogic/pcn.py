"""Positional Cube (PC) Notation."""

# pylint: disable = protected-access

from __future__ import annotations

from collections.abc import Generator, Iterable
from functools import cached_property

from .util import clog2

_ITEM_BITS = 2
_ITEM_MASK = 0b11

NULL = 0b00
ZERO = 0b01
ONE = 0b10
DC = 0b11  # "Don't Care"


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


def lnot(x: int) -> int:
    """Lifted NOT function.

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

    return y


def lnor(x0: int, x1: int) -> int:
    """Lifted NOR function.

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

    return y


def lor(x0: int, x1: int) -> int:
    """Lifted OR function.

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

    return y


def lnand(x0: int, x1: int) -> int:
    """Lifted NAND function.

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

    return y


def land(x0: int, x1: int) -> int:
    """Lifted AND function.

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

    return y


def lxnor(x0: int, x1: int) -> int:
    """Lifted XNOR function.

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

    return y


def lxor(x0: int, x1: int) -> int:
    """Lifted XOR function.

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

    return y


def limplies(p: int, q: int) -> int:
    """Lifted IMPLIES function.

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

    return y


class PcVec:
    """TODO(cjdrake): Write docstring."""

    def __class_getitem__(cls, key: int):
        pass  # pragma: no cover

    def __init__(self, n: int, data: int):
        """TODO(cjdrake): Write docstring."""
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        self._n = n

        a, b = 0, 1 << self.nbits
        if not a <= data < b:
            raise ValueError(f"Expected data in [{a}, {b}), got {data}")
        self._data = data

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: int | slice) -> PcVec:
        match key:
            case int() as i:
                d = self._get_item(self._norm_index(i))
                return PcVec(1, d)
            case slice() as sl:
                n, d = self._get_items(*self._norm_slice(sl))
                return PcVec(n, d)
            case _:
                raise TypeError("Expected key to be int or slice")

    def __iter__(self) -> Generator[PcVec[1], None, None]:
        for i in range(self._n):
            yield self.__getitem__(i)

    def __str__(self) -> str:
        return "".join(to_char[self._get_item(i)] for i in range(self._n))

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self.to_uint() != 0

    def __int__(self) -> int:
        return self.to_int()

    # Comparison
    def _eq(self, other: PcVec) -> bool:
        return self._n == other._n and self._data == other._data

    def __eq__(self, other) -> bool:
        match other:
            case PcVec():
                return self._eq(other)
            case _:
                return False

    def __hash__(self):
        return hash(self._n) ^ hash(self._data)

    # Bitwise Arithmetic
    def __invert__(self) -> PcVec:
        return self.lnot()

    def __or__(self, other: PcVec) -> PcVec:
        return self.lor(other)

    def __and__(self, other: PcVec) -> PcVec:
        return self.land(other)

    def __xor__(self, other: PcVec) -> PcVec:
        return self.lxor(other)

    def __lshift__(self, n: int) -> PcVec:
        return self.lsh(n)[0]

    def __rshift__(self, n: int) -> PcVec:
        return self.rsh(n)[0]

    def __add__(self, other: PcVec) -> PcVec:
        return self.add(other, ci=F)[0]

    def __sub__(self, other: PcVec) -> PcVec:
        return self.sub(other)[0]

    def __neg__(self) -> PcVec:
        return self.neg()[0]

    @property
    def data(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._data

    @cached_property
    def nbits(self) -> int:
        """Number of bits of data."""
        return _ITEM_BITS * self._n

    def lnot(self) -> PcVec:
        """Return output of "lifted" NOT function."""
        x_0 = self._bit_mask[0]
        x_01 = x_0 << 1
        x_1 = self._bit_mask[1]
        x_10 = x_1 >> 1

        y0 = x_10
        y1 = x_01
        y = y1 | y0

        return PcVec(self._n, y)

    def lnor(self, other: PcVec) -> PcVec:
        """Return output of "lifted" NOR function.

        y1 = x0[0] & x1[0]
        y0 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        """
        self._check_len(other)

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

        return PcVec(self._n, y)

    def lor(self, other: PcVec) -> PcVec:
        """Return output of "lifted" OR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[0]
        """
        self._check_len(other)

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]

        x1_0 = other._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = other._bit_mask[1]

        y0 = x0_0 & x1_0
        y1 = x0_01 & x1_1 | x0_1 & x1_01 | x0_1 & x1_1
        y = y1 | y0

        return PcVec(self._n, y)

    def ulor(self) -> PcVec[1]:
        """Return unary "lifted" OR of bits."""
        y = F
        for x in self:
            y = y.lor(x)
        return y

    def lnand(self, other: PcVec) -> PcVec:
        """Return output of "lifted" NAND function.

        y1 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[1] & x1[1]
        """
        self._check_len(other)

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

        return PcVec(self._n, y)

    def land(self, other: PcVec) -> PcVec:
        """Return output of "lifted" AND function.

        y1 = x0[1] & x1[1]
        y0 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        """
        self._check_len(other)

        x0_0 = self._bit_mask[0]
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = other._bit_mask[0]
        x1_1 = other._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_1 & x1_1
        y = y1 | y0

        return PcVec(self._n, y)

    def uland(self) -> PcVec[1]:
        """Return unary "lifted" AND of bits."""
        y = T
        for x in self:
            y = y.land(x)
        return y

    def lxnor(self, other: PcVec) -> PcVec:
        """Return output of "lifted" XNOR function.

        y1 = x0[0] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[1] | x0[1] & x1[0]
        """
        self._check_len(other)

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

        return PcVec(self._n, y)

    def lxor(self, other: PcVec) -> PcVec:
        """Return output of "lifted" XOR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[0] & x1[0] | x0[1] & x1[1]
        """
        self._check_len(other)

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

        return PcVec(self._n, y)

    def ulxor(self) -> PcVec[1]:
        """Return unary "lifted" XOR of bits."""
        y = F
        for x in self:
            y = y.lxor(x)
        return y

    def to_uint(self) -> int:
        """Convert to unsigned integer."""
        y = 0

        if self.has_unknown():
            raise ValueError("Cannot convert unknown to uint")

        i, data = 0, self._data
        while i <= (self._n - 8):
            y |= _wyde_uint[data & _WYDE_MASK] << i
            data >>= 16
            i += 8
        while i <= (self._n - 4):
            y |= _byte_uint[data & _BYTE_MASK] << i
            data >>= 8
            i += 4
        while i <= (self._n - 1):
            y |= _item_uint[data & _ITEM_MASK] << i
            data >>= 2
            i += 1

        return y

    def to_int(self) -> int:
        """Convert to signed integer."""
        if self._n == 0:
            return 0
        sign = self._get_item(self._n - 1)
        if sign == ONE:
            return -(self.lnot().to_uint() + 1)
        return self.to_uint()

    def ult(self, other: PcVec) -> bool:
        """Unsigned less than."""
        self._check_len(other)
        return self.to_uint() < other.to_uint()

    def slt(self, other: PcVec) -> bool:
        """Signed less than."""
        self._check_len(other)
        return self.to_int() < other.to_int()

    def zext(self, n: int) -> PcVec:
        """Zero extend by n bits."""
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if n == 0:
            return self
        prefix = _fill(ZERO, n)
        return PcVec(self._n + n, self._data | (prefix << self.nbits))

    def sext(self, n: int) -> PcVec:
        """Sign extend by n bits."""
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if n == 0:
            return self
        sign = self._get_item(self._n - 1)
        prefix = _fill(sign, n)
        return PcVec(self._n + n, self._data | (prefix << self.nbits))

    def lsh(self, n: int, ci: PcVec[1] | None = None) -> tuple[PcVec, PcVec]:
        """Left shift by n bits."""
        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, E
        if ci is None:
            ci = PcVec(n, _fill(ZERO, n))
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")
        sh, co = self[:-n], self[-n:]
        y = PcVec(self._n, ci._data | (sh._data << ci.nbits))
        return y, co

    def rsh(self, n: int, ci: PcVec[1] | None = None) -> tuple[PcVec, PcVec]:
        """Right shift by n bits."""
        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, E
        if ci is None:
            ci = PcVec(n, _fill(ZERO, n))
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")
        co, sh = self[:n], self[n:]
        y = PcVec(self._n, sh._data | (ci._data << sh.nbits))
        return y, co

    def arsh(self, n: int) -> tuple[PcVec, PcVec]:
        """Arithmetically right shift by n bits."""
        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, E
        sign = self._get_item(self._n - 1)
        ci_data = _fill(sign, n)
        co, sh = self[:n], self[n:]
        y = PcVec(self._n, sh._data | (ci_data << sh.nbits))
        return y, co

    def add(self, other: PcVec, ci: PcVec[1]) -> tuple[PcVec, PcVec[1], PcVec[1]]:
        """Twos complement addition."""
        # Rename for readability
        n, a, b = self._n, self, other

        if a.has_null() or b.has_null() or ci.has_null():
            return PcVec(n, _fill(NULL, n)), N, N
        if a.has_dc() or b.has_dc() or ci.has_dc():
            return PcVec(n, _fill(DC, n)), X, X

        s = a.to_uint() + b.to_uint() + ci.to_uint()

        data = 0
        for i in range(n):
            data |= from_int[s & 1] << (2 * i)
            s >>= 1

        # Carry out is True if there is leftover sum data
        co = (F, T)[s != 0]

        s = PcVec(n, data)

        # Overflow is true if sign A matches sign B, and mismatches sign S
        aa = a[-1]
        bb = b[-1]
        ss = s[-1]
        ovf = ~aa & ~bb & ss | aa & bb & ~ss

        return s, co, ovf

    def sub(self, other: PcVec) -> tuple[PcVec, PcVec[1], PcVec[1]]:
        """Twos complement subtraction."""
        return self.add(other.lnot(), ci=T)

    def neg(self) -> tuple[PcVec, PcVec[1], PcVec[1]]:
        """Twos complement negation."""
        zero = PcVec(self._n, _fill(ZERO, self._n))
        return zero.sub(self)

    def _check_len(self, other: PcVec):
        if self._n != other._n:
            s = f"Expected n = {self._n}, got {other._n}"
            raise ValueError(s)

    def _count(self, byte_cnt: dict[int, int], item: int) -> int:
        y = 0

        n, data = self._n, self._data
        while n >= 4:
            y += byte_cnt[data & _BYTE_MASK]
            n -= 4
            data >>= 8
        while n >= 1:
            y += (data & _ITEM_MASK) == item
            n -= 1
            data >>= 2

        return y

    def count_nulls(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._count(_byte_cnt_nulls, NULL)

    def count_zeros(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._count(_byte_cnt_zeros, ZERO)

    def count_ones(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._count(_byte_cnt_ones, ONE)

    def count_dcs(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self._count(_byte_cnt_dcs, DC)

    def count_known(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self.count_zeros() + self.count_ones()

    def count_unknown(self) -> int:
        """TODO(cjdrake): Write docstring."""
        return self.count_nulls() + self.count_dcs()

    def onehot(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return not self.has_unknown() and self.count_ones() == 1

    def onehot0(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return not self.has_unknown() and self.count_ones() <= 1

    def has_null(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self.count_nulls() != 0

    def has_dc(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self.count_dcs() != 0

    def has_unknown(self) -> bool:
        """TODO(cjdrake): Write docstring."""
        return self.has_null() or self.has_dc()

    def _norm_index(self, index: int) -> int:
        a, b = -self._n, self._n
        if not a <= index < b:
            s = f"Expected index in [{a}, {b}), got {index}"
            raise IndexError(s)
        # Normalize negative start index
        if index < 0:
            return index + self._n
        return index

    def _norm_slice(self, sl: slice) -> tuple[int, int]:
        if sl.step is not None:
            raise ValueError("Slice step is not supported")
        a, b = -self._n, self._n
        # Normalize start index
        start = sl.start
        if start is None or start < a:
            start = a
        if start < 0:
            start += self._n
        # Normalize stop index
        stop = sl.stop
        if stop is None or stop > b:
            stop = b
        if stop < 0:
            stop += self._n
        return start, stop

    def _get_item(self, i: int) -> int:
        return (self._data >> (_ITEM_BITS * i)) & _ITEM_MASK

    def _get_items(self, i: int, j: int) -> tuple[int, int]:
        n = j - i
        nbits = _ITEM_BITS * n
        mask = (1 << nbits) - 1
        return n, (self._data >> (_ITEM_BITS * i)) & mask

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
        zero_mask = _fill(ZERO, self._n)
        one_mask = zero_mask << 1
        return zero_mask, one_mask

    @cached_property
    def _bit_mask(self) -> tuple[int, int]:
        return self._data & self._mask[0], self._data & self._mask[1]


def _fill(x: int, n: int) -> int:
    """TODO(cjdrake): Write docstring."""
    data = 0
    for i in range(n):
        data |= x << (_ITEM_BITS * i)
    return data


def uint2vec(num: int, n: int | None = None) -> PcVec:
    """Convert nonnegative int to PcVec.

    Args:
        num: A nonnegative integer
        n: Optional output length

    Returns:
        A PcVec instance

    Raises:
        ValueError: If num is negative or overflows the output length.
    """
    if num < 0:
        raise ValueError(f"Expected num ≥ 0, got {num}")

    data = 0
    i = 0
    r = num

    while r >= 1:
        data |= from_int[r & 1] << (_ITEM_BITS * i)
        i += 1
        r >>= 1

    # Compute required number of bits
    req_n = clog2(num + 1)
    if n is None:
        n = req_n
    elif n < req_n:
        s = f"Overflow: num = {num} required n ≥ {req_n}, got {n}"
        raise ValueError(s)

    return PcVec(i, data).zext(n - i)


def int2vec(num: int, n: int | None = None) -> PcVec:
    """Convert int to PcVec.

    Args:
        num: An integer
        n: Optional output length

    Returns:
        A PcVec instance

    Raises:
        ValueError: If num overflows the output length.
    """
    neg = num < 0

    data = 0
    i = 0
    r = abs(num)

    while r >= 1:
        data |= from_int[r & 1] << (_ITEM_BITS * i)
        i += 1
        r >>= 1

    # Compute required number of bits
    if neg:
        req_n = clog2(-num) + 1
    else:
        req_n = clog2(num + 1) + 1
    if n is None:
        n = req_n
    elif n < req_n:
        s = f"Overflow: num = {num} required n ≥ {req_n}, got {n}"
        raise ValueError(s)

    v = PcVec(i, data).zext(n - i)
    return v.neg()[0] if neg else v


def from_pcitems(xs: Iterable[int] = ()) -> PcVec:
    """Convert an iterable of PcItems to a PcVec."""
    n = 0
    data = 0
    for i, x in enumerate(xs):
        n += 1
        data |= x << (_ITEM_BITS * i)
    return PcVec(n, data)


def from_quads(xs: Iterable[int] = ()) -> PcVec:
    """Convert an iterable of bytes (four PcItems each) to a PcVec."""
    n = 0
    data = 0
    for i, x in enumerate(xs):
        n += 4
        data |= x << (4 * _ITEM_BITS * i)
    return PcVec(n, data)


# Empty
E = PcVec(0, 0)

# One bit values
N = PcVec(1, NULL)
F = PcVec(1, ZERO)
T = PcVec(1, ONE)
X = PcVec(1, DC)

_byte_cnt_nulls = {
    0b00_00_00_00: 4,
    0b00_00_00_01: 3,
    0b00_00_00_10: 3,
    0b00_00_00_11: 3,
    0b00_00_01_00: 3,
    0b00_00_01_01: 2,
    0b00_00_01_10: 2,
    0b00_00_01_11: 2,
    0b00_00_10_00: 3,
    0b00_00_10_01: 2,
    0b00_00_10_10: 2,
    0b00_00_10_11: 2,
    0b00_00_11_00: 3,
    0b00_00_11_01: 2,
    0b00_00_11_10: 2,
    0b00_00_11_11: 2,
    0b00_01_00_00: 3,
    0b00_01_00_01: 2,
    0b00_01_00_10: 2,
    0b00_01_00_11: 2,
    0b00_01_01_00: 2,
    0b00_01_01_01: 1,
    0b00_01_01_10: 1,
    0b00_01_01_11: 1,
    0b00_01_10_00: 2,
    0b00_01_10_01: 1,
    0b00_01_10_10: 1,
    0b00_01_10_11: 1,
    0b00_01_11_00: 2,
    0b00_01_11_01: 1,
    0b00_01_11_10: 1,
    0b00_01_11_11: 1,
    0b00_10_00_00: 3,
    0b00_10_00_01: 2,
    0b00_10_00_10: 2,
    0b00_10_00_11: 2,
    0b00_10_01_00: 2,
    0b00_10_01_01: 1,
    0b00_10_01_10: 1,
    0b00_10_01_11: 1,
    0b00_10_10_00: 2,
    0b00_10_10_01: 1,
    0b00_10_10_10: 1,
    0b00_10_10_11: 1,
    0b00_10_11_00: 2,
    0b00_10_11_01: 1,
    0b00_10_11_10: 1,
    0b00_10_11_11: 1,
    0b00_11_00_00: 3,
    0b00_11_00_01: 2,
    0b00_11_00_10: 2,
    0b00_11_00_11: 2,
    0b00_11_01_00: 2,
    0b00_11_01_01: 1,
    0b00_11_01_10: 1,
    0b00_11_01_11: 1,
    0b00_11_10_00: 2,
    0b00_11_10_01: 1,
    0b00_11_10_10: 1,
    0b00_11_10_11: 1,
    0b00_11_11_00: 2,
    0b00_11_11_01: 1,
    0b00_11_11_10: 1,
    0b00_11_11_11: 1,
    0b01_00_00_00: 3,
    0b01_00_00_01: 2,
    0b01_00_00_10: 2,
    0b01_00_00_11: 2,
    0b01_00_01_00: 2,
    0b01_00_01_01: 1,
    0b01_00_01_10: 1,
    0b01_00_01_11: 1,
    0b01_00_10_00: 2,
    0b01_00_10_01: 1,
    0b01_00_10_10: 1,
    0b01_00_10_11: 1,
    0b01_00_11_00: 2,
    0b01_00_11_01: 1,
    0b01_00_11_10: 1,
    0b01_00_11_11: 1,
    0b01_01_00_00: 2,
    0b01_01_00_01: 1,
    0b01_01_00_10: 1,
    0b01_01_00_11: 1,
    0b01_01_01_00: 1,
    0b01_01_01_01: 0,
    0b01_01_01_10: 0,
    0b01_01_01_11: 0,
    0b01_01_10_00: 1,
    0b01_01_10_01: 0,
    0b01_01_10_10: 0,
    0b01_01_10_11: 0,
    0b01_01_11_00: 1,
    0b01_01_11_01: 0,
    0b01_01_11_10: 0,
    0b01_01_11_11: 0,
    0b01_10_00_00: 2,
    0b01_10_00_01: 1,
    0b01_10_00_10: 1,
    0b01_10_00_11: 1,
    0b01_10_01_00: 1,
    0b01_10_01_01: 0,
    0b01_10_01_10: 0,
    0b01_10_01_11: 0,
    0b01_10_10_00: 1,
    0b01_10_10_01: 0,
    0b01_10_10_10: 0,
    0b01_10_10_11: 0,
    0b01_10_11_00: 1,
    0b01_10_11_01: 0,
    0b01_10_11_10: 0,
    0b01_10_11_11: 0,
    0b01_11_00_00: 2,
    0b01_11_00_01: 1,
    0b01_11_00_10: 1,
    0b01_11_00_11: 1,
    0b01_11_01_00: 1,
    0b01_11_01_01: 0,
    0b01_11_01_10: 0,
    0b01_11_01_11: 0,
    0b01_11_10_00: 1,
    0b01_11_10_01: 0,
    0b01_11_10_10: 0,
    0b01_11_10_11: 0,
    0b01_11_11_00: 1,
    0b01_11_11_01: 0,
    0b01_11_11_10: 0,
    0b01_11_11_11: 0,
    0b10_00_00_00: 3,
    0b10_00_00_01: 2,
    0b10_00_00_10: 2,
    0b10_00_00_11: 2,
    0b10_00_01_00: 2,
    0b10_00_01_01: 1,
    0b10_00_01_10: 1,
    0b10_00_01_11: 1,
    0b10_00_10_00: 2,
    0b10_00_10_01: 1,
    0b10_00_10_10: 1,
    0b10_00_10_11: 1,
    0b10_00_11_00: 2,
    0b10_00_11_01: 1,
    0b10_00_11_10: 1,
    0b10_00_11_11: 1,
    0b10_01_00_00: 2,
    0b10_01_00_01: 1,
    0b10_01_00_10: 1,
    0b10_01_00_11: 1,
    0b10_01_01_00: 1,
    0b10_01_01_01: 0,
    0b10_01_01_10: 0,
    0b10_01_01_11: 0,
    0b10_01_10_00: 1,
    0b10_01_10_01: 0,
    0b10_01_10_10: 0,
    0b10_01_10_11: 0,
    0b10_01_11_00: 1,
    0b10_01_11_01: 0,
    0b10_01_11_10: 0,
    0b10_01_11_11: 0,
    0b10_10_00_00: 2,
    0b10_10_00_01: 1,
    0b10_10_00_10: 1,
    0b10_10_00_11: 1,
    0b10_10_01_00: 1,
    0b10_10_01_01: 0,
    0b10_10_01_10: 0,
    0b10_10_01_11: 0,
    0b10_10_10_00: 1,
    0b10_10_10_01: 0,
    0b10_10_10_10: 0,
    0b10_10_10_11: 0,
    0b10_10_11_00: 1,
    0b10_10_11_01: 0,
    0b10_10_11_10: 0,
    0b10_10_11_11: 0,
    0b10_11_00_00: 2,
    0b10_11_00_01: 1,
    0b10_11_00_10: 1,
    0b10_11_00_11: 1,
    0b10_11_01_00: 1,
    0b10_11_01_01: 0,
    0b10_11_01_10: 0,
    0b10_11_01_11: 0,
    0b10_11_10_00: 1,
    0b10_11_10_01: 0,
    0b10_11_10_10: 0,
    0b10_11_10_11: 0,
    0b10_11_11_00: 1,
    0b10_11_11_01: 0,
    0b10_11_11_10: 0,
    0b10_11_11_11: 0,
    0b11_00_00_00: 3,
    0b11_00_00_01: 2,
    0b11_00_00_10: 2,
    0b11_00_00_11: 2,
    0b11_00_01_00: 2,
    0b11_00_01_01: 1,
    0b11_00_01_10: 1,
    0b11_00_01_11: 1,
    0b11_00_10_00: 2,
    0b11_00_10_01: 1,
    0b11_00_10_10: 1,
    0b11_00_10_11: 1,
    0b11_00_11_00: 2,
    0b11_00_11_01: 1,
    0b11_00_11_10: 1,
    0b11_00_11_11: 1,
    0b11_01_00_00: 2,
    0b11_01_00_01: 1,
    0b11_01_00_10: 1,
    0b11_01_00_11: 1,
    0b11_01_01_00: 1,
    0b11_01_01_01: 0,
    0b11_01_01_10: 0,
    0b11_01_01_11: 0,
    0b11_01_10_00: 1,
    0b11_01_10_01: 0,
    0b11_01_10_10: 0,
    0b11_01_10_11: 0,
    0b11_01_11_00: 1,
    0b11_01_11_01: 0,
    0b11_01_11_10: 0,
    0b11_01_11_11: 0,
    0b11_10_00_00: 2,
    0b11_10_00_01: 1,
    0b11_10_00_10: 1,
    0b11_10_00_11: 1,
    0b11_10_01_00: 1,
    0b11_10_01_01: 0,
    0b11_10_01_10: 0,
    0b11_10_01_11: 0,
    0b11_10_10_00: 1,
    0b11_10_10_01: 0,
    0b11_10_10_10: 0,
    0b11_10_10_11: 0,
    0b11_10_11_00: 1,
    0b11_10_11_01: 0,
    0b11_10_11_10: 0,
    0b11_10_11_11: 0,
    0b11_11_00_00: 2,
    0b11_11_00_01: 1,
    0b11_11_00_10: 1,
    0b11_11_00_11: 1,
    0b11_11_01_00: 1,
    0b11_11_01_01: 0,
    0b11_11_01_10: 0,
    0b11_11_01_11: 0,
    0b11_11_10_00: 1,
    0b11_11_10_01: 0,
    0b11_11_10_10: 0,
    0b11_11_10_11: 0,
    0b11_11_11_00: 1,
    0b11_11_11_01: 0,
    0b11_11_11_10: 0,
    0b11_11_11_11: 0,
}

_byte_cnt_zeros = {
    0b00_00_00_00: 0,
    0b00_00_00_01: 1,
    0b00_00_00_10: 0,
    0b00_00_00_11: 0,
    0b00_00_01_00: 1,
    0b00_00_01_01: 2,
    0b00_00_01_10: 1,
    0b00_00_01_11: 1,
    0b00_00_10_00: 0,
    0b00_00_10_01: 1,
    0b00_00_10_10: 0,
    0b00_00_10_11: 0,
    0b00_00_11_00: 0,
    0b00_00_11_01: 1,
    0b00_00_11_10: 0,
    0b00_00_11_11: 0,
    0b00_01_00_00: 1,
    0b00_01_00_01: 2,
    0b00_01_00_10: 1,
    0b00_01_00_11: 1,
    0b00_01_01_00: 2,
    0b00_01_01_01: 3,
    0b00_01_01_10: 2,
    0b00_01_01_11: 2,
    0b00_01_10_00: 1,
    0b00_01_10_01: 2,
    0b00_01_10_10: 1,
    0b00_01_10_11: 1,
    0b00_01_11_00: 1,
    0b00_01_11_01: 2,
    0b00_01_11_10: 1,
    0b00_01_11_11: 1,
    0b00_10_00_00: 0,
    0b00_10_00_01: 1,
    0b00_10_00_10: 0,
    0b00_10_00_11: 0,
    0b00_10_01_00: 1,
    0b00_10_01_01: 2,
    0b00_10_01_10: 1,
    0b00_10_01_11: 1,
    0b00_10_10_00: 0,
    0b00_10_10_01: 1,
    0b00_10_10_10: 0,
    0b00_10_10_11: 0,
    0b00_10_11_00: 0,
    0b00_10_11_01: 1,
    0b00_10_11_10: 0,
    0b00_10_11_11: 0,
    0b00_11_00_00: 0,
    0b00_11_00_01: 1,
    0b00_11_00_10: 0,
    0b00_11_00_11: 0,
    0b00_11_01_00: 1,
    0b00_11_01_01: 2,
    0b00_11_01_10: 1,
    0b00_11_01_11: 1,
    0b00_11_10_00: 0,
    0b00_11_10_01: 1,
    0b00_11_10_10: 0,
    0b00_11_10_11: 0,
    0b00_11_11_00: 0,
    0b00_11_11_01: 1,
    0b00_11_11_10: 0,
    0b00_11_11_11: 0,
    0b01_00_00_00: 1,
    0b01_00_00_01: 2,
    0b01_00_00_10: 1,
    0b01_00_00_11: 1,
    0b01_00_01_00: 2,
    0b01_00_01_01: 3,
    0b01_00_01_10: 2,
    0b01_00_01_11: 2,
    0b01_00_10_00: 1,
    0b01_00_10_01: 2,
    0b01_00_10_10: 1,
    0b01_00_10_11: 1,
    0b01_00_11_00: 1,
    0b01_00_11_01: 2,
    0b01_00_11_10: 1,
    0b01_00_11_11: 1,
    0b01_01_00_00: 2,
    0b01_01_00_01: 3,
    0b01_01_00_10: 2,
    0b01_01_00_11: 2,
    0b01_01_01_00: 3,
    0b01_01_01_01: 4,
    0b01_01_01_10: 3,
    0b01_01_01_11: 3,
    0b01_01_10_00: 2,
    0b01_01_10_01: 3,
    0b01_01_10_10: 2,
    0b01_01_10_11: 2,
    0b01_01_11_00: 2,
    0b01_01_11_01: 3,
    0b01_01_11_10: 2,
    0b01_01_11_11: 2,
    0b01_10_00_00: 1,
    0b01_10_00_01: 2,
    0b01_10_00_10: 1,
    0b01_10_00_11: 1,
    0b01_10_01_00: 2,
    0b01_10_01_01: 3,
    0b01_10_01_10: 2,
    0b01_10_01_11: 2,
    0b01_10_10_00: 1,
    0b01_10_10_01: 2,
    0b01_10_10_10: 1,
    0b01_10_10_11: 1,
    0b01_10_11_00: 1,
    0b01_10_11_01: 2,
    0b01_10_11_10: 1,
    0b01_10_11_11: 1,
    0b01_11_00_00: 1,
    0b01_11_00_01: 2,
    0b01_11_00_10: 1,
    0b01_11_00_11: 1,
    0b01_11_01_00: 2,
    0b01_11_01_01: 3,
    0b01_11_01_10: 2,
    0b01_11_01_11: 2,
    0b01_11_10_00: 1,
    0b01_11_10_01: 2,
    0b01_11_10_10: 1,
    0b01_11_10_11: 1,
    0b01_11_11_00: 1,
    0b01_11_11_01: 2,
    0b01_11_11_10: 1,
    0b01_11_11_11: 1,
    0b10_00_00_00: 0,
    0b10_00_00_01: 1,
    0b10_00_00_10: 0,
    0b10_00_00_11: 0,
    0b10_00_01_00: 1,
    0b10_00_01_01: 2,
    0b10_00_01_10: 1,
    0b10_00_01_11: 1,
    0b10_00_10_00: 0,
    0b10_00_10_01: 1,
    0b10_00_10_10: 0,
    0b10_00_10_11: 0,
    0b10_00_11_00: 0,
    0b10_00_11_01: 1,
    0b10_00_11_10: 0,
    0b10_00_11_11: 0,
    0b10_01_00_00: 1,
    0b10_01_00_01: 2,
    0b10_01_00_10: 1,
    0b10_01_00_11: 1,
    0b10_01_01_00: 2,
    0b10_01_01_01: 3,
    0b10_01_01_10: 2,
    0b10_01_01_11: 2,
    0b10_01_10_00: 1,
    0b10_01_10_01: 2,
    0b10_01_10_10: 1,
    0b10_01_10_11: 1,
    0b10_01_11_00: 1,
    0b10_01_11_01: 2,
    0b10_01_11_10: 1,
    0b10_01_11_11: 1,
    0b10_10_00_00: 0,
    0b10_10_00_01: 1,
    0b10_10_00_10: 0,
    0b10_10_00_11: 0,
    0b10_10_01_00: 1,
    0b10_10_01_01: 2,
    0b10_10_01_10: 1,
    0b10_10_01_11: 1,
    0b10_10_10_00: 0,
    0b10_10_10_01: 1,
    0b10_10_10_10: 0,
    0b10_10_10_11: 0,
    0b10_10_11_00: 0,
    0b10_10_11_01: 1,
    0b10_10_11_10: 0,
    0b10_10_11_11: 0,
    0b10_11_00_00: 0,
    0b10_11_00_01: 1,
    0b10_11_00_10: 0,
    0b10_11_00_11: 0,
    0b10_11_01_00: 1,
    0b10_11_01_01: 2,
    0b10_11_01_10: 1,
    0b10_11_01_11: 1,
    0b10_11_10_00: 0,
    0b10_11_10_01: 1,
    0b10_11_10_10: 0,
    0b10_11_10_11: 0,
    0b10_11_11_00: 0,
    0b10_11_11_01: 1,
    0b10_11_11_10: 0,
    0b10_11_11_11: 0,
    0b11_00_00_00: 0,
    0b11_00_00_01: 1,
    0b11_00_00_10: 0,
    0b11_00_00_11: 0,
    0b11_00_01_00: 1,
    0b11_00_01_01: 2,
    0b11_00_01_10: 1,
    0b11_00_01_11: 1,
    0b11_00_10_00: 0,
    0b11_00_10_01: 1,
    0b11_00_10_10: 0,
    0b11_00_10_11: 0,
    0b11_00_11_00: 0,
    0b11_00_11_01: 1,
    0b11_00_11_10: 0,
    0b11_00_11_11: 0,
    0b11_01_00_00: 1,
    0b11_01_00_01: 2,
    0b11_01_00_10: 1,
    0b11_01_00_11: 1,
    0b11_01_01_00: 2,
    0b11_01_01_01: 3,
    0b11_01_01_10: 2,
    0b11_01_01_11: 2,
    0b11_01_10_00: 1,
    0b11_01_10_01: 2,
    0b11_01_10_10: 1,
    0b11_01_10_11: 1,
    0b11_01_11_00: 1,
    0b11_01_11_01: 2,
    0b11_01_11_10: 1,
    0b11_01_11_11: 1,
    0b11_10_00_00: 0,
    0b11_10_00_01: 1,
    0b11_10_00_10: 0,
    0b11_10_00_11: 0,
    0b11_10_01_00: 1,
    0b11_10_01_01: 2,
    0b11_10_01_10: 1,
    0b11_10_01_11: 1,
    0b11_10_10_00: 0,
    0b11_10_10_01: 1,
    0b11_10_10_10: 0,
    0b11_10_10_11: 0,
    0b11_10_11_00: 0,
    0b11_10_11_01: 1,
    0b11_10_11_10: 0,
    0b11_10_11_11: 0,
    0b11_11_00_00: 0,
    0b11_11_00_01: 1,
    0b11_11_00_10: 0,
    0b11_11_00_11: 0,
    0b11_11_01_00: 1,
    0b11_11_01_01: 2,
    0b11_11_01_10: 1,
    0b11_11_01_11: 1,
    0b11_11_10_00: 0,
    0b11_11_10_01: 1,
    0b11_11_10_10: 0,
    0b11_11_10_11: 0,
    0b11_11_11_00: 0,
    0b11_11_11_01: 1,
    0b11_11_11_10: 0,
    0b11_11_11_11: 0,
}

_byte_cnt_ones = {
    0b00_00_00_00: 0,
    0b00_00_00_01: 0,
    0b00_00_00_10: 1,
    0b00_00_00_11: 0,
    0b00_00_01_00: 0,
    0b00_00_01_01: 0,
    0b00_00_01_10: 1,
    0b00_00_01_11: 0,
    0b00_00_10_00: 1,
    0b00_00_10_01: 1,
    0b00_00_10_10: 2,
    0b00_00_10_11: 1,
    0b00_00_11_00: 0,
    0b00_00_11_01: 0,
    0b00_00_11_10: 1,
    0b00_00_11_11: 0,
    0b00_01_00_00: 0,
    0b00_01_00_01: 0,
    0b00_01_00_10: 1,
    0b00_01_00_11: 0,
    0b00_01_01_00: 0,
    0b00_01_01_01: 0,
    0b00_01_01_10: 1,
    0b00_01_01_11: 0,
    0b00_01_10_00: 1,
    0b00_01_10_01: 1,
    0b00_01_10_10: 2,
    0b00_01_10_11: 1,
    0b00_01_11_00: 0,
    0b00_01_11_01: 0,
    0b00_01_11_10: 1,
    0b00_01_11_11: 0,
    0b00_10_00_00: 1,
    0b00_10_00_01: 1,
    0b00_10_00_10: 2,
    0b00_10_00_11: 1,
    0b00_10_01_00: 1,
    0b00_10_01_01: 1,
    0b00_10_01_10: 2,
    0b00_10_01_11: 1,
    0b00_10_10_00: 2,
    0b00_10_10_01: 2,
    0b00_10_10_10: 3,
    0b00_10_10_11: 2,
    0b00_10_11_00: 1,
    0b00_10_11_01: 1,
    0b00_10_11_10: 2,
    0b00_10_11_11: 1,
    0b00_11_00_00: 0,
    0b00_11_00_01: 0,
    0b00_11_00_10: 1,
    0b00_11_00_11: 0,
    0b00_11_01_00: 0,
    0b00_11_01_01: 0,
    0b00_11_01_10: 1,
    0b00_11_01_11: 0,
    0b00_11_10_00: 1,
    0b00_11_10_01: 1,
    0b00_11_10_10: 2,
    0b00_11_10_11: 1,
    0b00_11_11_00: 0,
    0b00_11_11_01: 0,
    0b00_11_11_10: 1,
    0b00_11_11_11: 0,
    0b01_00_00_00: 0,
    0b01_00_00_01: 0,
    0b01_00_00_10: 1,
    0b01_00_00_11: 0,
    0b01_00_01_00: 0,
    0b01_00_01_01: 0,
    0b01_00_01_10: 1,
    0b01_00_01_11: 0,
    0b01_00_10_00: 1,
    0b01_00_10_01: 1,
    0b01_00_10_10: 2,
    0b01_00_10_11: 1,
    0b01_00_11_00: 0,
    0b01_00_11_01: 0,
    0b01_00_11_10: 1,
    0b01_00_11_11: 0,
    0b01_01_00_00: 0,
    0b01_01_00_01: 0,
    0b01_01_00_10: 1,
    0b01_01_00_11: 0,
    0b01_01_01_00: 0,
    0b01_01_01_01: 0,
    0b01_01_01_10: 1,
    0b01_01_01_11: 0,
    0b01_01_10_00: 1,
    0b01_01_10_01: 1,
    0b01_01_10_10: 2,
    0b01_01_10_11: 1,
    0b01_01_11_00: 0,
    0b01_01_11_01: 0,
    0b01_01_11_10: 1,
    0b01_01_11_11: 0,
    0b01_10_00_00: 1,
    0b01_10_00_01: 1,
    0b01_10_00_10: 2,
    0b01_10_00_11: 1,
    0b01_10_01_00: 1,
    0b01_10_01_01: 1,
    0b01_10_01_10: 2,
    0b01_10_01_11: 1,
    0b01_10_10_00: 2,
    0b01_10_10_01: 2,
    0b01_10_10_10: 3,
    0b01_10_10_11: 2,
    0b01_10_11_00: 1,
    0b01_10_11_01: 1,
    0b01_10_11_10: 2,
    0b01_10_11_11: 1,
    0b01_11_00_00: 0,
    0b01_11_00_01: 0,
    0b01_11_00_10: 1,
    0b01_11_00_11: 0,
    0b01_11_01_00: 0,
    0b01_11_01_01: 0,
    0b01_11_01_10: 1,
    0b01_11_01_11: 0,
    0b01_11_10_00: 1,
    0b01_11_10_01: 1,
    0b01_11_10_10: 2,
    0b01_11_10_11: 1,
    0b01_11_11_00: 0,
    0b01_11_11_01: 0,
    0b01_11_11_10: 1,
    0b01_11_11_11: 0,
    0b10_00_00_00: 1,
    0b10_00_00_01: 1,
    0b10_00_00_10: 2,
    0b10_00_00_11: 1,
    0b10_00_01_00: 1,
    0b10_00_01_01: 1,
    0b10_00_01_10: 2,
    0b10_00_01_11: 1,
    0b10_00_10_00: 2,
    0b10_00_10_01: 2,
    0b10_00_10_10: 3,
    0b10_00_10_11: 2,
    0b10_00_11_00: 1,
    0b10_00_11_01: 1,
    0b10_00_11_10: 2,
    0b10_00_11_11: 1,
    0b10_01_00_00: 1,
    0b10_01_00_01: 1,
    0b10_01_00_10: 2,
    0b10_01_00_11: 1,
    0b10_01_01_00: 1,
    0b10_01_01_01: 1,
    0b10_01_01_10: 2,
    0b10_01_01_11: 1,
    0b10_01_10_00: 2,
    0b10_01_10_01: 2,
    0b10_01_10_10: 3,
    0b10_01_10_11: 2,
    0b10_01_11_00: 1,
    0b10_01_11_01: 1,
    0b10_01_11_10: 2,
    0b10_01_11_11: 1,
    0b10_10_00_00: 2,
    0b10_10_00_01: 2,
    0b10_10_00_10: 3,
    0b10_10_00_11: 2,
    0b10_10_01_00: 2,
    0b10_10_01_01: 2,
    0b10_10_01_10: 3,
    0b10_10_01_11: 2,
    0b10_10_10_00: 3,
    0b10_10_10_01: 3,
    0b10_10_10_10: 4,
    0b10_10_10_11: 3,
    0b10_10_11_00: 2,
    0b10_10_11_01: 2,
    0b10_10_11_10: 3,
    0b10_10_11_11: 2,
    0b10_11_00_00: 1,
    0b10_11_00_01: 1,
    0b10_11_00_10: 2,
    0b10_11_00_11: 1,
    0b10_11_01_00: 1,
    0b10_11_01_01: 1,
    0b10_11_01_10: 2,
    0b10_11_01_11: 1,
    0b10_11_10_00: 2,
    0b10_11_10_01: 2,
    0b10_11_10_10: 3,
    0b10_11_10_11: 2,
    0b10_11_11_00: 1,
    0b10_11_11_01: 1,
    0b10_11_11_10: 2,
    0b10_11_11_11: 1,
    0b11_00_00_00: 0,
    0b11_00_00_01: 0,
    0b11_00_00_10: 1,
    0b11_00_00_11: 0,
    0b11_00_01_00: 0,
    0b11_00_01_01: 0,
    0b11_00_01_10: 1,
    0b11_00_01_11: 0,
    0b11_00_10_00: 1,
    0b11_00_10_01: 1,
    0b11_00_10_10: 2,
    0b11_00_10_11: 1,
    0b11_00_11_00: 0,
    0b11_00_11_01: 0,
    0b11_00_11_10: 1,
    0b11_00_11_11: 0,
    0b11_01_00_00: 0,
    0b11_01_00_01: 0,
    0b11_01_00_10: 1,
    0b11_01_00_11: 0,
    0b11_01_01_00: 0,
    0b11_01_01_01: 0,
    0b11_01_01_10: 1,
    0b11_01_01_11: 0,
    0b11_01_10_00: 1,
    0b11_01_10_01: 1,
    0b11_01_10_10: 2,
    0b11_01_10_11: 1,
    0b11_01_11_00: 0,
    0b11_01_11_01: 0,
    0b11_01_11_10: 1,
    0b11_01_11_11: 0,
    0b11_10_00_00: 1,
    0b11_10_00_01: 1,
    0b11_10_00_10: 2,
    0b11_10_00_11: 1,
    0b11_10_01_00: 1,
    0b11_10_01_01: 1,
    0b11_10_01_10: 2,
    0b11_10_01_11: 1,
    0b11_10_10_00: 2,
    0b11_10_10_01: 2,
    0b11_10_10_10: 3,
    0b11_10_10_11: 2,
    0b11_10_11_00: 1,
    0b11_10_11_01: 1,
    0b11_10_11_10: 2,
    0b11_10_11_11: 1,
    0b11_11_00_00: 0,
    0b11_11_00_01: 0,
    0b11_11_00_10: 1,
    0b11_11_00_11: 0,
    0b11_11_01_00: 0,
    0b11_11_01_01: 0,
    0b11_11_01_10: 1,
    0b11_11_01_11: 0,
    0b11_11_10_00: 1,
    0b11_11_10_01: 1,
    0b11_11_10_10: 2,
    0b11_11_10_11: 1,
    0b11_11_11_00: 0,
    0b11_11_11_01: 0,
    0b11_11_11_10: 1,
    0b11_11_11_11: 0,
}

_byte_cnt_dcs = {
    0b00_00_00_00: 0,
    0b00_00_00_01: 0,
    0b00_00_00_10: 0,
    0b00_00_00_11: 1,
    0b00_00_01_00: 0,
    0b00_00_01_01: 0,
    0b00_00_01_10: 0,
    0b00_00_01_11: 1,
    0b00_00_10_00: 0,
    0b00_00_10_01: 0,
    0b00_00_10_10: 0,
    0b00_00_10_11: 1,
    0b00_00_11_00: 1,
    0b00_00_11_01: 1,
    0b00_00_11_10: 1,
    0b00_00_11_11: 2,
    0b00_01_00_00: 0,
    0b00_01_00_01: 0,
    0b00_01_00_10: 0,
    0b00_01_00_11: 1,
    0b00_01_01_00: 0,
    0b00_01_01_01: 0,
    0b00_01_01_10: 0,
    0b00_01_01_11: 1,
    0b00_01_10_00: 0,
    0b00_01_10_01: 0,
    0b00_01_10_10: 0,
    0b00_01_10_11: 1,
    0b00_01_11_00: 1,
    0b00_01_11_01: 1,
    0b00_01_11_10: 1,
    0b00_01_11_11: 2,
    0b00_10_00_00: 0,
    0b00_10_00_01: 0,
    0b00_10_00_10: 0,
    0b00_10_00_11: 1,
    0b00_10_01_00: 0,
    0b00_10_01_01: 0,
    0b00_10_01_10: 0,
    0b00_10_01_11: 1,
    0b00_10_10_00: 0,
    0b00_10_10_01: 0,
    0b00_10_10_10: 0,
    0b00_10_10_11: 1,
    0b00_10_11_00: 1,
    0b00_10_11_01: 1,
    0b00_10_11_10: 1,
    0b00_10_11_11: 2,
    0b00_11_00_00: 1,
    0b00_11_00_01: 1,
    0b00_11_00_10: 1,
    0b00_11_00_11: 2,
    0b00_11_01_00: 1,
    0b00_11_01_01: 1,
    0b00_11_01_10: 1,
    0b00_11_01_11: 2,
    0b00_11_10_00: 1,
    0b00_11_10_01: 1,
    0b00_11_10_10: 1,
    0b00_11_10_11: 2,
    0b00_11_11_00: 2,
    0b00_11_11_01: 2,
    0b00_11_11_10: 2,
    0b00_11_11_11: 3,
    0b01_00_00_00: 0,
    0b01_00_00_01: 0,
    0b01_00_00_10: 0,
    0b01_00_00_11: 1,
    0b01_00_01_00: 0,
    0b01_00_01_01: 0,
    0b01_00_01_10: 0,
    0b01_00_01_11: 1,
    0b01_00_10_00: 0,
    0b01_00_10_01: 0,
    0b01_00_10_10: 0,
    0b01_00_10_11: 1,
    0b01_00_11_00: 1,
    0b01_00_11_01: 1,
    0b01_00_11_10: 1,
    0b01_00_11_11: 2,
    0b01_01_00_00: 0,
    0b01_01_00_01: 0,
    0b01_01_00_10: 0,
    0b01_01_00_11: 1,
    0b01_01_01_00: 0,
    0b01_01_01_01: 0,
    0b01_01_01_10: 0,
    0b01_01_01_11: 1,
    0b01_01_10_00: 0,
    0b01_01_10_01: 0,
    0b01_01_10_10: 0,
    0b01_01_10_11: 1,
    0b01_01_11_00: 1,
    0b01_01_11_01: 1,
    0b01_01_11_10: 1,
    0b01_01_11_11: 2,
    0b01_10_00_00: 0,
    0b01_10_00_01: 0,
    0b01_10_00_10: 0,
    0b01_10_00_11: 1,
    0b01_10_01_00: 0,
    0b01_10_01_01: 0,
    0b01_10_01_10: 0,
    0b01_10_01_11: 1,
    0b01_10_10_00: 0,
    0b01_10_10_01: 0,
    0b01_10_10_10: 0,
    0b01_10_10_11: 1,
    0b01_10_11_00: 1,
    0b01_10_11_01: 1,
    0b01_10_11_10: 1,
    0b01_10_11_11: 2,
    0b01_11_00_00: 1,
    0b01_11_00_01: 1,
    0b01_11_00_10: 1,
    0b01_11_00_11: 2,
    0b01_11_01_00: 1,
    0b01_11_01_01: 1,
    0b01_11_01_10: 1,
    0b01_11_01_11: 2,
    0b01_11_10_00: 1,
    0b01_11_10_01: 1,
    0b01_11_10_10: 1,
    0b01_11_10_11: 2,
    0b01_11_11_00: 2,
    0b01_11_11_01: 2,
    0b01_11_11_10: 2,
    0b01_11_11_11: 3,
    0b10_00_00_00: 0,
    0b10_00_00_01: 0,
    0b10_00_00_10: 0,
    0b10_00_00_11: 1,
    0b10_00_01_00: 0,
    0b10_00_01_01: 0,
    0b10_00_01_10: 0,
    0b10_00_01_11: 1,
    0b10_00_10_00: 0,
    0b10_00_10_01: 0,
    0b10_00_10_10: 0,
    0b10_00_10_11: 1,
    0b10_00_11_00: 1,
    0b10_00_11_01: 1,
    0b10_00_11_10: 1,
    0b10_00_11_11: 2,
    0b10_01_00_00: 0,
    0b10_01_00_01: 0,
    0b10_01_00_10: 0,
    0b10_01_00_11: 1,
    0b10_01_01_00: 0,
    0b10_01_01_01: 0,
    0b10_01_01_10: 0,
    0b10_01_01_11: 1,
    0b10_01_10_00: 0,
    0b10_01_10_01: 0,
    0b10_01_10_10: 0,
    0b10_01_10_11: 1,
    0b10_01_11_00: 1,
    0b10_01_11_01: 1,
    0b10_01_11_10: 1,
    0b10_01_11_11: 2,
    0b10_10_00_00: 0,
    0b10_10_00_01: 0,
    0b10_10_00_10: 0,
    0b10_10_00_11: 1,
    0b10_10_01_00: 0,
    0b10_10_01_01: 0,
    0b10_10_01_10: 0,
    0b10_10_01_11: 1,
    0b10_10_10_00: 0,
    0b10_10_10_01: 0,
    0b10_10_10_10: 0,
    0b10_10_10_11: 1,
    0b10_10_11_00: 1,
    0b10_10_11_01: 1,
    0b10_10_11_10: 1,
    0b10_10_11_11: 2,
    0b10_11_00_00: 1,
    0b10_11_00_01: 1,
    0b10_11_00_10: 1,
    0b10_11_00_11: 2,
    0b10_11_01_00: 1,
    0b10_11_01_01: 1,
    0b10_11_01_10: 1,
    0b10_11_01_11: 2,
    0b10_11_10_00: 1,
    0b10_11_10_01: 1,
    0b10_11_10_10: 1,
    0b10_11_10_11: 2,
    0b10_11_11_00: 2,
    0b10_11_11_01: 2,
    0b10_11_11_10: 2,
    0b10_11_11_11: 3,
    0b11_00_00_00: 1,
    0b11_00_00_01: 1,
    0b11_00_00_10: 1,
    0b11_00_00_11: 2,
    0b11_00_01_00: 1,
    0b11_00_01_01: 1,
    0b11_00_01_10: 1,
    0b11_00_01_11: 2,
    0b11_00_10_00: 1,
    0b11_00_10_01: 1,
    0b11_00_10_10: 1,
    0b11_00_10_11: 2,
    0b11_00_11_00: 2,
    0b11_00_11_01: 2,
    0b11_00_11_10: 2,
    0b11_00_11_11: 3,
    0b11_01_00_00: 1,
    0b11_01_00_01: 1,
    0b11_01_00_10: 1,
    0b11_01_00_11: 2,
    0b11_01_01_00: 1,
    0b11_01_01_01: 1,
    0b11_01_01_10: 1,
    0b11_01_01_11: 2,
    0b11_01_10_00: 1,
    0b11_01_10_01: 1,
    0b11_01_10_10: 1,
    0b11_01_10_11: 2,
    0b11_01_11_00: 2,
    0b11_01_11_01: 2,
    0b11_01_11_10: 2,
    0b11_01_11_11: 3,
    0b11_10_00_00: 1,
    0b11_10_00_01: 1,
    0b11_10_00_10: 1,
    0b11_10_00_11: 2,
    0b11_10_01_00: 1,
    0b11_10_01_01: 1,
    0b11_10_01_10: 1,
    0b11_10_01_11: 2,
    0b11_10_10_00: 1,
    0b11_10_10_01: 1,
    0b11_10_10_10: 1,
    0b11_10_10_11: 2,
    0b11_10_11_00: 2,
    0b11_10_11_01: 2,
    0b11_10_11_10: 2,
    0b11_10_11_11: 3,
    0b11_11_00_00: 2,
    0b11_11_00_01: 2,
    0b11_11_00_10: 2,
    0b11_11_00_11: 3,
    0b11_11_01_00: 2,
    0b11_11_01_01: 2,
    0b11_11_01_10: 2,
    0b11_11_01_11: 3,
    0b11_11_10_00: 2,
    0b11_11_10_01: 2,
    0b11_11_10_10: 2,
    0b11_11_10_11: 3,
    0b11_11_11_00: 3,
    0b11_11_11_01: 3,
    0b11_11_11_10: 3,
    0b11_11_11_11: 4,
}

_item_uint = {0b01: 0, 0b10: 1}

_BYTE_MASK = 0b11_11_11_11

_byte_uint = {
    0b01_01_01_01: 0x0,
    0b01_01_01_10: 0x1,
    0b01_01_10_01: 0x2,
    0b01_01_10_10: 0x3,
    0b01_10_01_01: 0x4,
    0b01_10_01_10: 0x5,
    0b01_10_10_01: 0x6,
    0b01_10_10_10: 0x7,
    0b10_01_01_01: 0x8,
    0b10_01_01_10: 0x9,
    0b10_01_10_01: 0xA,
    0b10_01_10_10: 0xB,
    0b10_10_01_01: 0xC,
    0b10_10_01_10: 0xD,
    0b10_10_10_01: 0xE,
    0b10_10_10_10: 0xF,
}

_WYDE_MASK = 0b11_11_11_11_11_11_11_11

_wyde_uint = {
    0b01_01_01_01_01_01_01_01: 0x00,
    0b01_01_01_01_01_01_01_10: 0x01,
    0b01_01_01_01_01_01_10_01: 0x02,
    0b01_01_01_01_01_01_10_10: 0x03,
    0b01_01_01_01_01_10_01_01: 0x04,
    0b01_01_01_01_01_10_01_10: 0x05,
    0b01_01_01_01_01_10_10_01: 0x06,
    0b01_01_01_01_01_10_10_10: 0x07,
    0b01_01_01_01_10_01_01_01: 0x08,
    0b01_01_01_01_10_01_01_10: 0x09,
    0b01_01_01_01_10_01_10_01: 0x0A,
    0b01_01_01_01_10_01_10_10: 0x0B,
    0b01_01_01_01_10_10_01_01: 0x0C,
    0b01_01_01_01_10_10_01_10: 0x0D,
    0b01_01_01_01_10_10_10_01: 0x0E,
    0b01_01_01_01_10_10_10_10: 0x0F,
    0b01_01_01_10_01_01_01_01: 0x10,
    0b01_01_01_10_01_01_01_10: 0x11,
    0b01_01_01_10_01_01_10_01: 0x12,
    0b01_01_01_10_01_01_10_10: 0x13,
    0b01_01_01_10_01_10_01_01: 0x14,
    0b01_01_01_10_01_10_01_10: 0x15,
    0b01_01_01_10_01_10_10_01: 0x16,
    0b01_01_01_10_01_10_10_10: 0x17,
    0b01_01_01_10_10_01_01_01: 0x18,
    0b01_01_01_10_10_01_01_10: 0x19,
    0b01_01_01_10_10_01_10_01: 0x1A,
    0b01_01_01_10_10_01_10_10: 0x1B,
    0b01_01_01_10_10_10_01_01: 0x1C,
    0b01_01_01_10_10_10_01_10: 0x1D,
    0b01_01_01_10_10_10_10_01: 0x1E,
    0b01_01_01_10_10_10_10_10: 0x1F,
    0b01_01_10_01_01_01_01_01: 0x20,
    0b01_01_10_01_01_01_01_10: 0x21,
    0b01_01_10_01_01_01_10_01: 0x22,
    0b01_01_10_01_01_01_10_10: 0x23,
    0b01_01_10_01_01_10_01_01: 0x24,
    0b01_01_10_01_01_10_01_10: 0x25,
    0b01_01_10_01_01_10_10_01: 0x26,
    0b01_01_10_01_01_10_10_10: 0x27,
    0b01_01_10_01_10_01_01_01: 0x28,
    0b01_01_10_01_10_01_01_10: 0x29,
    0b01_01_10_01_10_01_10_01: 0x2A,
    0b01_01_10_01_10_01_10_10: 0x2B,
    0b01_01_10_01_10_10_01_01: 0x2C,
    0b01_01_10_01_10_10_01_10: 0x2D,
    0b01_01_10_01_10_10_10_01: 0x2E,
    0b01_01_10_01_10_10_10_10: 0x2F,
    0b01_01_10_10_01_01_01_01: 0x30,
    0b01_01_10_10_01_01_01_10: 0x31,
    0b01_01_10_10_01_01_10_01: 0x32,
    0b01_01_10_10_01_01_10_10: 0x33,
    0b01_01_10_10_01_10_01_01: 0x34,
    0b01_01_10_10_01_10_01_10: 0x35,
    0b01_01_10_10_01_10_10_01: 0x36,
    0b01_01_10_10_01_10_10_10: 0x37,
    0b01_01_10_10_10_01_01_01: 0x38,
    0b01_01_10_10_10_01_01_10: 0x39,
    0b01_01_10_10_10_01_10_01: 0x3A,
    0b01_01_10_10_10_01_10_10: 0x3B,
    0b01_01_10_10_10_10_01_01: 0x3C,
    0b01_01_10_10_10_10_01_10: 0x3D,
    0b01_01_10_10_10_10_10_01: 0x3E,
    0b01_01_10_10_10_10_10_10: 0x3F,
    0b01_10_01_01_01_01_01_01: 0x40,
    0b01_10_01_01_01_01_01_10: 0x41,
    0b01_10_01_01_01_01_10_01: 0x42,
    0b01_10_01_01_01_01_10_10: 0x43,
    0b01_10_01_01_01_10_01_01: 0x44,
    0b01_10_01_01_01_10_01_10: 0x45,
    0b01_10_01_01_01_10_10_01: 0x46,
    0b01_10_01_01_01_10_10_10: 0x47,
    0b01_10_01_01_10_01_01_01: 0x48,
    0b01_10_01_01_10_01_01_10: 0x49,
    0b01_10_01_01_10_01_10_01: 0x4A,
    0b01_10_01_01_10_01_10_10: 0x4B,
    0b01_10_01_01_10_10_01_01: 0x4C,
    0b01_10_01_01_10_10_01_10: 0x4D,
    0b01_10_01_01_10_10_10_01: 0x4E,
    0b01_10_01_01_10_10_10_10: 0x4F,
    0b01_10_01_10_01_01_01_01: 0x50,
    0b01_10_01_10_01_01_01_10: 0x51,
    0b01_10_01_10_01_01_10_01: 0x52,
    0b01_10_01_10_01_01_10_10: 0x53,
    0b01_10_01_10_01_10_01_01: 0x54,
    0b01_10_01_10_01_10_01_10: 0x55,
    0b01_10_01_10_01_10_10_01: 0x56,
    0b01_10_01_10_01_10_10_10: 0x57,
    0b01_10_01_10_10_01_01_01: 0x58,
    0b01_10_01_10_10_01_01_10: 0x59,
    0b01_10_01_10_10_01_10_01: 0x5A,
    0b01_10_01_10_10_01_10_10: 0x5B,
    0b01_10_01_10_10_10_01_01: 0x5C,
    0b01_10_01_10_10_10_01_10: 0x5D,
    0b01_10_01_10_10_10_10_01: 0x5E,
    0b01_10_01_10_10_10_10_10: 0x5F,
    0b01_10_10_01_01_01_01_01: 0x60,
    0b01_10_10_01_01_01_01_10: 0x61,
    0b01_10_10_01_01_01_10_01: 0x62,
    0b01_10_10_01_01_01_10_10: 0x63,
    0b01_10_10_01_01_10_01_01: 0x64,
    0b01_10_10_01_01_10_01_10: 0x65,
    0b01_10_10_01_01_10_10_01: 0x66,
    0b01_10_10_01_01_10_10_10: 0x67,
    0b01_10_10_01_10_01_01_01: 0x68,
    0b01_10_10_01_10_01_01_10: 0x69,
    0b01_10_10_01_10_01_10_01: 0x6A,
    0b01_10_10_01_10_01_10_10: 0x6B,
    0b01_10_10_01_10_10_01_01: 0x6C,
    0b01_10_10_01_10_10_01_10: 0x6D,
    0b01_10_10_01_10_10_10_01: 0x6E,
    0b01_10_10_01_10_10_10_10: 0x6F,
    0b01_10_10_10_01_01_01_01: 0x70,
    0b01_10_10_10_01_01_01_10: 0x71,
    0b01_10_10_10_01_01_10_01: 0x72,
    0b01_10_10_10_01_01_10_10: 0x73,
    0b01_10_10_10_01_10_01_01: 0x74,
    0b01_10_10_10_01_10_01_10: 0x75,
    0b01_10_10_10_01_10_10_01: 0x76,
    0b01_10_10_10_01_10_10_10: 0x77,
    0b01_10_10_10_10_01_01_01: 0x78,
    0b01_10_10_10_10_01_01_10: 0x79,
    0b01_10_10_10_10_01_10_01: 0x7A,
    0b01_10_10_10_10_01_10_10: 0x7B,
    0b01_10_10_10_10_10_01_01: 0x7C,
    0b01_10_10_10_10_10_01_10: 0x7D,
    0b01_10_10_10_10_10_10_01: 0x7E,
    0b01_10_10_10_10_10_10_10: 0x7F,
    0b10_01_01_01_01_01_01_01: 0x80,
    0b10_01_01_01_01_01_01_10: 0x81,
    0b10_01_01_01_01_01_10_01: 0x82,
    0b10_01_01_01_01_01_10_10: 0x83,
    0b10_01_01_01_01_10_01_01: 0x84,
    0b10_01_01_01_01_10_01_10: 0x85,
    0b10_01_01_01_01_10_10_01: 0x86,
    0b10_01_01_01_01_10_10_10: 0x87,
    0b10_01_01_01_10_01_01_01: 0x88,
    0b10_01_01_01_10_01_01_10: 0x89,
    0b10_01_01_01_10_01_10_01: 0x8A,
    0b10_01_01_01_10_01_10_10: 0x8B,
    0b10_01_01_01_10_10_01_01: 0x8C,
    0b10_01_01_01_10_10_01_10: 0x8D,
    0b10_01_01_01_10_10_10_01: 0x8E,
    0b10_01_01_01_10_10_10_10: 0x8F,
    0b10_01_01_10_01_01_01_01: 0x90,
    0b10_01_01_10_01_01_01_10: 0x91,
    0b10_01_01_10_01_01_10_01: 0x92,
    0b10_01_01_10_01_01_10_10: 0x93,
    0b10_01_01_10_01_10_01_01: 0x94,
    0b10_01_01_10_01_10_01_10: 0x95,
    0b10_01_01_10_01_10_10_01: 0x96,
    0b10_01_01_10_01_10_10_10: 0x97,
    0b10_01_01_10_10_01_01_01: 0x98,
    0b10_01_01_10_10_01_01_10: 0x99,
    0b10_01_01_10_10_01_10_01: 0x9A,
    0b10_01_01_10_10_01_10_10: 0x9B,
    0b10_01_01_10_10_10_01_01: 0x9C,
    0b10_01_01_10_10_10_01_10: 0x9D,
    0b10_01_01_10_10_10_10_01: 0x9E,
    0b10_01_01_10_10_10_10_10: 0x9F,
    0b10_01_10_01_01_01_01_01: 0xA0,
    0b10_01_10_01_01_01_01_10: 0xA1,
    0b10_01_10_01_01_01_10_01: 0xA2,
    0b10_01_10_01_01_01_10_10: 0xA3,
    0b10_01_10_01_01_10_01_01: 0xA4,
    0b10_01_10_01_01_10_01_10: 0xA5,
    0b10_01_10_01_01_10_10_01: 0xA6,
    0b10_01_10_01_01_10_10_10: 0xA7,
    0b10_01_10_01_10_01_01_01: 0xA8,
    0b10_01_10_01_10_01_01_10: 0xA9,
    0b10_01_10_01_10_01_10_01: 0xAA,
    0b10_01_10_01_10_01_10_10: 0xAB,
    0b10_01_10_01_10_10_01_01: 0xAC,
    0b10_01_10_01_10_10_01_10: 0xAD,
    0b10_01_10_01_10_10_10_01: 0xAE,
    0b10_01_10_01_10_10_10_10: 0xAF,
    0b10_01_10_10_01_01_01_01: 0xB0,
    0b10_01_10_10_01_01_01_10: 0xB1,
    0b10_01_10_10_01_01_10_01: 0xB2,
    0b10_01_10_10_01_01_10_10: 0xB3,
    0b10_01_10_10_01_10_01_01: 0xB4,
    0b10_01_10_10_01_10_01_10: 0xB5,
    0b10_01_10_10_01_10_10_01: 0xB6,
    0b10_01_10_10_01_10_10_10: 0xB7,
    0b10_01_10_10_10_01_01_01: 0xB8,
    0b10_01_10_10_10_01_01_10: 0xB9,
    0b10_01_10_10_10_01_10_01: 0xBA,
    0b10_01_10_10_10_01_10_10: 0xBB,
    0b10_01_10_10_10_10_01_01: 0xBC,
    0b10_01_10_10_10_10_01_10: 0xBD,
    0b10_01_10_10_10_10_10_01: 0xBE,
    0b10_01_10_10_10_10_10_10: 0xBF,
    0b10_10_01_01_01_01_01_01: 0xC0,
    0b10_10_01_01_01_01_01_10: 0xC1,
    0b10_10_01_01_01_01_10_01: 0xC2,
    0b10_10_01_01_01_01_10_10: 0xC3,
    0b10_10_01_01_01_10_01_01: 0xC4,
    0b10_10_01_01_01_10_01_10: 0xC5,
    0b10_10_01_01_01_10_10_01: 0xC6,
    0b10_10_01_01_01_10_10_10: 0xC7,
    0b10_10_01_01_10_01_01_01: 0xC8,
    0b10_10_01_01_10_01_01_10: 0xC9,
    0b10_10_01_01_10_01_10_01: 0xCA,
    0b10_10_01_01_10_01_10_10: 0xCB,
    0b10_10_01_01_10_10_01_01: 0xCC,
    0b10_10_01_01_10_10_01_10: 0xCD,
    0b10_10_01_01_10_10_10_01: 0xCE,
    0b10_10_01_01_10_10_10_10: 0xCF,
    0b10_10_01_10_01_01_01_01: 0xD0,
    0b10_10_01_10_01_01_01_10: 0xD1,
    0b10_10_01_10_01_01_10_01: 0xD2,
    0b10_10_01_10_01_01_10_10: 0xD3,
    0b10_10_01_10_01_10_01_01: 0xD4,
    0b10_10_01_10_01_10_01_10: 0xD5,
    0b10_10_01_10_01_10_10_01: 0xD6,
    0b10_10_01_10_01_10_10_10: 0xD7,
    0b10_10_01_10_10_01_01_01: 0xD8,
    0b10_10_01_10_10_01_01_10: 0xD9,
    0b10_10_01_10_10_01_10_01: 0xDA,
    0b10_10_01_10_10_01_10_10: 0xDB,
    0b10_10_01_10_10_10_01_01: 0xDC,
    0b10_10_01_10_10_10_01_10: 0xDD,
    0b10_10_01_10_10_10_10_01: 0xDE,
    0b10_10_01_10_10_10_10_10: 0xDF,
    0b10_10_10_01_01_01_01_01: 0xE0,
    0b10_10_10_01_01_01_01_10: 0xE1,
    0b10_10_10_01_01_01_10_01: 0xE2,
    0b10_10_10_01_01_01_10_10: 0xE3,
    0b10_10_10_01_01_10_01_01: 0xE4,
    0b10_10_10_01_01_10_01_10: 0xE5,
    0b10_10_10_01_01_10_10_01: 0xE6,
    0b10_10_10_01_01_10_10_10: 0xE7,
    0b10_10_10_01_10_01_01_01: 0xE8,
    0b10_10_10_01_10_01_01_10: 0xE9,
    0b10_10_10_01_10_01_10_01: 0xEA,
    0b10_10_10_01_10_01_10_10: 0xEB,
    0b10_10_10_01_10_10_01_01: 0xEC,
    0b10_10_10_01_10_10_01_10: 0xED,
    0b10_10_10_01_10_10_10_01: 0xEE,
    0b10_10_10_01_10_10_10_10: 0xEF,
    0b10_10_10_10_01_01_01_01: 0xF0,
    0b10_10_10_10_01_01_01_10: 0xF1,
    0b10_10_10_10_01_01_10_01: 0xF2,
    0b10_10_10_10_01_01_10_10: 0xF3,
    0b10_10_10_10_01_10_01_01: 0xF4,
    0b10_10_10_10_01_10_01_10: 0xF5,
    0b10_10_10_10_01_10_10_01: 0xF6,
    0b10_10_10_10_01_10_10_10: 0xF7,
    0b10_10_10_10_10_01_01_01: 0xF8,
    0b10_10_10_10_10_01_01_10: 0xF9,
    0b10_10_10_10_10_01_10_01: 0xFA,
    0b10_10_10_10_10_01_10_10: 0xFB,
    0b10_10_10_10_10_10_01_01: 0xFC,
    0b10_10_10_10_10_10_01_10: 0xFD,
    0b10_10_10_10_10_10_10_01: 0xFE,
    0b10_10_10_10_10_10_10_10: 0xFF,
}
