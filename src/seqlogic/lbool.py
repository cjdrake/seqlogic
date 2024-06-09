"""Lifted Boolean scalar and vector data types.

The conventional Boolean type consists of {False, True}, or {0, 1}.
You can pack Booleans using an int, e.g. hex(0xaa | 0x55) == '0xff'.

The "lifted" Boolean data type expands {0, 1} to {0, 1, Neither, DontCare}.

Zero and One retain their original meanings.

"Neither" means neither zero nor one.
This is an illogical or metastable value that always dominates other values.
We will use a shorthand character 'X'.

"DontCare" means either zero or one.
This is an unknown or uninitialized value that may either dominate or be
dominated by other values, depending on the operation.

The 'vec' class packs multiple lifted Booleans into a vector,
which enables efficient, bit-wise operations and arithmetic.
"""

# Simplify access to friend object attributes
# pylint: disable = protected-access

# PyLint/PyRight are confused by MetaClass behavior
# pyright: reportAttributeAccessIssue=false


from __future__ import annotations

import re
from collections.abc import Generator, Iterable
from functools import cached_property, partial

from .lbconst import (
    BYTE_MASK,
    ITEM_BITS,
    ITEM_MASK,
    WYDE_MASK,
    byte_cnt_dcs,
    byte_cnt_ones,
    byte_cnt_xes,
    byte_cnt_zeros,
    byte_uint,
    from_bit,
    from_char,
    from_hexchar,
    item_uint,
    to_char,
    wyde_uint,
)
from .util import classproperty, clog2

# Scalars
_X = 0b00
_0 = 0b01
_1 = 0b10
_W = 0b11


_LNOT = (_X, _1, _0, _W)


def not_(x: int) -> int:
    """Lifted NOT function.

    f(x) -> y:
        X => X | 00 => 00
        0 => 1 | 01 => 10
        1 => 0 | 10 => 01
        - => - | 11 => 11
    """
    return _LNOT[x]


_LNOR = (
    (_X, _X, _X, _X),
    (_X, _1, _0, _W),
    (_X, _0, _0, _0),
    (_X, _W, _0, _W),
)


def nor(x0: int, x1: int) -> int:
    """Lifted NOR function.

    f(x0, x1) -> y:
        0 0 => 1
        1 - => 0
        X - => X
        - 0 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+
       01 |00|10|11|01|
          +--+--+--+--+
       11 |00|11|11|01|  y0 = x0[0] & x1[1]
          +--+--+--+--+     | x0[1] & x1[0]
       10 |00|01|01|01|     | x0[1] & x1[1]
          +--+--+--+--+
    """
    return _LNOR[x0][x1]


_LOR = (
    (_X, _X, _X, _X),
    (_X, _0, _1, _W),
    (_X, _1, _1, _1),
    (_X, _W, _1, _W),
)


def or_(x0: int, x1: int) -> int:
    """Lifted OR function.

    f(x0, x1) -> y:
        0 0 => 0
        1 - => 1
        X - => X
        - 0 => -

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
    return _LOR[x0][x1]


_LNAND = (
    (_X, _X, _X, _X),
    (_X, _1, _1, _1),
    (_X, _1, _0, _W),
    (_X, _1, _W, _W),
)


def nand(x0: int, x1: int) -> int:
    """Lifted NAND function.

    f(x0, x1) -> y:
        1 1 => 0
        0 - => 1
        X - => X
        - 1 => -

           x1
           00 01 11 10
          +--+--+--+--+
    x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
          +--+--+--+--+     | x0[0] & x1[1]
       01 |00|10|10|10|     | x0[1] & x1[0]
          +--+--+--+--+
       11 |00|10|11|11|  y0 = x0[1] & x1[1]
          +--+--+--+--+
       10 |00|10|11|01|
          +--+--+--+--+
    """
    return _LNAND[x0][x1]


_LAND = (
    (_X, _X, _X, _X),
    (_X, _0, _0, _0),
    (_X, _0, _1, _W),
    (_X, _0, _W, _W),
)


def and_(x0: int, x1: int) -> int:
    """Lifted AND function.

    f(x0, x1) -> y:
        1 1 => 1
        0 - => 0
        X - => X
        - 1 => -

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
    return _LAND[x0][x1]


_LXNOR = (
    (_X, _X, _X, _X),
    (_X, _1, _0, _W),
    (_X, _0, _1, _W),
    (_X, _W, _W, _W),
)


def xnor(x0: int, x1: int) -> int:
    """Lifted XNOR function.

    f(x0, x1) -> y:
        0 0 => 1
        0 1 => 0
        1 0 => 0
        1 1 => 1
        X - => X
        - 0 => -
        - 1 => -

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
    return _LXNOR[x0][x1]


_LXOR = (
    (_X, _X, _X, _X),
    (_X, _0, _1, _W),
    (_X, _1, _0, _W),
    (_X, _W, _W, _W),
)


def xor(x0: int, x1: int) -> int:
    """Lifted XOR function.

    f(x0, x1) -> y:
        0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 0
        X - => X
        - 0 => -
        - 1 => -

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
    return _LXOR[x0][x1]


_LIMPLIES = (
    (_X, _X, _X, _X),
    (_X, _1, _1, _1),
    (_X, _0, _1, _W),
    (_X, _W, _1, _W),
)


def implies(p: int, q: int) -> int:
    """Lifted IMPLIES function.

    f(p, q) -> y:
        0 0 => 1
        0 1 => 1
        1 0 => 0
        1 1 => 1
        N - => N
        0 - => 1
        1 - => -
        - 0 => -
        - 1 => 1
        - - => -

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
    return _LIMPLIES[p][q]


_VecN = {}


class Vec:
    """One dimensional vector of lbool items.

    Though it is possible to construct an lbool Vec directly,
    it is easier to use one of the factory functions:

    * vec
    * uint2vec
    * int2vec
    * zeros
    * ones
    """

    def __class_getitem__(cls, n: int):
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if (vec_n := _VecN.get(n)) is None:
            _VecN[n] = vec_n = type(f"Vec[{n}]", (Vec,), {"_n": n})
        return vec_n

    @classproperty
    def n(cls):  # pylint: disable=no-self-argument
        """Return vector length."""
        return cls._n

    @classproperty
    def nbits(cls):  # pylint: disable=no-self-argument
        """Number of bits of data."""
        return ITEM_BITS * cls._n

    @classmethod
    def check_len(cls, n: int):
        """Check for valid input length."""
        if n != cls._n:
            raise TypeError(f"Expected n = {cls._n}, got {n}")

    @classmethod
    def check_data(cls, data: int):
        """Check for valid input data."""
        a, b = 0, 1 << cls.nbits
        if not a <= data < b:
            raise ValueError(f"Expected data in [{a}, {b}), got {data}")

    @classmethod
    def xes(cls):
        obj = object.__new__(cls)
        obj._data = 0
        return obj

    @classmethod
    def dcs(cls):
        obj = object.__new__(cls)
        obj._data = (1 << cls.nbits) - 1
        return obj

    @classmethod
    def xprop(cls, sel):
        if sel.has_x():
            return cls.xes()
        return cls.dcs()

    def __init__(self, data: int):
        """Initialize.

        Args:
            data: lbool items packed into an int.

        Raises:
            ValueError if data is invalid/inconsistent
        """
        self.check_data(data)
        self._data = data

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: int | slice) -> Vec:
        if isinstance(key, int):
            i = self._norm_index(key)
            d = self._get_item(i)
            return Vec[1](d)
        if isinstance(key, slice):
            i, j = self._norm_slice(key)
            n, d = self._get_items(i, j)
            return Vec[n](d)
        raise TypeError("Expected key to be int or slice")

    def __iter__(self) -> Generator[Vec[1], None, None]:
        for i in range(self._n):
            yield self.__getitem__(i)

    def __str__(self) -> str:
        if self._n == 0:
            return ""
        prefix = f"{self._n}b"
        chars = []
        for i in range(self._n):
            if i % 4 == 0 and i != 0:
                chars.append("_")
            chars.append(to_char[self._get_item(i)])
        return prefix + "".join(reversed(chars))

    def __repr__(self) -> str:
        return f"Vec[{self._n}](0b{self._data:0{self.nbits}b})"

    def __bool__(self) -> bool:
        return self.to_uint() != 0

    def __int__(self) -> int:
        return self.to_int()

    # Comparison
    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Vec[self._n]):
            return self._data == obj.data
        if isinstance(obj, str):
            n, data = _lit2vec(obj)
            return self._n == n and self._data == data
        return False

    def __hash__(self) -> int:
        return hash(self._n) ^ hash(self._data)

    # Bitwise Arithmetic
    def __invert__(self) -> Vec:
        return self.not_()

    def __or__(self, other: Vec | str) -> Vec:
        return self.or_(other)

    def __ror__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__or__(self)

    def __and__(self, other: Vec | str) -> Vec:
        return self.and_(other)

    def __rand__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__and__(self)

    def __xor__(self, other: Vec | str) -> Vec:
        return self.xor(other)

    def __rxor__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__xor__(self)

    def __lshift__(self, n: int | Vec) -> Vec:
        return self.lsh(n)[0]

    def __rlshift__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__lshift__(self)

    def __rshift__(self, n: int | Vec) -> Vec:
        return self.rsh(n)[0]

    def __rrshift__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__rshift__(self)

    def __add__(self, other: Vec | str) -> Vec:
        return self.add(other, ci=_Vec0)[0]

    def __radd__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__add__(self)

    def __sub__(self, other: Vec | str) -> Vec:
        return self.sub(other)[0]

    def __rsub__(self, other: Vec | str) -> Vec:
        v = self._to_vec(other)
        return v.__sub__(self)

    def __neg__(self) -> Vec:
        return self.neg()[0]

    @property
    def data(self) -> int:
        """Packed items."""
        return self._data

    def not_(self) -> Vec:
        """Bitwise lifted NOT.

        Returns:
            vec of equal length and inverted data.
        """
        x_0 = self._bit_mask[0]
        x_01 = x_0 << 1
        x_1 = self._bit_mask[1]
        x_10 = x_1 >> 1

        y0 = x_10
        y1 = x_01
        y = y1 | y0

        return Vec[self._n](y)

    def nor(self, other: Vec | str) -> Vec:
        """Bitwise lifted NOR.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains NOR result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = v._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = v._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_01
        y = y1 | y0

        return Vec[self._n](y)

    def or_(self, other: Vec | str) -> Vec:
        """Bitwise lifted OR.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains OR result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]

        x1_0 = v._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = v._bit_mask[1]

        y0 = x0_0 & x1_0
        y1 = x0_01 & x1_1 | x0_1 & x1_01 | x0_1 & x1_1
        y = y1 | y0

        return Vec[self._n](y)

    def uor(self) -> Vec[1]:
        """Unary lifted OR reduction.

        Returns:
            One-bit vec, data contains OR reduction.
        """
        data = _0
        for i in range(self._n):
            data = or_(data, self._get_item(i))
        return Vec[1](data)

    def nand(self, other: Vec | str) -> Vec:
        """Bitwise lifted NAND.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains NAND result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = v._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = v._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_10 & x1_10
        y1 = x0_01 & x1_01 | x0_01 & x1_1 | x0_1 & x1_01
        y = y1 | y0

        return Vec[self._n](y)

    def and_(self, other: Vec | str) -> Vec:
        """Bitwise lifted AND.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains AND result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = v._bit_mask[0]
        x1_1 = v._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_1 & x1_1
        y = y1 | y0

        return Vec[self._n](y)

    def uand(self) -> Vec[1]:
        """Unary lifted AND reduction.

        Returns:
            One-bit vec, data contains AND reduction.
        """
        data = _1
        for i in range(self._n):
            data = and_(data, self._get_item(i))
        return Vec[1](data)

    def xnor(self, other: Vec | str) -> Vec:
        """Bitwise lifted XNOR.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains XNOR result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = v._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = v._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_01 & x1_01 | x0_1 & x1_1
        y = y1 | y0

        return Vec[self._n](y)

    def xor(self, other: Vec | str) -> Vec:
        """Bitwise lifted XOR.

        Args:
            other: vec of equal length.

        Returns:
            vec of equal length, data contains XOR result.

        Raises:
            ValueError: vec lengths do not match.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        x0_0 = self._bit_mask[0]
        x0_01 = x0_0 << 1
        x0_1 = self._bit_mask[1]
        x0_10 = x0_1 >> 1

        x1_0 = v._bit_mask[0]
        x1_01 = x1_0 << 1
        x1_1 = v._bit_mask[1]
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_1 | x0_1 & x1_01
        y = y1 | y0

        return Vec[self._n](y)

    def uxnor(self) -> Vec[1]:
        """Unary lifted XNOR reduction.

        Returns:
            One-bit vec, data contains XNOR reduction.
        """
        data = _1
        for i in range(self._n):
            data = xnor(data, self._get_item(i))
        return Vec[1](data)

    def uxor(self) -> Vec[1]:
        """Unary lifted XOR reduction.

        Returns:
            One-bit vec, data contains XOR reduction.
        """
        data = _0
        for i in range(self._n):
            data = xor(data, self._get_item(i))
        return Vec[1](data)

    def to_uint(self) -> int:
        """Convert to unsigned integer.

        Returns:
            An unsigned int.

        Raises:
            ValueError: vec is partially unknown.
        """
        if self.has_unknown():
            raise ValueError("Cannot convert unknown to uint")

        y = 0
        n, data = 0, self._data

        stride = 8
        while n <= (self._n - stride):
            y |= wyde_uint[data & WYDE_MASK] << n
            n += stride
            data >>= ITEM_BITS * stride
        stride = 4
        while n <= (self._n - stride):
            y |= byte_uint[data & BYTE_MASK] << n
            n += stride
            data >>= ITEM_BITS * stride
        stride = 1
        while n <= (self._n - stride):
            y |= item_uint[data & ITEM_MASK] << n
            n += stride
            data >>= ITEM_BITS * stride

        return y

    def to_int(self) -> int:
        """Convert to signed integer.

        Returns:
            A signed int, from two's complement encoding.

        Raises:
            ValueError: vec is partially unknown.
        """
        if self._n == 0:
            return 0
        sign = self._get_item(self._n - 1)
        if sign == _1:
            return -(self.not_().to_uint() + 1)
        return self.to_uint()

    def eq(self, other: Vec | str) -> Vec[1]:
        """Equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of self == other
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() == v.to_uint()]
        except ValueError:
            return _VecX

    def neq(self, other: Vec | str) -> Vec[1]:
        """Not Equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of self != other
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() != v.to_uint()]
        except ValueError:
            return _VecX

    def ltu(self, other: Vec | str) -> Vec[1]:
        """Less than operator (unsigned).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) < unsigned(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() < v.to_uint()]
        except ValueError:
            return _VecX

    def lteu(self, other: Vec | str) -> Vec[1]:
        """Less than or equal operator (unsigned).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) ≤ unsigned(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() <= v.to_uint()]
        except ValueError:
            return _VecX

    def lt(self, other: Vec | str) -> Vec[1]:
        """Less than operator (signed).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) < signed(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_int() < v.to_int()]
        except ValueError:
            return _VecX

    def lte(self, other: Vec | str) -> Vec[1]:
        """Less than or equal operator (signed).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) ≤ signed(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_int() <= v.to_int()]
        except ValueError:
            return _VecX

    def gtu(self, other: Vec | str) -> Vec[1]:
        """Greater than operator (unsigned).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) > unsigned(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() > v.to_uint()]
        except ValueError:
            return _VecX

    def gteu(self, other: Vec | str) -> Vec[1]:
        """Greater than or equal operator (unsigned).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) ≥ unsigned(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_uint() >= v.to_uint()]
        except ValueError:
            return _VecX

    def gt(self, other: Vec | str) -> Vec[1]:
        """Greater than operator (signed).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) > signed(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_int() > v.to_int()]
        except ValueError:
            return _VecX

    def gte(self, other: Vec | str) -> Vec[1]:
        """Greater than or equal operator (signed).

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) ≥ signed(other)
        """
        v = self._to_vec(other)
        self.check_len(len(v))
        try:
            return (_Vec0, _Vec1)[self.to_int() >= v.to_int()]
        except ValueError:
            return _VecX

    def _match(self, pattern: Vec | str) -> bool:
        """Pattern match operator."""
        if isinstance(pattern, str):
            pattern = lit2vec(pattern)

        self.check_len(len(pattern))

        if self.has_x() or pattern.has_x():
            return False

        for i in range(self._n):
            a = self._get_item(i)
            b = pattern._get_item(i)
            # Mismatch on (0b01, 0b10) or (0b10, 0b01)
            if a ^ b == 0b11:
                return False

        return True

    def zext(self, n: int) -> Vec:
        """Zero extend by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            vec zero-extended by n bits.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if n == 0:
            return self
        return Vec[self._n + n](self._data | (_fill(_0, n) << self.nbits))

    def sext(self, n: int) -> Vec:
        """Sign extend by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            vec sign-extended by n bits.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if n == 0:
            return self
        sign = self._get_item(self._n - 1)
        return Vec[self._n + n](self._data | (_fill(sign, n) << self.nbits))

    def lsh(self, n: int | Vec, ci: Vec[1] | None = None) -> tuple[Vec, Vec]:
        """Left shift by n bits.

        Args:
            n: Non-negative number of bits.
            ci: Optional "carry in"

        Returns:
            vec left-shifted by n bits. If ci is provided, use it for shift
            input. Otherwise use zeros.

        Raises:
            ValueError: If n or ci are invalid/inconsistent.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes(), _VecE
            if n.has_dc():
                return self.dcs(), _VecE
            n = n.to_uint()

        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, _VecE
        if ci is None:
            ci = Vec[n](_fill(_0, n))
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")
        sh, co = self[:-n], self[-n:]
        y = Vec[self._n](ci._data | (sh._data << ci.nbits))
        return y, co

    def rsh(self, n: int | Vec, ci: Vec[1] | None = None) -> tuple[Vec, Vec]:
        """Right shift by n bits.

        Args:
            n: Non-negative number of bits.
            ci: Optional "carry in"

        Returns:
            vec right-shifted by n bits. If ci is provided, use it for shift
            input. Otherwise use zeros.

        Raises:
            ValueError: If n or ci are invalid/inconsistent.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes(), _VecE
            if n.has_dc():
                return self.dcs(), _VecE
            n = n.to_uint()

        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, _VecE
        if ci is None:
            ci = Vec[n](_fill(_0, n))
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")
        co, sh = self[:n], self[n:]
        y = Vec[self._n](sh._data | (ci._data << sh.nbits))
        return y, co

    def arsh(self, n: int | Vec) -> tuple[Vec, Vec]:
        """Arithmetically right shift by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            vec arithmetically right-shifted by n bits.

        Raises:
            ValueError: If n is invalid.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes(), _VecE
            if n.has_dc():
                return self.dcs(), _VecE
            n = n.to_uint()

        if not 0 <= n <= self._n:
            raise ValueError(f"Expected 0 ≤ n ≤ {self._n}, got {n}")
        if n == 0:
            return self, _VecE
        sign = self._get_item(self._n - 1)
        co, sh = self[:n], self[n:]
        y = Vec[self._n](sh._data | (_fill(sign, n) << sh.nbits))
        return y, co

    def add(self, other: Vec | str, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
        """Twos complement addition.

        Args:
            other: vec of equal length.

        Returns:
            3-tuple of (sum, carry-out, overflow).

        Raises:
            ValueError: vec lengths are invalid/inconsistent.
        """
        v = self._to_vec(other)
        self.check_len(len(v))

        # Rename for readability
        n, a, b = self._n, self, v

        if a.has_x() or b.has_x() or ci.has_x():
            return Vec[n](_fill(_X, n)), _VecX
        if a.has_dc() or b.has_dc() or ci.has_dc():
            return Vec[n](_fill(_W, n)), _VecW

        s = a.to_uint() + b.to_uint() + ci.to_uint()

        data = 0
        for i in range(n):
            data |= from_bit[s & 1] << (ITEM_BITS * i)
            s >>= 1

        # Carry out is True if there is leftover sum data
        co = (_Vec0, _Vec1)[s != 0]

        return Vec[n](data), co

    def sub(self, other: Vec | str) -> tuple[Vec, Vec[1]]:
        """Twos complement subtraction.

        Args:
            other: vec of equal length.

        Returns:
            3-tuple of (sum, carry-out, overflow).

        Raises:
            ValueError: vec lengths are invalid/inconsistent.
        """
        v = self._to_vec(other)
        return self.add(v.not_(), ci=_Vec1)

    def neg(self) -> tuple[Vec, Vec[1]]:
        """Twos complement negation.

        Computed using 0 - self.

        Returns:
            3-tuple of (sum, carry-out, overflow).
        """
        zero = Vec[self._n](_fill(_0, self._n))
        return zero.sub(self)

    def _count(self, byte_cnt: dict[int, int], item: int) -> int:
        y = 0
        n, data = self._n, self._data

        stride = 4
        while n >= stride:
            y += byte_cnt[data & BYTE_MASK]
            n -= stride
            data >>= ITEM_BITS * stride
        stride = 1
        while n >= stride:
            y += (data & ITEM_MASK) == item
            n -= stride
            data >>= ITEM_BITS * stride

        return y

    def count_xes(self) -> int:
        """Return number of X items."""
        return self._count(byte_cnt_xes, _X)

    def count_zeros(self) -> int:
        """Return number of 0 items."""
        return self._count(byte_cnt_zeros, _0)

    def count_ones(self) -> int:
        """Return number of 1 items."""
        return self._count(byte_cnt_ones, _1)

    def count_dcs(self) -> int:
        """Return number of DC items."""
        return self._count(byte_cnt_dcs, _W)

    def count_unknown(self) -> int:
        """Return number of X/DC items."""
        return self.count_xes() + self.count_dcs()

    def onehot(self) -> bool:
        """Return True if vec contains exactly one 1 item."""
        return not self.has_unknown() and self.count_ones() == 1

    def onehot0(self) -> bool:
        """Return True if vec contains at most one 1 item."""
        return not self.has_unknown() and self.count_ones() <= 1

    def has_x(self) -> bool:
        """Return True if vec contains at least one X item."""
        return self.count_xes() != 0

    def has_dc(self) -> bool:
        """Return True if vec contains at least one DC item."""
        return self.count_dcs() != 0

    def has_unknown(self) -> bool:
        """Return True if vec contains at least one X/DC item."""
        return self.count_unknown() != 0

    def _to_vec(self, obj: Vec | str) -> Vec:
        if isinstance(obj, Vec[self._n]):
            return obj
        if isinstance(obj, str):
            return lit2vec(obj)
        s = f"Expected Vec[{self._n}] or lit, got {obj.__class__.__name__}"
        raise TypeError(s)

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
        return (self._data >> (ITEM_BITS * i)) & ITEM_MASK

    def _get_items(self, i: int, j: int) -> tuple[int, int]:
        n = j - i
        nbits = ITEM_BITS * n
        mask = (1 << nbits) - 1
        return n, (self._data >> (ITEM_BITS * i)) & mask

    @cached_property
    def _mask(self) -> tuple[int, int]:
        zero_mask = _fill(_0, self._n)
        one_mask = zero_mask << 1
        return zero_mask, one_mask

    @cached_property
    def _bit_mask(self) -> tuple[int, int]:
        return self._data & self._mask[0], self._data & self._mask[1]


def _fill(x: int, n: int) -> int:
    data = 0
    for i in range(n):
        data |= x << (ITEM_BITS * i)
    return data


def uint2vec(num: int, n: int | None = None) -> Vec:
    """Convert nonnegative int to vec.

    Args:
        num: A nonnegative integer.
        n: Optional output length.

    Returns:
        A Vec instance.

    Raises:
        ValueError: If num is negative or overflows the output length.
    """
    if num < 0:
        raise ValueError(f"Expected num ≥ 0, got {num}")

    data = 0
    i = 0
    r = num

    while r >= 1:
        data |= from_bit[r & 1] << (ITEM_BITS * i)
        i += 1
        r >>= 1

    # Compute required number of bits
    req_n = clog2(num + 1)
    if n is None:
        n = req_n
    elif n < req_n:
        s = f"Overflow: num = {num} required n ≥ {req_n}, got {n}"
        raise ValueError(s)

    return Vec[i](data).zext(n - i)


def int2vec(num: int, n: int | None = None) -> Vec:
    """Convert int to vec.

    Args:
        num: An integer.
        n: Optional output length.

    Returns:
        A Vec instance.

    Raises:
        ValueError: If num overflows the output length.
    """
    neg = num < 0

    data = 0
    i = 0
    r = abs(num)

    while r >= 1:
        data |= from_bit[r & 1] << (ITEM_BITS * i)
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

    v = Vec[i](data).zext(n - i)
    return v.neg()[0] if neg else v


def bools2vec(xs: Iterable[int]) -> Vec:
    """Convert an iterable of bools to a vec.

    This is a convenience function.
    For data in the form of [0, 1, 0, 1, ...],
    or [False, True, False, True, ...].
    """
    n, data = 0, 0
    for x in xs:
        data |= from_bit[x] << (ITEM_BITS * n)
        n += 1
    return Vec[n](data)


_LIT_RE = re.compile(
    r"((?P<BinSize>[1-9][0-9]*)b(?P<BinDigits>[X01\-_]+))|"
    r"((?P<HexSize>[1-9][0-9]*)h(?P<HexDigits>[0-9a-fA-F_]+))"
)


def _lit2vec(lit: str) -> tuple[int, int]:
    if m := _LIT_RE.fullmatch(lit):
        # Binary
        if m.group("BinSize"):
            n = int(m.group("BinSize"))
            digits = m.group("BinDigits").replace("_", "")
            if len(digits) != n:
                s = f"Expected {n} digits, got {len(digits)}"
                raise ValueError(s)
            data = 0
            for i, c in enumerate(reversed(digits)):
                data |= from_char[c] << (i * ITEM_BITS)
            return n, data
        # Hexadecimal
        elif m.group("HexSize"):
            n = int(m.group("HexSize"))
            digits = m.group("HexDigits").replace("_", "")
            exp = (n + 3) // 4
            if len(digits) != exp:
                s = f"Expected {exp} digits, got {len(digits)}"
                raise ValueError(s)
            data = 0
            for i, c in enumerate(reversed(digits)):
                try:
                    x = from_hexchar[min(n - 4 * i, 4)][c]
                except KeyError as e:
                    raise ValueError(f"Character overflows size: {c}") from e
                data |= x << (4 * i * ITEM_BITS)
            return n, data
        else:  # pragma: no cover
            assert False
    else:
        raise ValueError(f"Expected str literal, got {lit}")


def lit2vec(lit: str) -> Vec:
    """Convert a string literal to a vec.

    A string literal is in the form {width}{base}{characters},
    where width is the number of bits, base is either 'b' for binary or
    'h' for hexadecimal, and characters is a string of legal characters.
    The character string can contains '_' separators for readability.

    For example:
        4b1010
        6b11_-10X
        64hdead_beef_feed_face

    Returns:
        A Vec instance.

    Raises:
        ValueError: If input literal has a syntax error.
    """
    n, data = _lit2vec(lit)
    return Vec[n](data)


def vec(obj=None) -> Vec:
    """Create a Vec using standard input formats.

    vec() or vec(None) will return the empty vec
    vec(False | True) will return a length 1 vec
    vec([False | True, ...]) will return a length n vec
    vec(str) will parse a string literal and return an arbitrary vec

    Args:
        obj: Object that can be converted to an lbool Vec.

    Returns:
        A Vec instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    match obj:
        case None | []:
            return _VecE
        case 0 | 1 as x:
            return (_Vec0, _Vec1)[x]
        case [0 | 1 as x, *rst]:
            return bools2vec([x, *rst])
        case str() as lit:
            return lit2vec(lit)
        case _:
            raise TypeError(f"Invalid input: {obj}")


def cat(*objs: Vec | int | str) -> Vec:
    """Concatenate a sequence of vectors.

    Args:
        objs: a sequence of vec/bool/lit objects.

    Returns:
        A Vec instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    if len(objs) == 0:
        return _VecE

    # Convert inputs
    vs = []
    for obj in objs:
        if isinstance(obj, Vec):
            vs.append(obj)
        elif obj in (0, 1):
            vs.append((_Vec0, _Vec1)[obj])
        elif isinstance(obj, str):
            vs.append(lit2vec(obj))
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(vs) == 1:
        return vs[0]

    n, data = 0, 0
    for v in vs:
        data |= v.data << (ITEM_BITS * n)
        n += len(v)
    return Vec[n](data)


def rep(obj: Vec | int | str, n: int) -> Vec:
    """Repeat a vector n times."""
    objs = [obj] * n
    return cat(*objs)


def zeros(n: int) -> Vec:
    """Return a vec packed with n 0 items."""
    return Vec[n](_fill(_0, n))


def ones(n: int) -> Vec:
    """Return a vec packed with n 1 items."""
    return Vec[n](_fill(_1, n))


# Empty
_VecE = Vec[0](0)

# One bit values
_VecX = Vec[1](_X)
_Vec0 = Vec[1](_0)
_Vec1 = Vec[1](_1)
_VecW = Vec[1](_W)


class _VecEnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "VecEnum":
            return super().__new__(mcs, name, bases, attrs)

        enum_attrs = {}
        data2name: dict[int, str] = {}
        n = None
        dc_data = None
        for key, val in attrs.items():
            if key.startswith("__"):
                enum_attrs[key] = val
            # NAME = lit
            else:
                if n is None:
                    n, data = _lit2vec(val)
                    dc_data = (1 << ITEM_BITS * n) - 1
                else:
                    n_i, data = _lit2vec(val)
                    if n_i != n:
                        raise ValueError(f"Expected lit len {n}, got {n_i}")
                if key in ("X", "DC"):
                    raise ValueError(f"Cannot use reserved name = '{key}'")
                if data in (0, dc_data):
                    raise ValueError(f"Cannot use reserved data = {data}")
                if data in data2name:
                    raise ValueError(f"Duplicate data: {val}")
                data2name[data] = key

        # Empty Enum
        if n is None:
            raise ValueError("Empty Enum is not supported")

        # Help the type checker
        assert dc_data is not None

        # Add X/DC members
        data2name[0] = "X"
        data2name[dc_data] = "DC"

        # Create Enum class
        enum = super().__new__(mcs, name, bases + (Vec[n],), enum_attrs)

        # Instantiate members
        for data, name in data2name.items():
            obj = object.__new__(enum)  # pyright: ignore[reportArgumentType]
            obj._data = data
            obj._name = name
            setattr(enum, name, obj)

        # Override Vec __new__ method
        def _new(cls: type[Vec], arg: Vec | str | int):
            if isinstance(arg, Vec[cls.n]):
                data = arg.data
            elif isinstance(arg, str):
                n, data = _lit2vec(arg)
                cls.check_len(n)
            elif isinstance(arg, int):
                data = arg
                cls.check_data(data)
            else:
                s = f"Expected arg to be Vec[{cls.n}], str, or int"
                raise TypeError(s)
            try:
                obj = getattr(cls, data2name[data])
            except KeyError:
                obj = object.__new__(enum)  # pyright: ignore[reportArgumentType]
                obj._data = data
                obj._name = f"{cls.__name__}({Vec[cls.n].__str__(obj)})"
            return obj

        enum.__new__ = _new

        # Override Vec __init__ method (to do nothing)
        enum.__init__ = lambda self, arg: None

        # Override Vec xes/dcs methods
        enum.xes = classmethod(lambda cls: getattr(cls, "X"))
        enum.dcs = classmethod(lambda cls: getattr(cls, "DC"))

        # Create name property
        enum.name = property(fget=lambda self: self._name)

        return enum


class VecEnum(metaclass=_VecEnumMeta):
    """Enum Base Class: Create enums."""


def _struct_init_source(fields: list[tuple[str, type]]) -> str:
    """Return source code for Struct __init__ method w/ fields."""
    lines = []
    s = ", ".join(f"{fn}: {ft.__name__} | None = None" for fn, ft in fields)
    lines.append(f"def struct_init(self, {s}):\n")
    lines.append("    data = 0\n")
    for fn, _ in fields:
        s = f"Expected field {fn} to have {{exp}} bits, got {{got}}"
        lines.append(f"    if {fn} is not None:\n")
        lines.append(f"        if isinstance({fn}, str):\n")
        lines.append(f"            {fn} = lit2vec({fn})\n")
        lines.append(f"        got, exp = len({fn}), self._{fn}_size\n")
        lines.append("        if got != exp:\n")
        lines.append(f'            raise TypeError(f"{s}")\n')
        lines.append(f"        data |= {fn}.data << ({ITEM_BITS} * self._{fn}_base)\n")
    lines.append("    self._data = data\n")
    return "".join(lines)


class _VecStructMeta(type):
    """Struct Metaclass: Create struct base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "VecStruct":
            return super().__new__(mcs, name, bases, attrs)

        # Scan attributes for field_name: field_type items
        struct_attrs = {}
        fields: list[tuple[str, type[Vec]]] = []
        for key, val in attrs.items():
            if key == "__annotations__":
                for field_name, field_type in val.items():
                    fields.append((field_name, field_type))
            # name: Type
            else:
                struct_attrs[key] = val

        # Add struct member base/size attributes
        base = 0
        for field_name, field_type in fields:
            struct_attrs[f"_{field_name}_base"] = base
            struct_attrs[f"_{field_name}_size"] = field_type.n
            base += field_type.n

        # Create Struct class
        n = sum(field_type.n for _, field_type in fields)
        struct = super().__new__(mcs, name, bases + (Vec[n],), struct_attrs)

        # Override Vec __init__ method
        source = _struct_init_source(fields)
        globals_ = {"Vec": Vec, "lit2vec": lit2vec}
        globals_.update({ft.__name__: ft for _, ft in fields})
        locals_ = {}
        exec(source, globals_, locals_)  # pylint: disable=exec-used
        struct.__init__ = locals_["struct_init"]

        # Override Vec __str__ method
        def _str(self):
            args = []
            for fn, ft in fields:
                v = getattr(self, fn)
                if issubclass(ft, VecEnum):
                    arg = f"{fn}={ft.__name__}.{v.name}"
                else:
                    arg = f"{fn}={v!s}"
                args.append(arg)
            return f'{name}({", ".join(args)})'

        struct.__str__ = _str

        # Override Vec __repr__ method
        def _repr(self):
            args = []
            for fn, ft in fields:
                v = getattr(self, fn)
                if issubclass(ft, VecEnum):
                    arg = f"{fn}={ft.__name__}.{v.name}"
                else:
                    arg = f"{fn}={v!r}"
                args.append(arg)
            return f'{name}({", ".join(args)})'

        struct.__repr__ = _repr

        # Create Struct fields
        def _fget(name, cls, self):
            nbits = ITEM_BITS * getattr(self, f"_{name}_size")
            offset = ITEM_BITS * getattr(self, f"_{name}_base")
            mask = (1 << nbits) - 1
            data = (self._data >> offset) & mask
            if issubclass(cls, (VecStruct, VecUnion)):
                obj = object.__new__(cls)
                obj._data = data
                return obj
            # Vec, VecEnum
            return cls(data)

        for fn, ft in fields:
            setattr(struct, fn, property(fget=partial(_fget, fn, ft)))

        return struct


class VecStruct(metaclass=_VecStructMeta):
    """Struct Base Class: Create struct."""


def _union_init_source(n: int, fields: list[tuple[str, type]]) -> str:
    """Return source code for Union __init__ method w/ fields."""
    lines = []
    s1 = " | ".join(ft.__name__ for _, ft in fields)
    s2 = "Expected input to have at most {{exp}} bits, got {{got}}"
    lines.append(f"def union_init(self, v: {s1}):\n")
    lines.append(f"    got, exp = len(v), {n}\n")
    lines.append("    if got > exp:\n")
    lines.append(f'        raise TypeError("{s2}")\n')
    lines.append("    self._data = v.data\n")
    return "".join(lines)


class _VecUnionMeta(type):
    """Union Metaclass: Create union base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "VecUnion":
            return super().__new__(mcs, name, bases, attrs)

        # Scan attributes for field_name: field_type items
        union_attrs = {}
        fields: list[tuple[str, type[Vec]]] = []
        for key, val in attrs.items():
            if key == "__annotations__":
                for field_name, field_type in val.items():
                    fields.append((field_name, field_type))
            # name: Type
            else:
                union_attrs[key] = val

        # Add union member base/size attributes
        for field_name, field_type in fields:
            union_attrs[f"_{field_name}_size"] = field_type.n

        # Create Union class
        n = max(field_type.n for _, field_type in fields)
        union = super().__new__(mcs, name, bases + (Vec[n],), union_attrs)

        # Override Vec __init__ method
        source = _union_init_source(n, fields)
        globals_ = {"Vec": Vec}
        globals_.update({ft.__name__: ft for _, ft in fields})
        locals_ = {}
        exec(source, globals_, locals_)  # pylint: disable=exec-used
        union.__init__ = locals_["union_init"]

        # Create Union fields
        def _fget(name, cls, self):
            nbits = ITEM_BITS * getattr(self, f"_{name}_size")
            mask = (1 << nbits) - 1
            data = self._data & mask
            if issubclass(cls, (VecStruct, VecUnion)):
                obj = object.__new__(cls)
                obj._data = data
                return obj
            # Vec, VecEnum
            return cls(data)

        for fn, ft in fields:
            setattr(union, fn, property(fget=partial(_fget, fn, ft)))

        return union


class VecUnion(metaclass=_VecUnionMeta):
    """Union Base Class: Create union."""
