"""Logic vector data types."""

# Ignore warnings due to classproperty usage
# pylint: disable = comparison-with-callable
# pylint: disable = invalid-unary-operand-type
# Simplify access to friend object attributes
# pylint: disable = protected-access

# PyLint/PyRight are confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

from __future__ import annotations

import math
import random
import re
from collections import namedtuple
from collections.abc import Generator, Iterable
from functools import cache, partial

from .lbconst import _W, _X, _0, _1, from_char, to_char, to_vcd_char
from .util import classproperty, clog2

_VecN = {}


def _vec_n(n: int) -> type[Vec]:
    """Return Vec[n] type."""
    if n < 0:
        raise ValueError(f"Expected n ≥ 0, got {n}")
    if (cls := _VecN.get(n)) is None:
        _VecN[n] = cls = type(f"Vec[{n}]", (Vec,), {"_size": n})
    return cls


@cache
def _mask(n: int) -> int:
    """Return n bit mask."""
    return (1 << n) - 1


AddResult = namedtuple("AddResult", ["s", "co"])


class Vec:
    """One dimensional vector of lbool items.

    Do NOT construct an lbool Vec directly.
    Use one of the factory functions:
    * vec
    * uint2vec
    * int2vec
    """

    _size: int

    def __class_getitem__(cls, n: int) -> type[Vec]:
        return _vec_n(n)

    @classproperty
    def size(cls) -> int:  # pylint: disable=no-self-argument
        return cls._size

    @classproperty
    def shape(cls) -> tuple[int, ...] | None:  # pylint: disable=no-self-argument
        if cls._size == 0:
            return None
        if cls._size == 1:
            return ()
        return (cls._size,)

    @classmethod
    def xes(cls) -> Vec:
        return cls._from_data(0, 0)

    @classmethod
    def zeros(cls) -> Vec:
        return cls._from_data(cls._dmax, 0)

    @classmethod
    def ones(cls) -> Vec:
        return cls._from_data(0, cls._dmax)

    @classmethod
    def dcs(cls) -> Vec:
        return cls._from_data(cls._dmax, cls._dmax)

    @classmethod
    def rand(cls) -> Vec:
        d1 = random.getrandbits(cls.size)
        return cls._from_data(cls._dmax ^ d1, d1)

    @classmethod
    def xprop(cls, sel: Vec) -> Vec:
        if sel.has_x():
            return cls.xes()
        return cls.dcs()

    @classproperty
    def _dmax(cls) -> int:  # pylint: disable=no-self-argument
        return _mask(cls.size)

    @classmethod
    def _from_data(cls, d0: int, d1: int) -> Vec:
        obj = object.__new__(cls)
        obj._data = (d0, d1)
        return obj

    @classmethod
    def _check_size(cls, size: int):
        if size != cls.size:
            raise TypeError(f"Expected size = {cls.size}, got {size}")

    def __init__(self, d0: int, d1: int):
        self._data = (d0, d1)

    def __len__(self) -> int:  # pylint: disable=invalid-length-returned
        return self.size

    def __getitem__(self, key: int | slice) -> Vec:
        if isinstance(key, int):
            i = self._norm_index(key)
            d0, d1 = self._get_item(i)
            return Vec[1](d0, d1)
        if isinstance(key, slice):
            i, j = self._norm_slice(key)
            if i == 0 and j == self.size:
                return self
            size, (d0, d1) = self._get_items(i, j)
            return Vec[size](d0, d1)
        raise TypeError("Expected key to be int or slice")

    def __iter__(self) -> Generator[Vec[1], None, None]:
        for i in range(self.size):
            yield self.__getitem__(i)

    def __str__(self) -> str:
        if self.size == 0:
            return ""
        prefix = f"{self.size}b"
        chars = []
        for i in range(self.size):
            if i % 4 == 0 and i != 0:
                chars.append("_")
            chars.append(to_char[self._get_item(i)])
        return prefix + "".join(reversed(chars))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n = self.size
        d0, d1 = self._data
        return f"{name}(0b{d0:0{n}b}, 0b{d1:0{n}b})"

    def vcd_var(self) -> str:
        """Return VCD variable type."""
        return "reg"

    def vcd_val(self) -> str:
        """Return VCD variable value."""
        return "".join(to_vcd_char[self._get_item(i)] for i in range(self.size - 1, -1, -1))

    def __bool__(self) -> bool:
        return self.to_uint() != 0

    def __int__(self) -> int:
        return self.to_int()

    # Comparison
    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Vec[self.size]):
            return self._data == obj._data
        if isinstance(obj, str):
            size, data = _parse_lit(obj)
            return self.size == size and self._data == data
        return False

    def __hash__(self) -> int:
        return hash(self.shape) ^ hash(self._data)

    # Bitwise Arithmetic
    def __invert__(self) -> Vec:
        return self.not_()

    def __or__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        return _or_(self, other)

    def __ror__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        self._check_size(other.size)
        return _or_(other, self)

    def __and__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        return _and_(self, other)

    def __rand__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        self._check_size(other.size)
        return _and_(other, self)

    def __xor__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        return _xor(self, other)

    def __rxor__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        self._check_size(other.size)
        return _xor(other, self)

    # Note: Drop carry-out
    def __lshift__(self, n: int | Vec) -> Vec:
        y, _ = self.lsh(n)
        return y

    def __rlshift__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        y, _ = other.lsh(self)
        return y

    # Note: Drop carry-out
    def __rshift__(self, n: int | Vec) -> Vec:
        y, _ = self.rsh(n)
        return y

    def __rrshift__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        y, _ = other.rsh(self)
        return y

    # Note: Keep carry-out
    def __add__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        s, co = _add(self, other, _Vec0)
        return cat(s, co)

    def __radd__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        self._check_size(other.size)
        s, co = _add(other, self, _Vec0)
        return cat(s, co)

    # Note: Keep carry-out
    def __sub__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        s, co = _add(self, other.not_(), _Vec1)
        return cat(s, co)

    def __rsub__(self, other: Vec | str) -> Vec:
        if isinstance(other, str):
            other = _lit2vec(other)
        self._check_size(other.size)
        s, co = _add(other, self.not_(), _Vec1)
        return cat(s, co)

    # Note: Keep carry-out
    def __neg__(self) -> Vec:
        s, co = self.neg()
        return cat(s, co)

    @property
    def data(self) -> tuple[int, int]:
        return self._data

    def reshape(self, shape: tuple[int, ...]):
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)

        return _bits(shape)._from_data(self._data[0], self._data[1])

    def not_(self) -> Vec:
        """Bitwise lifted NOT.

        f(x) -> y:
            X => X | 00 => 00
            0 => 1 | 01 => 10
            1 => 0 | 10 => 01
            - => - | 11 => 11

        Returns:
            vec of equal length and inverted data.
        """
        x0, x1 = self._data
        y0, y1 = x1, x0
        return self._from_data(y0, y1)

    def uor(self) -> Vec[1]:
        """Unary lifted OR reduction.

        Returns:
            One-bit vec, data contains OR reduction.
        """
        y0, y1 = _0
        for i in range(self.size):
            x0, x1 = self._get_item(i)
            y0, y1 = (y0 & x0, y0 & x1 | y1 & x0 | y1 & x1)
        return Vec[1](y0, y1)

    def uand(self) -> Vec[1]:
        """Unary lifted AND reduction.

        Returns:
            One-bit vec, data contains AND reduction.
        """
        y0, y1 = _1
        for i in range(self.size):
            x0, x1 = self._get_item(i)
            y0, y1 = (y0 & x0 | y0 & x1 | y1 & x0, y1 & x1)
        return Vec[1](y0, y1)

    def uxnor(self) -> Vec[1]:
        """Unary lifted XNOR reduction.

        Returns:
            One-bit vec, data contains XNOR reduction.
        """
        y0, y1 = _1
        for i in range(self.size):
            x0, x1 = self._get_item(i)
            y0, y1 = (y0 & x1 | y1 & x0, y0 & x0 | y1 & x1)
        return Vec[1](y0, y1)

    def uxor(self) -> Vec[1]:
        """Unary lifted XOR reduction.

        Returns:
            One-bit vec, data contains XOR reduction.
        """
        y0, y1 = _0
        for i in range(self.size):
            x0, x1 = self._get_item(i)
            y0, y1 = (y0 & x0 | y1 & x1, y0 & x1 | y1 & x0)
        return Vec[1](y0, y1)

    def to_uint(self) -> int:
        """Convert to unsigned integer.

        Returns:
            An unsigned int.

        Raises:
            ValueError: vec is partially unknown.
        """
        if self.has_unknown():
            raise ValueError("Cannot convert unknown to uint")
        return self._data[1]

    def to_int(self) -> int:
        """Convert to signed integer.

        Returns:
            A signed int, from two's complement encoding.

        Raises:
            ValueError: vec is partially unknown.
        """
        if self.size == 0:
            return 0
        sign = self._get_item(self.size - 1)
        if sign == _1:
            return -(self.not_().to_uint() + 1)
        return self.to_uint()

    def _eq(self, v: Vec) -> Vec[1]:
        return _xnor(self, v).uand()

    def eq(self, other: Vec | str) -> Vec[1]:
        """Equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of self == other
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        return self._eq(other)

    def _ne(self, v: Vec) -> Vec[1]:
        return _xor(self, v).uor()

    def ne(self, other: Vec | str) -> Vec[1]:
        """Not Equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of self != other
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        return self._ne(other)

    def lt(self, other: Vec | str) -> Vec[1]:
        """Unsigned less than operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) < unsigned(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_uint() < other.to_uint()]
        except ValueError:
            return _VecX

    def slt(self, other: Vec | str) -> Vec[1]:
        """Signed Less than operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) < signed(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_int() < other.to_int()]
        except ValueError:
            return _VecX

    def le(self, other: Vec | str) -> Vec[1]:
        """Unsigned less than or equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) ≤ unsigned(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_uint() <= other.to_uint()]
        except ValueError:
            return _VecX

    def sle(self, other: Vec | str) -> Vec[1]:
        """Signed less than or equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) ≤ signed(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_int() <= other.to_int()]
        except ValueError:
            return _VecX

    def gt(self, other: Vec | str) -> Vec[1]:
        """Unsigned greater than operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) > unsigned(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_uint() > other.to_uint()]
        except ValueError:
            return _VecX

    def sgt(self, other: Vec | str) -> Vec[1]:
        """Signed greater than operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) > signed(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_int() > other.to_int()]
        except ValueError:
            return _VecX

    def ge(self, other: Vec | str) -> Vec[1]:
        """Unsigned greater than or equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of unsigned(self) ≥ unsigned(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_uint() >= other.to_uint()]
        except ValueError:
            return _VecX

    def sge(self, other: Vec | str) -> Vec[1]:
        """Signed greater than or equal operator.

        Args:
            other: vec of equal length.

        Returns:
            Vec[1] result of signed(self) ≥ signed(other)
        """
        if isinstance(other, str):
            other = _lit2vec(other)
        other._check_size(self.size)
        try:
            return (_Vec0, _Vec1)[self.to_int() >= other.to_int()]
        except ValueError:
            return _VecX

    def xt(self, n: int) -> Vec:
        """Unsigned extend by n bits.

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

        ext0 = _mask(n)
        d0 = self._data[0] | ext0 << self.size
        d1 = self._data[1]
        return Vec[self.size + n](d0, d1)

    def sxt(self, n: int) -> Vec:
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

        sign0, sign1 = self._get_item(self.size - 1)
        ext0 = _mask(n) * sign0
        ext1 = _mask(n) * sign1
        d0 = self._data[0] | ext0 << self.size
        d1 = self._data[1] | ext1 << self.size
        return Vec[self.size + n](d0, d1)

    def _lsh(self, n: int, ci: Vec) -> tuple[Vec, Vec]:
        co_size, (co0, co1) = self._get_items(self.size - n, self.size)
        co = Vec[co_size](co0, co1)

        _, (sh0, sh1) = self._get_items(0, self.size - n)
        d0 = ci._data[0] | sh0 << n
        d1 = ci._data[1] | sh1 << n
        y = self._from_data(d0, d1)

        return y, co

    def lsh(self, n: int | Vec, ci: Vec | None = None) -> tuple[Vec, Vec]:
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

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _VecE
        if ci is None:
            ci = Vec[n](_mask(n), 0)
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")

        return self._lsh(n, ci)

    def _rsh(self, n: int, ci: Vec) -> tuple[Vec, Vec]:
        co_size, (co0, co1) = self._get_items(0, n)
        co = Vec[co_size](co0, co1)

        sh_size, (sh0, sh1) = self._get_items(n, self.size)
        d0 = sh0 | ci._data[0] << sh_size
        d1 = sh1 | ci._data[1] << sh_size
        y = self._from_data(d0, d1)

        return y, co

    def rsh(self, n: int | Vec, ci: Vec | None = None) -> tuple[Vec, Vec]:
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

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _VecE
        if ci is None:
            ci = Vec[n](_mask(n), 0)
        elif len(ci) != n:
            raise ValueError(f"Expected ci to have len {n}")

        return self._rsh(n, ci)

    def _srsh(self, n: int) -> tuple[Vec, Vec]:
        co_size, (co0, co1) = self._get_items(0, n)
        co = Vec[co_size](co0, co1)

        sign0, sign1 = self._get_item(self.size - 1)
        ci0, ci1 = _mask(n) * sign0, _mask(n) * sign1

        sh_size, (sh0, sh1) = self._get_items(n, self.size)
        d0 = sh0 | ci0 << sh_size
        d1 = sh1 | ci1 << sh_size
        y = self._from_data(d0, d1)

        return y, co

    def srsh(self, n: int | Vec) -> tuple[Vec, Vec]:
        """Signed (arithmetic) right shift by n bits.

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

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _VecE

        return self._srsh(n)

    def _lrot(self, n: int) -> Vec:
        _, (co0, co1) = self._get_items(self.size - n, self.size)
        _, (sh0, sh1) = self._get_items(0, self.size - n)
        d0 = co0 | sh0 << n
        d1 = co1 | sh1 << n
        return self._from_data(d0, d1)

    def lrot(self, n: int | Vec) -> Vec:
        """Left rotate by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            vec left-rotated by n bits.

        Raises:
            ValueError: If n is invalid/inconsistent.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes()
            if n.has_dc():
                return self.dcs()
            n = n.to_uint()

        if not 0 <= n < self.size:
            raise ValueError(f"Expected 0 ≤ n < {self.size}, got {n}")
        if n == 0:
            return self

        return self._lrot(n)

    def _rrot(self, n: int) -> Vec:
        _, (co0, co1) = self._get_items(0, n)
        sh_size, (sh0, sh1) = self._get_items(n, self.size)
        d0 = sh0 | co0 << sh_size
        d1 = sh1 | co1 << sh_size
        return self._from_data(d0, d1)

    def rrot(self, n: int | Vec) -> Vec:
        """Right rotate by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            vec right-rotated by n bits.

        Raises:
            ValueError: If n is invalid/inconsistent.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes()
            if n.has_dc():
                return self.dcs()
            n = n.to_uint()

        if not 0 <= n < self.size:
            raise ValueError(f"Expected 0 ≤ n < {self.size}, got {n}")
        if n == 0:
            return self

        return self._rrot(n)

    def rot(self, n: int | Vec) -> Vec:
        """Rotate by n bits.

        Args:
            n: Number of bits.

        Returns:
            vec rotated by n bits.

        Raises:
            ValueError: If n is invalid/inconsistent.
        """
        if isinstance(n, Vec):
            if n.has_x():
                return self.xes()
            if n.has_dc():
                return self.dcs()
            n = n.to_uint()

        if n == 0:
            return self
        # Positive defined as left rotate
        if 0 < n < self.size:
            return self._lrot(n)
        # Negative defined as right rotate
        if -self.size < n < 0:
            return self._rrot(-n)

        raise ValueError(f"Expected -{self.size} < n < {self.size}, got {n}")

    def neg(self) -> AddResult:
        """Twos complement negation.

        Computed using 0 - self.

        Returns:
            2-tuple of (sum, carry-out).
        """
        zero = self._from_data(self._dmax, 0)
        s, co = _add(zero, self.not_(), ci=_Vec1)
        return AddResult(s, co)

    def count_xes(self) -> int:
        """Return number of X items."""
        d: int = (self._data[0] | self._data[1]) ^ self._dmax
        return d.bit_count()

    def count_zeros(self) -> int:
        """Return number of 0 items."""
        d: int = self._data[0] & (self._data[1] ^ self._dmax)
        return d.bit_count()

    def count_ones(self) -> int:
        """Return number of 1 items."""
        d: int = (self._data[0] ^ self._dmax) & self._data[1]
        return d.bit_count()

    def count_dcs(self) -> int:
        """Return number of DC items."""
        return (self._data[0] & self._data[1]).bit_count()

    def count_unknown(self) -> int:
        """Return number of X/DC items."""
        d: int = self._data[0] ^ self._data[1] ^ self._dmax
        return d.bit_count()

    def onehot(self) -> bool:
        """Return True if vec contains exactly one 1 item."""
        return not self.has_unknown() and self.count_ones() == 1

    def onehot0(self) -> bool:
        """Return True if vec contains at most one 1 item."""
        return not self.has_unknown() and self.count_ones() <= 1

    def has_x(self) -> bool:
        """Return True if vec contains at least one X item."""
        return bool((self._data[0] | self._data[1]) ^ self._dmax)

    def has_dc(self) -> bool:
        """Return True if vec contains at least one DC item."""
        return bool(self._data[0] & self._data[1])

    def has_unknown(self) -> bool:
        """Return True if vec contains at least one X/DC item."""
        return bool(self._data[0] ^ self._data[1] ^ self._dmax)

    def _norm_index(self, index: int) -> int:
        lo, hi = -self.size, self.size
        if not lo <= index < hi:
            s = f"Expected index in [{lo}, {hi}), got {index}"
            raise IndexError(s)
        # Normalize negative start index
        if index < 0:
            return index + hi
        return index

    def _norm_slice(self, sl: slice) -> tuple[int, int]:
        if sl.step is not None:
            raise ValueError("Slice step is not supported")
        lo, hi = -self.size, self.size
        # Normalize start index
        start = sl.start
        if start is None or start < lo:
            start = lo
        if start < 0:
            start += hi
        # Normalize stop index
        stop = sl.stop
        if stop is None or stop > hi:
            stop = hi
        if stop < 0:
            stop += hi
        return start, stop

    def _get_item(self, i: int) -> tuple[int, int]:
        return (self._data[0] >> i) & 1, (self._data[1] >> i) & 1

    def _get_items(self, i: int, j: int) -> tuple[int, tuple[int, int]]:
        size = j - i
        mask = _mask(size)
        return size, ((self._data[0] >> i) & mask, (self._data[1] >> i) & mask)


def _or_(v0: Vec, v1: Vec) -> Vec:
    x0, x1 = v0._data, v1._data
    y0 = x0[0] & x1[0]
    y1 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
    return v0._from_data(y0, y1)


def or_(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted OR.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains OR result.

    Raises:
        ValueError: vec lengths do not match.
    """
    if isinstance(v0, str):
        v0 = _lit2vec(v0)
    y = v0
    for v in vs:
        if isinstance(v, str):
            v = _lit2vec(v)
        v._check_size(v0.size)
        y = _or_(y, v)
    return y


def nor(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted NOR.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains NOR result.

    Raises:
        ValueError: vec lengths do not match.
    """
    return ~or_(v0, *vs)


def _and_(v0: Vec, v1: Vec) -> Vec:
    x0, x1 = v0._data, v1._data
    y0 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
    y1 = x0[1] & x1[1]
    return v0._from_data(y0, y1)


def and_(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted AND.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains AND result.

    Raises:
        ValueError: vec lengths do not match.
    """
    if isinstance(v0, str):
        v0 = _lit2vec(v0)
    y = v0
    for v in vs:
        if isinstance(v, str):
            v = _lit2vec(v)
        v._check_size(v0.size)
        y = _and_(y, v)
    return y


def nand(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted NAND.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains NAND result.

    Raises:
        ValueError: vec lengths do not match.
    """
    return ~and_(v0, *vs)


def _xnor(v0: Vec, v1: Vec) -> Vec:
    x0, x1 = v0._data, v1._data
    y0 = x0[0] & x1[1] | x0[1] & x1[0]
    y1 = x0[0] & x1[0] | x0[1] & x1[1]
    return v0._from_data(y0, y1)


def _xor(v0: Vec, v1: Vec) -> Vec:
    x0, x1 = v0._data, v1._data
    y0 = x0[0] & x1[0] | x0[1] & x1[1]
    y1 = x0[0] & x1[1] | x0[1] & x1[0]
    return v0._from_data(y0, y1)


def xor(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted XOR.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains XOR result.

    Raises:
        ValueError: vec lengths do not match.
    """
    if isinstance(v0, str):
        v0 = _lit2vec(v0)
    y = v0
    for v in vs:
        if isinstance(v, str):
            v = _lit2vec(v)
        v._check_size(v0.size)
        y = _xor(y, v)
    return y


def xnor(v0: Vec | str, *vs: Vec | str) -> Vec:
    """Bitwise lifted XNOR.

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

    Args:
        other: vec of equal length.

    Returns:
        vec of equal length, data contains XNOR result.

    Raises:
        ValueError: vec lengths do not match.
    """
    return ~xor(v0, *vs)


def _add(a: Vec, b: Vec, ci: Vec[1]) -> tuple[Vec, Vec[1]]:
    # X/DC propagation
    if a.has_x() or b.has_x() or ci.has_x():
        return a.xes(), _VecX
    if a.has_dc() or b.has_dc() or ci.has_dc():
        return a.dcs(), _VecW

    dmax = _mask(a.size)
    s = a._data[1] + b._data[1] + ci._data[1]
    co = (_Vec0, _Vec1)[s > dmax]
    s &= dmax

    return a._from_data(s ^ dmax, s), co


def add(a: Vec | str, b: Vec | str, ci: Vec[1] | str | None = None) -> AddResult:
    """Addition with carry-in.

    Args:
        a: vec
        b: vec of equal length.
        ci: one bit carry-in vec.

    Returns:
        2-tuple of (sum, carry-out).

    Raises:
        ValueError: vec lengths are invalid/inconsistent.
    """
    if isinstance(a, str):
        a = _lit2vec(a)
    if isinstance(b, str):
        b = _lit2vec(b)
        b._check_size(a.size)
    if ci is None:
        ci = _Vec0
    elif isinstance(ci, str):
        ci = _lit2vec(ci)
        ci._check_size(1)
    s, co = _add(a, b, ci)
    return AddResult(s, co)


def sub(a: Vec | str, b: Vec | str) -> AddResult:
    """Twos complement subtraction.

    Args:
        a: vec
        b: vec of equal length.

    Returns:
        2-tuple of (sum, carry-out).

    Raises:
        ValueError: vec lengths are invalid/inconsistent.
    """
    if isinstance(a, str):
        a = _lit2vec(a)
    if isinstance(b, str):
        b = _lit2vec(b)
        b._check_size(a.size)
    s, co = _add(a, b.not_(), ci=_Vec1)
    return AddResult(s, co)


def _bools2vec(xs: Iterable[int]) -> Vec:
    """Convert an iterable of bools to a vec.

    This is a convenience function.
    For data in the form of [0, 1, 0, 1, ...],
    or [False, True, False, True, ...].
    """
    size, d1 = 0, 0
    for x in xs:
        d1 |= x << size
        size += 1
    return Vec[size](d1 ^ _mask(size), d1)


_LIT_PREFIX_RE = re.compile(r"(?P<Size>[1-9][0-9]*)(?P<Base>[bdh])")


def _parse_lit(lit: str) -> tuple[int, tuple[int, int]]:
    if m := _LIT_PREFIX_RE.match(lit):
        size = int(m.group("Size"))
        base = m.group("Base")
        prefix_len = len(m.group())
        digits = lit[prefix_len:]
        # Binary
        if base == "b":
            digits = digits.replace("_", "")
            if len(digits) != size:
                s = f"Expected {size} digits, got {len(digits)}"
                raise ValueError(s)
            d0, d1 = 0, 0
            for i, c in enumerate(reversed(digits)):
                try:
                    x = from_char[c]
                except KeyError as e:
                    raise ValueError(f"Invalid literal: {lit}") from e
                d0 |= x[0] << i
                d1 |= x[1] << i
            return size, (d0, d1)
        # Decimal
        elif base == "d":
            d1 = int(digits, base=10)
            dmax = _mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        # Hexadecimal
        elif base == "h":
            d1 = int(digits, base=16)
            dmax = _mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        else:  # pragma: no cover
            assert False
    else:
        raise ValueError(f"Invalid literal: {lit}")


def _lit2vec(lit: str) -> Vec:
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
    size, (d0, d1) = _parse_lit(lit)
    return Vec[size](d0, d1)


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
            return _bools2vec([x, *rst])
        case str() as lit:
            return _lit2vec(lit)
        case _:
            raise TypeError(f"Invalid input: {obj}")


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

    # Compute required number of bits
    req_n = clog2(num + 1)
    if n is None:
        n = req_n
    elif n < req_n:
        s = f"Overflow: num = {num} required n ≥ {req_n}, got {n}"
        raise ValueError(s)

    return Vec[n](num ^ _mask(n), num)


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

    # Compute required number of bits
    if neg:
        d1 = -num
        req_n = clog2(d1) + 1
    else:
        d1 = num
        req_n = clog2(d1 + 1) + 1
    if n is None:
        n = req_n
    elif n < req_n:
        s = f"Overflow: num = {num} required n ≥ {req_n}, got {n}"
        raise ValueError(s)

    v = Vec[n](d1 ^ _mask(n), d1)
    return v.neg().s if neg else v


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
            vs.append(_lit2vec(obj))
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(vs) == 1:
        return vs[0]

    size = 0
    d0, d1 = 0, 0
    for v in vs:
        d0 |= v._data[0] << size
        d1 |= v._data[1] << size
        size += v.size
    return Vec[size](d0, d1)


def rep(obj: Vec | int | str, n: int) -> Vec:
    """Repeat a vector n times."""
    objs = [obj] * n
    return cat(*objs)


# Empty
_VecE = Vec[0](*_X)

# One bit values
_VecX = Vec[1](*_X)
_Vec0 = Vec[1](*_0)
_Vec1 = Vec[1](*_1)
_VecW = Vec[1](*_W)


class _VecEnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "VecEnum":
            return super().__new__(mcs, name, bases, attrs)

        enum_attrs = {}
        data2name: dict[tuple[int, int], str] = {}
        size = None
        for key, val in attrs.items():
            if key.startswith("__"):
                enum_attrs[key] = val
            # NAME = lit
            else:
                if size is None:
                    size, data = _parse_lit(val)
                else:
                    size_i, data = _parse_lit(val)
                    if size_i != size:
                        raise ValueError(f"Expected lit len {size}, got {size_i}")
                if key in ("X", "DC"):
                    raise ValueError(f"Cannot use reserved name = '{key}'")
                dmax = _mask(size)
                if data in ((0, 0), (dmax, dmax)):
                    raise ValueError(f"Cannot use reserved value = {val}")
                if data in data2name:
                    raise ValueError(f"Duplicate value: {val}")
                data2name[data] = key

        # Empty Enum
        if size is None:
            raise ValueError("Empty Enum is not supported")

        # Add X/DC members
        data2name[(0, 0)] = "X"
        dmax = _mask(size)
        data2name[(dmax, dmax)] = "DC"

        # Create Enum class
        enum = super().__new__(mcs, name, bases + (Vec[size],), enum_attrs)

        # Instantiate members
        for data, name in data2name.items():
            obj = object.__new__(enum)  # pyright: ignore[reportArgumentType]
            obj._data = data
            obj._name = name
            setattr(enum, name, obj)

        def _from_data(cls, d0: int, d1: int) -> Vec:
            try:
                obj = getattr(cls, data2name[(d0, d1)])
            except KeyError:
                obj = object.__new__(cls)  # pyright: ignore[reportArgumentType]
                obj._data = (d0, d1)
                obj._name = f"{cls.__name__}({Vec[cls.size].__str__(obj)})"
            return obj

        # Override Vec._from_data method
        enum._from_data = classmethod(_from_data)

        # Override Vec.__new__ method
        def _new(cls: type[Vec], v: Vec | str):
            if isinstance(v, str):
                v = _lit2vec(v)
            v._check_size(cls.size)
            return cls._from_data(v._data[0], v._data[1])

        enum.__new__ = _new

        # Override Vec.__init__ method (to do nothing)
        enum.__init__ = lambda self, v: None

        # Create name property
        enum.name = property(fget=lambda self: self._name)

        # Override VCD methods
        enum.vcd_var = lambda _: "string"
        enum.vcd_val = lambda self: self._name

        return enum


class VecEnum(metaclass=_VecEnumMeta):
    """Enum Base Class: Create enums."""


def _struct_init_source(fields: list[tuple[str, type]]) -> str:
    """Return source code for Struct __init__ method w/ fields."""
    lines = []
    s = ", ".join(f"{fn}=None" for fn, _ in fields)
    lines.append(f"def init(self, {s}):\n")
    lines.append("    d0, d1 = 0, 0\n")
    for fn, _ in fields:
        s = f"Expected field {fn} to have {{exp}} bits, got {{got}}"
        lines.append(f"    if {fn} is not None:\n")
        lines.append(f"        if isinstance({fn}, str):\n")
        lines.append(f"            {fn} = _lit2vec({fn})\n")
        lines.append(f"        got, exp = {fn}.size, self._{fn}_size\n")
        lines.append("        if got != exp:\n")
        lines.append(f'            raise TypeError(f"{s}")\n')
        lines.append(f"        d0 |= {fn}._data[0] << self._{fn}_base\n")
        lines.append(f"        d1 |= {fn}._data[1] << self._{fn}_base\n")
    lines.append("    self._data = (d0, d1)\n")
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

        if not fields:
            raise ValueError("Empty Struct is not supported")

        # Add struct member base/size attributes
        base = 0
        for field_name, field_type in fields:
            struct_attrs[f"_{field_name}_base"] = base
            struct_attrs[f"_{field_name}_size"] = field_type.size
            base += field_type.size

        # Create Struct class
        size = sum(field_type.size for _, field_type in fields)
        struct = super().__new__(mcs, name, bases + (Vec[size],), struct_attrs)

        # Override Vec __init__ method
        source = _struct_init_source(fields)
        globals_ = {"Vec": Vec, "_lit2vec": _lit2vec}
        globals_.update({ft.__name__: ft for _, ft in fields})
        locals_ = {}
        exec(source, globals_, locals_)  # pylint: disable=exec-used
        struct.__init__ = locals_["init"]

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
            size = getattr(self, f"_{name}_size")
            offset = getattr(self, f"_{name}_base")
            mask = _mask(size)
            d0 = (self._data[0] >> offset) & mask
            d1 = (self._data[1] >> offset) & mask
            return cls._from_data(d0, d1)

        for fn, ft in fields:
            setattr(struct, fn, property(fget=partial(_fget, fn, ft)))

        return struct


class VecStruct(metaclass=_VecStructMeta):
    """Struct Base Class: Create struct."""


def _union_init_source(size: int) -> str:
    """Return source code for Union __init__ method w/ fields."""
    lines = []
    s = "Expected input to have at most {{exp}} bits, got {{got}}"
    lines.append("def init(self, v):\n")
    lines.append("    if isinstance(v, str):")
    lines.append("        v = _lit2vec(v)\n")
    lines.append(f"    got, exp = v.size, {size}\n")
    lines.append("    if got > exp:\n")
    lines.append(f'        raise TypeError("{s}")\n')
    lines.append("    self._data = v._data\n")
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

        if not fields:
            raise ValueError("Empty Union is not supported")

        # Add union member base/size attributes
        for field_name, field_type in fields:
            union_attrs[f"_{field_name}_size"] = field_type.size

        # Create Union class
        size = max(field_type.size for _, field_type in fields)
        union = super().__new__(mcs, name, bases + (Vec[size],), union_attrs)

        # Override Vec __init__ method
        source = _union_init_source(size)
        globals_ = {"Vec": Vec, "_lit2vec": _lit2vec}
        globals_.update({ft.__name__: ft for _, ft in fields})
        locals_ = {}
        exec(source, globals_, locals_)  # pylint: disable=exec-used
        union.__init__ = locals_["init"]

        # Create Union fields
        def _fget(name, cls, self):
            size = getattr(self, f"_{name}_size")
            mask = _mask(size)
            d0 = self._data[0] & mask
            d1 = self._data[1] & mask
            return cls._from_data(d0, d1)

        for fn, ft in fields:
            setattr(union, fn, property(fget=partial(_fget, fn, ft)))

        return union


class VecUnion(metaclass=_VecUnionMeta):
    """Union Base Class: Create union."""


_BitsShape = {}


def _bits(shape: int | tuple[int, ...] | None) -> type[Bits] | type[Vec]:
    if shape is None:
        return Vec[0]
    if isinstance(shape, tuple) and len(shape) == 0:
        return Vec[1]
    if isinstance(shape, tuple) and len(shape) == 1:
        return Vec[shape[0]]
    if isinstance(shape, int):
        return Vec[shape]

    for i, n in enumerate(shape):
        if n < 2:
            raise ValueError(f"Expected dim {i} len > 1, got {n}")

    name = f'Bits[{",".join(str(n) for n in shape)}]'
    size = math.prod(shape)
    ndim = len(shape)
    vec_n = Vec[size]
    if (bits_shape := _BitsShape.get(shape)) is None:
        _BitsShape[shape] = bits_shape = type(name, (vec_n,), {"_shape": shape})

    # Class properties
    bits_shape.size = classproperty(lambda _: size)
    bits_shape.shape = classproperty(lambda _: shape)

    # Override Vec.__len__ method
    bits_shape.__len__ = lambda _: shape[0]

    # Override Vec.__getitem__ method
    def _getitem(self, key: int | slice | tuple[int | slice, ...]) -> Vec | Bits:
        if isinstance(key, (int, slice)):
            nkey = _norm_key([key])
            return _sel(self, nkey)
        if isinstance(key, tuple):
            nkey = _norm_key(list(key))
            return _sel(self, nkey)
        s = "Expected key to be int, slice, or tuple[int | slice, ...]"
        raise TypeError(s)

    bits_shape.__getitem__ = _getitem

    # Override Vec.__iter__ method
    def _iter(self) -> Generator[Bits, None, None]:
        for i in range(shape[0]):
            yield self[i]

    bits_shape.__iter__ = _iter

    def _rstr(indent: str, b: Bits) -> str:
        # 1-D Vector
        if len(b.shape) == 1:
            return str(b)

        # 2-D Matrix
        if len(b.shape) == 2:
            sep = ", "
        # 3-D
        elif len(b.shape) == 3:
            sep = ",\n" + indent
        # N-D
        else:
            sep = ",\n\n" + indent

        f = partial(_rstr, indent + " ")
        return "[" + sep.join(map(f, b)) + "]"

    def _str(self) -> str:
        prefix = "bits"
        indent = " " * len(prefix) + "  "
        return f"{prefix}({_rstr(indent, self)})"

    bits_shape.__str__ = _str

    # Bits.flatten
    bits_shape.flatten = lambda self: vec_n(self._data[0], self._data[1])

    # Bits.flat
    def _flat(self) -> Generator[Vec[1], None, None]:
        yield from self.flatten()

    bits_shape.flat = property(fget=_flat)

    # Protected methods
    def _norm_index(index: int, i: int) -> tuple[int, int]:
        lo, hi = -shape[i], shape[i]
        if not lo <= index < hi:
            s = f"Expected index in [{lo}, {hi}), got {index}"
            raise IndexError(s)
        # Normalize negative start index
        if index < 0:
            index += hi
        return (index, index + 1)

    def _norm_slice(sl: slice, i: int) -> tuple[int, int]:
        if sl.step is not None:
            raise ValueError("Slice step is not supported")
        lo, hi = -shape[i], shape[i]
        # Normalize start index
        start = sl.start
        if start is None or start < lo:
            start = lo
        if start < 0:
            start += hi
        # Normalize stop index
        stop = sl.stop
        if stop is None or stop > hi:
            stop = hi
        if stop < 0:
            stop += hi
        return start, stop

    def _norm_key(key: list[int | slice]) -> tuple[tuple[int, int], ...]:
        klen = len(key)
        if klen > ndim:
            s = f"Expected ≤ {ndim} key items, got {klen}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(ndim - klen):
            key.append(slice(None))

        # Normalize key dimensions
        nkey = []
        for i, key_i in enumerate(key):
            if isinstance(key_i, int):
                nkey.append(_norm_index(key_i, i))
            elif isinstance(key_i, slice):
                nkey.append(_norm_slice(key_i, i))
            else:  # pragma: no cover
                assert False

        return tuple(nkey)

    # Return Bits type
    return bits_shape


class Bits:
    def __class_getitem__(cls, shape: int | tuple[int, ...]) -> type[Bits] | type[Vec]:
        return _bits(shape)


def _sel(b: Bits, key: tuple[tuple[int, int], ...]) -> Vec | Bits:
    (start, stop), key_r = key[0], key[1:]
    shape_r: tuple[int, ...] = b.shape[1:]

    assert 0 <= start <= stop <= b.shape[0]

    size: int = math.prod(shape_r)
    mask = (1 << size) - 1

    def chunk(i: int) -> Bits:
        d0 = (b._data[0] >> (size * i)) & mask
        d1 = (b._data[1] >> (size * i)) & mask
        return Bits[shape_r](d0, d1)

    chunks = [chunk(i) for i in range(start, stop)]
    if key_r:
        chunks = [_sel(chunk, key_r) for chunk in chunks]
    return stack(*chunks)


def _rank2(fst: Vec, rst) -> Bits:
    d0, d1 = fst._data
    for i, v in enumerate(rst, start=1):
        if isinstance(v, str):
            v = _lit2vec(v)
            v._check_size(fst.size)
        elif not isinstance(v, Vec[fst.size]):
            s = f"Expected lit or Vec[{fst.size}]"
            raise TypeError(s)
        d0 |= v._data[0] << (fst.size * i)
        d1 |= v._data[1] << (fst.size * i)
    shape = (len(rst) + 1,) + fst.shape
    return Bits[shape](d0, d1)


def bits(obj=None) -> Vec | Bits:
    """Create a Vec/Bits using standard input formats.

    bits() or bits(None) will return the empty vec
    bits(False | True) will return a length 1 vec
    bits([False | True, ...]) will return a length n vec
    bits(str) will parse a string literal and return an arbitrary vec

    Args:
        obj: Object that can be converted to an lbool Vec/Bits.

    Returns:
        A Vec/Bits instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    match obj:
        case None | []:
            return _VecE
        case 0 | 1 as x:
            return (_Vec0, _Vec1)[x]
        case [0 | 1 as x, *rst]:
            return _bools2vec([x, *rst])
        case str() as lit:
            return _lit2vec(lit)
        case [str() as lit, *rst]:
            v = _lit2vec(lit)
            return _rank2(v, rst)
        case [Vec() as v, *rst]:
            return _rank2(v, rst)
        case [*objs]:
            return stack(*[bits(obj) for obj in objs])
        case _:
            raise TypeError(f"Invalid input: {obj}")


def stack(*objs: Vec | Bits | int | str) -> Vec | Bits:
    """Stack a sequence of Vec/Bits.

    Args:
        objs: a sequence of vec/bits/bool/lit objects.

    Returns:
        A Vec/Bits instance.

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
            vs.append(_lit2vec(obj))
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(vs) == 1:
        return vs[0]

    fst, rst = vs[0], vs[1:]

    size = fst.size
    d0, d1 = fst._data
    for v in rst:
        if v.shape != fst.shape:
            s = f"Expected shape {fst.shape}, got {v.shape}"
            raise TypeError(s)
        d0 |= v._data[0] << size
        d1 |= v._data[1] << size
        size += v.size

    shape = (len(vs),) + fst.shape
    return Bits[shape](d0, d1)
