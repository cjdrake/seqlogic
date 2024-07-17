"""Logic vector data types."""

# PyLint is confused by my hacky classproperty implementation
# pylint: disable=comparison-with-callable
# pylint: disable=invalid-unary-operand-type
# pylint: disable=no-self-argument

# I need exec to define a Struct.__init__ method
# pylint: disable=exec-used

# PyLint is confused by Enum/Struct/Union metaclass implementation
# pylint: disable=protected-access

from __future__ import annotations

import math
import random
import re
from collections import namedtuple
from collections.abc import Generator, Iterable
from functools import cache, partial

from .lbconst import _W, _X, _0, _1, from_char, to_char, to_vcd_char
from .util import classproperty, clog2

AddResult = namedtuple("AddResult", ["s", "co"])


@cache
def _mask(n: int) -> int:
    """Return n bit mask."""
    return (1 << n) - 1


_VectorSize: dict[int, type[Vector]] = {}


def _get_vec_size(size: int) -> type[Vector]:
    """Return Vector[size] type."""
    assert isinstance(size, int) and size > 1
    try:
        return _VectorSize[size]
    except KeyError:
        name = f"Vector[{size}]"
        _VectorSize[size] = cls = type(name, (Vector,), {"_size": size})
        return cls


def _vec_size(size: int) -> type[Empty] | type[Scalar] | type[Vector]:
    """Vector[size] class factory."""
    # Check size value
    if size < 0:
        raise ValueError(f"Expected size ≥ 0, got {size}")
    # Degenerate case: Null
    if size == 0:
        return Empty
    # Degenerate case: 0-D
    if size == 1:
        return Scalar
    # General case: 1-D
    return _get_vec_size(size)


_ArrayShape: dict[tuple[int, ...], type[Array]] = {}


def _get_array_shape(shape: tuple[int, ...]) -> type[Array]:
    """Return Array[shape] type."""
    assert isinstance(shape, tuple) and len(shape) > 1 and all(n > 1 for n in shape)
    try:
        return _ArrayShape[shape]
    except KeyError:
        name = f'Array[{",".join(str(n) for n in shape)}]'
        _ArrayShape[shape] = cls = type(name, (Array,), {"_shape": shape})
        return cls


def _array_shape(shape: tuple[int, ...]) -> type[Scalar] | type[Vector] | type[Array]:
    """Array[shape] class factory."""
    # Check shape value
    for i, n in enumerate(shape):
        if n < 2:
            raise ValueError(f"Expected shape[{i}] > 1, got {n}")
    # Degenerate case: 0-D
    if len(shape) == 0:
        return Scalar
    # Degenerate case: 1-D
    if len(shape) == 1:
        return _vec_size(shape[0])
    # General case: N-D
    return _get_array_shape(shape)


def _expect_type(arg, t: type[Bits]):
    if isinstance(arg, str):
        b = _lit2vec(arg)
    else:
        b = arg
    if not isinstance(b, t):
        raise TypeError(f"Expected arg to be {t.__name__} or str literal")
    return b


def _expect_size(arg, size: int) -> Bits:
    b = _expect_type(arg, Bits)
    if b.size != size:
        raise TypeError(f"Expected size {size}, got {b.size}")
    return b


class _ShapeIf:
    """Shaping interface."""

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        raise NotImplementedError()  # pragma: no cover


class Bits:
    """Base class for Bits

    Bits
    |
    +-- Empty
    |
    +-- Scalar
    |
    +-- Vector -- Enum
    |
    +-- Array
    |
    +-- Struct
    |
    +-- Union

    Do NOT construct a bit array directly.
    Use one of the factory functions:

        * vec
        * uint2vec
        * int2vec
        * bits
    """

    @classproperty
    def size(cls) -> int:
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def cast(cls, b: Bits) -> Bits:
        if b.size != cls.size:
            raise TypeError(f"Expected size {cls.size}, got {b.size}")
        obj = object.__new__(cls)
        Bits.__init__(obj, b.data[0], b.data[1])
        return obj

    @classmethod
    def cast_data(cls, d0: int, d1: int) -> Bits:
        obj = object.__new__(cls)
        Bits.__init__(obj, d0, d1)
        return obj

    @classmethod
    def xes(cls) -> Bits:
        return cls.cast_data(0, 0)

    @classmethod
    def zeros(cls) -> Bits:
        return cls.cast_data(cls._dmax, 0)

    @classmethod
    def ones(cls) -> Bits:
        return cls.cast_data(0, cls._dmax)

    @classmethod
    def dcs(cls) -> Bits:
        return cls.cast_data(cls._dmax, cls._dmax)

    @classmethod
    def rand(cls) -> Bits:
        d1 = random.getrandbits(cls.size)
        return cls.cast_data(cls._dmax ^ d1, d1)

    @classmethod
    def xprop(cls, sel: Bits) -> Bits:
        if sel.has_x():
            return cls.xes()
        return cls.dcs()

    @classproperty
    def _dmax(cls) -> int:
        return _mask(cls.size)

    def __init__(self, d0: int, d1: int):
        self._data = (d0, d1)

    @property
    def data(self) -> tuple[int, int]:
        return self._data

    def __str__(self) -> str:
        if self.size == 0:
            return ""
        prefix = f"{self.size}b"
        chars = []
        for i in range(self.size):
            if i % 4 == 0 and i != 0:
                chars.append("_")
            chars.append(to_char[self._get_index(i)])
        return prefix + "".join(reversed(chars))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        d0, d1 = self._data
        return f"{name}(0b{d0:0{self.size}b}, 0b{d1:0{self.size}b})"

    def __bool__(self) -> bool:
        return self.to_uint() != 0

    def __int__(self) -> int:
        return self.to_int()

    # Comparison
    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, str):
            size, data = _parse_lit(obj)
            return self.size == size and self._data == data
        if isinstance(obj, Bits):
            return self.size == obj.size and self._data == obj.data
        return False

    def __hash__(self) -> int:
        return hash(self.shape) ^ hash(self._data)

    # Bitwise Operations
    def __invert__(self) -> Bits:
        return self.not_()

    def __or__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _or_(self, other)

    def __ror__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _or_(other, self)

    def __and__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _and_(self, other)

    def __rand__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _and_(other, self)

    def __xor__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _xor(self, other)

    def __rxor__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _xor(other, self)

    # Note: Drop carry-out
    def __lshift__(self, n: int | Bits) -> Bits:
        y, _ = self.lsh(n)
        return y

    def __rlshift__(self, other: Bits | str) -> Bits:
        other = _expect_type(other, Bits)
        y, _ = other.lsh(self)
        return y

    # Note: Drop carry-out
    def __rshift__(self, n: int | Bits) -> Bits:
        y, _ = self.rsh(n)
        return y

    def __rrshift__(self, other: Bits | str) -> Bits:
        other = _expect_type(other, Bits)
        y, _ = other.rsh(self)
        return y

    # Note: Keep carry-out
    def __add__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(self, other, _Scalar0)
        return cat(s, co)

    def __radd__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(other, self, _Scalar0)
        return cat(s, co)

    # Note: Keep carry-out
    def __sub__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(self, other.not_(), _Scalar1)
        return cat(s, co)

    def __rsub__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(other, self.not_(), _Scalar1)
        return cat(s, co)

    # Note: Keep carry-out
    def __neg__(self) -> Bits:
        s, co = self.neg()
        return cat(s, co)

    def not_(self) -> Bits:
        """Bitwise NOT.

        f(x) -> y:
            X => X | 00 => 00
            0 => 1 | 01 => 10
            1 => 0 | 10 => 01
            - => - | 11 => 11

        Returns:
            Bits of equal size w/ inverted data.
        """
        x0, x1 = self._data
        y0, y1 = x1, x0
        cls = self.__class__
        return cls.cast_data(y0, y1)

    def uor(self) -> Scalar:
        """Unary OR reduction.

        Returns:
            Scalar w/ OR reduction.
        """
        y0, y1 = _0
        for i in range(self.size):
            x0, x1 = self._get_index(i)
            y0, y1 = (y0 & x0, y0 & x1 | y1 & x0 | y1 & x1)
        return Scalar(y0, y1)

    def uand(self) -> Scalar:
        """Unary AND reduction.

        Returns:
            Scalar w/ AND reduction.
        """
        y0, y1 = _1
        for i in range(self.size):
            x0, x1 = self._get_index(i)
            y0, y1 = (y0 & x0 | y0 & x1 | y1 & x0, y1 & x1)
        return Scalar(y0, y1)

    def uxnor(self) -> Scalar:
        """Unary XNOR reduction.

        Returns:
            Scalar w/ XNOR reduction.
        """
        y0, y1 = _1
        for i in range(self.size):
            x0, x1 = self._get_index(i)
            y0, y1 = (y0 & x1 | y1 & x0, y0 & x0 | y1 & x1)
        return Scalar(y0, y1)

    def uxor(self) -> Scalar:
        """Unary XOR reduction.

        Returns:
            Scalar w/ XOR reduction.
        """
        y0, y1 = _0
        for i in range(self.size):
            x0, x1 = self._get_index(i)
            y0, y1 = (y0 & x0 | y1 & x1, y0 & x1 | y1 & x0)
        return Scalar(y0, y1)

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
        sign = self._get_index(self.size - 1)
        if sign == _1:
            return -(self.not_().to_uint() + 1)
        return self.to_uint()

    def _eq(self, b: Bits) -> Scalar:
        return _xnor(self, b).uand()

    def eq(self, other: Bits | str) -> Scalar:
        """Equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of self == other
        """
        other = _expect_size(other, self.size)
        return self._eq(other)

    def _ne(self, b: Bits) -> Scalar:
        return _xor(self, b).uor()

    def ne(self, other: Bits | str) -> Scalar:
        """Not Equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of self != other
        """
        other = _expect_size(other, self.size)
        return self._ne(other)

    def lt(self, other: Bits | str) -> Scalar:
        """Unsigned less than operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of unsigned(self) < unsigned(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_uint() < other.to_uint()]
        except ValueError:
            return _ScalarX

    def slt(self, other: Bits | str) -> Scalar:
        """Signed Less than operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of signed(self) < signed(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_int() < other.to_int()]
        except ValueError:
            return _ScalarX

    def le(self, other: Bits | str) -> Scalar:
        """Unsigned less than or equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of unsigned(self) ≤ unsigned(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_uint() <= other.to_uint()]
        except ValueError:
            return _ScalarX

    def sle(self, other: Bits | str) -> Scalar:
        """Signed less than or equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of signed(self) ≤ signed(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_int() <= other.to_int()]
        except ValueError:
            return _ScalarX

    def gt(self, other: Bits | str) -> Scalar:
        """Unsigned greater than operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of unsigned(self) > unsigned(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_uint() > other.to_uint()]
        except ValueError:
            return _ScalarX

    def sgt(self, other: Bits | str) -> Scalar:
        """Signed greater than operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of signed(self) > signed(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_int() > other.to_int()]
        except ValueError:
            return _ScalarX

    def ge(self, other: Bits | str) -> Scalar:
        """Unsigned greater than or equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of unsigned(self) ≥ unsigned(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_uint() >= other.to_uint()]
        except ValueError:
            return _ScalarX

    def sge(self, other: Bits | str) -> Scalar:
        """Signed greater than or equal operator.

        Args:
            other: Bits of equal size.

        Returns:
            Scalar result of signed(self) ≥ signed(other)
        """
        other = _expect_size(other, self.size)
        try:
            return (_Scalar0, _Scalar1)[self.to_int() >= other.to_int()]
        except ValueError:
            return _ScalarX

    def xt(self, n: int) -> Bits:
        """Unsigned extend by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            Vector zero-extended by n bits.

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
        return _vec_size(self.size + n)(d0, d1)

    def sxt(self, n: int) -> Bits:
        """Sign extend by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            Vector sign-extended by n bits.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"Expected n ≥ 0, got {n}")
        if n == 0:
            return self

        sign0, sign1 = self._get_index(self.size - 1)
        ext0 = _mask(n) * sign0
        ext1 = _mask(n) * sign1
        d0 = self._data[0] | ext0 << self.size
        d1 = self._data[1] | ext1 << self.size
        return _vec_size(self.size + n)(d0, d1)

    def _lsh(self, n: int, ci: Bits) -> tuple[Bits, Empty | Scalar | Vector]:
        co_size, (co0, co1) = self._get_slice(self.size - n, self.size)
        co = _vec_size(co_size)(co0, co1)

        _, (sh0, sh1) = self._get_slice(0, self.size - n)
        d0 = ci.data[0] | sh0 << n
        d1 = ci.data[1] | sh1 << n
        cls = self.__class__
        y = cls.cast_data(d0, d1)

        return y, co

    def lsh(self, n: int | Bits, ci: Bits | None = None) -> tuple[Bits, Empty | Scalar | Vector]:
        """Left shift by n bits.

        Args:
            n: Non-negative number of bits.
            ci: Optional "carry in"

        Returns:
            Bits left-shifted by n bits.
            If ci is provided, use it for shift input.
            Otherwise use zeros.

        Raises:
            ValueError: If n or ci are invalid/inconsistent.
        """
        if isinstance(n, Bits):
            if n.has_x():
                return self.xes(), _Empty
            if n.has_dc():
                return self.dcs(), _Empty
            n = n.to_uint()

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _Empty
        if ci is None:
            ci = _vec_size(n)(_mask(n), 0)
        elif ci.size != n:
            raise ValueError(f"Expected ci to have size {n}")

        return self._lsh(n, ci)

    def _rsh(self, n: int, ci: Bits) -> tuple[Bits, Empty | Scalar | Vector]:
        co_size, (co0, co1) = self._get_slice(0, n)
        co = _vec_size(co_size)(co0, co1)

        sh_size, (sh0, sh1) = self._get_slice(n, self.size)
        d0 = sh0 | ci.data[0] << sh_size
        d1 = sh1 | ci.data[1] << sh_size
        cls = self.__class__
        y = cls.cast_data(d0, d1)

        return y, co

    def rsh(self, n: int | Bits, ci: Bits | None = None) -> tuple[Bits, Empty | Scalar | Vector]:
        """Right shift by n bits.

        Args:
            n: Non-negative number of bits.
            ci: Optional "carry in"

        Returns:
            Bits right-shifted by n bits.
            If ci is provided, use it for shift input.
            Otherwise use zeros.

        Raises:
            ValueError: If n or ci are invalid/inconsistent.
        """
        if isinstance(n, Bits):
            if n.has_x():
                return self.xes(), _Empty
            if n.has_dc():
                return self.dcs(), _Empty
            n = n.to_uint()

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _Empty
        if ci is None:
            ci = _vec_size(n)(_mask(n), 0)
        elif ci.size != n:
            raise ValueError(f"Expected ci to have size {n}")

        return self._rsh(n, ci)

    def _srsh(self, n: int) -> tuple[Bits, Empty | Scalar | Vector]:
        co_size, (co0, co1) = self._get_slice(0, n)
        co = _vec_size(co_size)(co0, co1)

        sign0, sign1 = self._get_index(self.size - 1)
        ci0, ci1 = _mask(n) * sign0, _mask(n) * sign1

        sh_size, (sh0, sh1) = self._get_slice(n, self.size)
        d0 = sh0 | ci0 << sh_size
        d1 = sh1 | ci1 << sh_size
        cls = self.__class__
        y = cls.cast_data(d0, d1)

        return y, co

    def srsh(self, n: int | Bits) -> tuple[Bits, Empty | Scalar | Vector]:
        """Signed (arithmetic) right shift by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            Bits arithmetically right-shifted by n bits.

        Raises:
            ValueError: If n is invalid.
        """
        if isinstance(n, Bits):
            if n.has_x():
                return self.xes(), _Empty
            if n.has_dc():
                return self.dcs(), _Empty
            n = n.to_uint()

        if not 0 <= n <= self.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {self.size}, got {n}")
        if n == 0:
            return self, _Empty

        return self._srsh(n)

    def _lrot(self, n: int) -> Bits:
        _, (co0, co1) = self._get_slice(self.size - n, self.size)
        _, (sh0, sh1) = self._get_slice(0, self.size - n)
        d0 = co0 | sh0 << n
        d1 = co1 | sh1 << n
        cls = self.__class__
        return cls.cast_data(d0, d1)

    def lrot(self, n: int | Bits) -> Bits:
        """Left rotate by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            Bits left-rotated by n bits.

        Raises:
            ValueError: If n is invalid/inconsistent.
        """
        if isinstance(n, Bits):
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

    def _rrot(self, n: int) -> Bits:
        _, (co0, co1) = self._get_slice(0, n)
        sh_size, (sh0, sh1) = self._get_slice(n, self.size)
        d0 = sh0 | co0 << sh_size
        d1 = sh1 | co1 << sh_size
        cls = self.__class__
        return cls.cast_data(d0, d1)

    def rrot(self, n: int | Bits) -> Bits:
        """Right rotate by n bits.

        Args:
            n: Non-negative number of bits.

        Returns:
            Bits right-rotated by n bits.

        Raises:
            ValueError: If n is invalid/inconsistent.
        """
        if isinstance(n, Bits):
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

    def neg(self) -> AddResult:
        """Twos complement negation.

        Computed using 0 - self.

        Returns:
            2-tuple of (sum, carry-out).
        """
        cls = self.__class__
        zero = cls.cast_data(self._dmax, 0)
        s, co = _add(zero, self.not_(), ci=_Scalar1)
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

    def vcd_var(self) -> str:
        """Return VCD variable type."""
        return "reg"

    def vcd_val(self) -> str:
        """Return VCD variable value."""
        return "".join(to_vcd_char[self._get_index(i)] for i in range(self.size - 1, -1, -1))

    def _get_index(self, i: int) -> tuple[int, int]:
        return (self._data[0] >> i) & 1, (self._data[1] >> i) & 1

    def _get_slice(self, i: int, j: int) -> tuple[int, tuple[int, int]]:
        size = j - i
        mask = _mask(size)
        return size, ((self._data[0] >> i) & mask, (self._data[1] >> i) & mask)

    def _get_key(self, key: int | slice) -> tuple[int, tuple[int, int]]:
        if isinstance(key, int):
            index = _norm_index(self.size, key)
            return 1, self._get_index(index)
        if isinstance(key, slice):
            start, stop = _norm_slice(self.size, key)
            if start != 0 or stop != self.size:
                return self._get_slice(start, stop)
            return self.size, self._data
        raise TypeError("Expected key to be int or slice")


class Empty(Bits, _ShapeIf):
    """Empty sequence of bits."""

    @classproperty
    def size(cls) -> int:
        return 0

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (0,)

    def __len__(self) -> int:
        return 0

    # __getitem__ is NOT defined on Empty

    def __iter__(self) -> Generator[Scalar, None, None]:
        yield from ()


_Empty = Empty(*_X)


class Scalar(Bits, _ShapeIf):
    """Zero dimensional (scalar) sequence of bits."""

    @classproperty
    def size(cls) -> int:
        return 1

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (1,)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: int | slice) -> Empty | Scalar:
        size, (d0, d1) = self._get_key(key)
        return _vec_size(size)(d0, d1)

    def __iter__(self) -> Generator[Scalar, None, None]:
        yield self


_ScalarX = Scalar(*_X)
_Scalar0 = Scalar(*_0)
_Scalar1 = Scalar(*_1)
_ScalarW = Scalar(*_W)


class Vector(Bits, _ShapeIf):
    """One dimensional (vector) sequence of bits."""

    _size: int

    def __class_getitem__(cls, size: int) -> type[Empty] | type[Scalar] | type[Vector]:
        match size:
            case int():
                return _vec_size(size)
            case _:
                raise TypeError(f"Invalid size parameter: {size}")

    @classproperty
    def size(cls) -> int:
        return cls._size

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (cls._size,)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: int | slice) -> Empty | Scalar | Vector:
        size, (d0, d1) = self._get_key(key)
        return _vec_size(size)(d0, d1)

    def __iter__(self) -> Generator[Scalar, None, None]:
        for i in range(self._size):
            d0, d1 = self._get_index(i)
            yield _vec_size(1)(d0, d1)

    def reshape(self, shape: tuple[int, ...]) -> Vector | Array:
        if shape == self.shape:
            return self
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return Array[shape](self._data[0], self._data[1])

    def flatten(self) -> Vector:
        return self


class Array(Bits, _ShapeIf):
    """N dimensional (array) sequence of bits."""

    _shape: tuple[int, ...]

    def __class_getitem__(cls, shape: tuple[int, ...]) -> type[_ShapeIf]:
        match shape:
            case []:
                return Scalar
            case int() as size:
                return _vec_size(size)
            case [int() as size]:
                return _vec_size(size)
            case [int(), *rst] if all(isinstance(n, int) for n in rst):
                return _array_shape(shape)
            case _:
                raise TypeError(f"Invalid shape parameter: {shape}")

    @classproperty
    def size(cls) -> int:
        return math.prod(cls._shape)

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return cls._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> _ShapeIf:
        if isinstance(key, (int, slice)):
            nkey = self._norm_key([key])
            return _sel(self, nkey)
        if isinstance(key, tuple):
            nkey = self._norm_key(list(key))
            return _sel(self, nkey)
        s = "Expected key to be int, slice, or tuple[int | slice, ...]"
        raise TypeError(s)

    def __iter__(self) -> Generator[_ShapeIf, None, None]:
        for i in range(self._shape[0]):
            yield self[i]

    def __str__(self) -> str:
        prefix = "bits"
        indent = " " * len(prefix) + "  "
        return f"{prefix}({_rstr(indent, self)})"

    def reshape(self, shape: tuple[int, ...]) -> Vector | Array:
        if shape == self.shape:
            return self
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return Array[shape](self._data[0], self._data[1])

    def flatten(self) -> Vector:
        return _vec_size(self.size)(self._data[0], self._data[1])

    @classmethod
    def _norm_key(cls, keys: list[int | slice]) -> tuple[tuple[int, int], ...]:
        ndim = len(cls._shape)
        klen = len(keys)

        if klen > ndim:
            s = f"Expected ≤ {ndim} key items, got {klen}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(ndim - klen):
            keys.append(slice(None))

        # Normalize key dimensions
        def f(n: int, key: int | slice) -> tuple[int, int]:
            if isinstance(key, int):
                i = _norm_index(n, key)
                return (i, i + 1)
            if isinstance(key, slice):
                return _norm_slice(n, key)
            assert False  # pragma: no cover

        return tuple(f(n, key) for n, key in zip(cls._shape, keys))


def _or_(b0: Bits, b1: Bits) -> Bits:
    x0, x1 = b0.data, b1.data
    y0 = x0[0] & x1[0]
    y1 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
    cls = b0.__class__
    return cls.cast_data(y0, y1)


def or_(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise OR.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ OR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    b0 = _expect_type(b0, Bits)
    y = b0
    for b in bs:
        b = _expect_size(b, b0.size)
        y = _or_(y, b)
    return y


def nor(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise NOR.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ NOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return ~or_(b0, *bs)


def _and_(b0: Bits, b1: Bits) -> Bits:
    x0, x1 = b0.data, b1.data
    y0 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
    y1 = x0[1] & x1[1]
    cls = b0.__class__
    return cls.cast_data(y0, y1)


def and_(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise AND.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ AND result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    b0 = _expect_type(b0, Bits)
    y = b0
    for b in bs:
        b = _expect_size(b, b0.size)
        y = _and_(y, b)
    return y


def nand(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise NAND.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ NAND result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return ~and_(b0, *bs)


def _xnor(b0: Bits, b1: Bits) -> Bits:
    x0, x1 = b0.data, b1.data
    y0 = x0[0] & x1[1] | x0[1] & x1[0]
    y1 = x0[0] & x1[0] | x0[1] & x1[1]
    cls = b0.__class__
    return cls.cast_data(y0, y1)


def _xor(b0: Bits, b1: Bits) -> Bits:
    x0, x1 = b0.data, b1.data
    y0 = x0[0] & x1[0] | x0[1] & x1[1]
    y1 = x0[0] & x1[1] | x0[1] & x1[0]
    cls = b0.__class__
    return cls.cast_data(y0, y1)


def xor(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise XOR.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ XOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    b0 = _expect_type(b0, Bits)
    y = b0
    for b in bs:
        b = _expect_size(b, b0.size)
        y = _xor(y, b)
    return y


def xnor(b0: Bits | str, *bs: Bits | str) -> Bits:
    """Bitwise XNOR.

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
        other: Bits of equal size.

    Returns:
        Bits of equal size, w/ XNOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return ~xor(b0, *bs)


def _add(a: Bits, b: Bits, ci: Scalar) -> tuple[Bits, Scalar]:
    # X/DC propagation
    if a.has_x() or b.has_x() or ci.has_x():
        return a.xes(), _ScalarX
    if a.has_dc() or b.has_dc() or ci.has_dc():
        return a.dcs(), _ScalarW

    dmax = _mask(a.size)
    s = a.data[1] + b.data[1] + ci.data[1]
    co = (_Scalar0, _Scalar1)[s > dmax]
    s &= dmax

    cls = a.__class__
    return cls.cast_data(s ^ dmax, s), co


def add(a: Bits | str, b: Bits | str, ci: Scalar | str | None = None) -> AddResult:
    """Addition with carry-in.

    Args:
        a: Bits
        b: Bits of equal length.
        ci: Scalar carry-in.

    Returns:
        2-tuple of (sum, carry-out).

    Raises:
        ValueError: Bits sizes are invalid/inconsistent.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    if ci is None:
        ci = _Scalar0
    else:
        ci = _expect_type(ci, Scalar)
    s, co = _add(a, b, ci)
    return AddResult(s, co)


def sub(a: Bits | str, b: Bits | str) -> AddResult:
    """Twos complement subtraction.

    Args:
        a: Bits
        b: Bits of equal length.

    Returns:
        2-tuple of (sum, carry-out).

    Raises:
        ValueError: Bits sizes are invalid/inconsistent.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    s, co = _add(a, b.not_(), ci=_Scalar1)
    return AddResult(s, co)


def _bools2vec(xs: Iterable[int]) -> Empty | Scalar | Vector:
    """Convert an iterable of bools to a vec.

    This is a convenience function.
    For data in the form of [0, 1, 0, 1, ...],
    or [False, True, False, True, ...].
    """
    size, d1 = 0, 0
    for x in xs:
        d1 |= x << size
        size += 1
    return _vec_size(size)(d1 ^ _mask(size), d1)


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
                    raise ValueError(f"Invalid lit: {lit}") from e
                d0 |= x[0] << i
                d1 |= x[1] << i
            return size, (d0, d1)
        # Decimal
        if base == "d":
            d1 = int(digits, base=10)
            dmax = _mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        # Hexadecimal
        if base == "h":
            d1 = int(digits, base=16)
            dmax = _mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        assert False  # pragma: no cover
    raise ValueError(f"Invalid lit: {lit}")


def _lit2vec(lit: str) -> Scalar | Vector:
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
    return _vec_size(size)(d0, d1)


def uint2vec(n: int, size: int | None = None) -> Empty | Scalar | Vector:
    """Convert nonnegative int to Vector.

    Args:
        n: A nonnegative integer.
        size: Optional output length.

    Returns:
        A Vector instance.

    Raises:
        ValueError: If n is negative or overflows the output length.
    """
    if n < 0:
        raise ValueError(f"Expected n ≥ 0, got {n}")

    # Compute required number of bits
    min_size = clog2(n + 1)
    if size is None:
        size = min_size
    elif size < min_size:
        s = f"Overflow: n = {n} required size ≥ {min_size}, got {size}"
        raise ValueError(s)

    return _vec_size(size)(n ^ _mask(size), n)


def int2vec(n: int, size: int | None = None) -> Scalar | Vector:
    """Convert int to Vector.

    Args:
        n: An integer.
        size: Optional output length.

    Returns:
        A Vec instance.

    Raises:
        ValueError: If n overflows the output length.
    """
    neg = n < 0

    # Compute required number of bits
    if neg:
        d1 = -n
        min_size = clog2(d1) + 1
    else:
        d1 = n
        min_size = clog2(d1 + 1) + 1
    if size is None:
        size = min_size
    elif size < min_size:
        s = f"Overflow: n = {n} required size ≥ {min_size}, got {size}"
        raise ValueError(s)

    v = _vec_size(size)(d1 ^ _mask(size), d1)
    if neg:
        return v.neg().s
    return v


def cat(*objs: Bits | int | str) -> Empty | Scalar | Vector:
    """Concatenate a sequence of Vectors.

    Args:
        objs: a sequence of vec/bool/lit objects.

    Returns:
        A Vec instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    if len(objs) == 0:
        return _Empty

    # Convert inputs
    bs = []
    for obj in objs:
        if isinstance(obj, Bits):
            bs.append(obj)
        elif obj in (0, 1):
            bs.append((_Scalar0, _Scalar1)[obj])
        elif isinstance(obj, str):
            v = _lit2vec(obj)
            bs.append(v)
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(bs) == 1:
        return bs[0]

    size = 0
    d0, d1 = 0, 0
    for b in bs:
        d0 |= b.data[0] << size
        d1 |= b.data[1] << size
        size += b.size
    return _vec_size(size)(d0, d1)


def rep(obj: Bits | int | str, n: int) -> Empty | Scalar | Vector:
    """Repeat a Vector n times."""
    objs = [obj] * n
    return cat(*objs)


class _EnumMeta(type):
    """Enum Metaclass: Create enum base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Enum":
            return super().__new__(mcs, name, bases, attrs)

        enum_attrs = {}
        data2key: dict[tuple[int, int], str] = {}
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
                if data in data2key:
                    raise ValueError(f"Duplicate value: {val}")
                data2key[data] = key

        # Empty Enum
        if size is None:
            raise ValueError("Empty Enum is not supported")

        # Add X/DC members
        data2key[(0, 0)] = "X"
        dmax = _mask(size)
        data2key[(dmax, dmax)] = "DC"

        # Create Enum class
        vec_n = _vec_size(size)
        enum = super().__new__(mcs, name, bases + (vec_n,), enum_attrs)

        # Instantiate members
        for (d0, d1), key in data2key.items():
            obj = object.__new__(enum)
            Bits.__init__(obj, d0, d1)
            setattr(enum, key, obj)

        # Override Vector.cast method
        def _cast(cls, b: Bits) -> Bits:
            try:
                obj = getattr(cls, data2key[b.data])
            except KeyError:
                obj = object.__new__(cls)
                Bits.__init__(obj, b.data[0], b.data[1])
            return obj

        enum.cast = classmethod(_cast)

        # Override Vector.cast_data method
        def _cast_data(cls, d0: int, d1: int) -> Bits:
            try:
                obj = getattr(cls, data2key[(d0, d1)])
            except KeyError:
                obj = object.__new__(cls)
                Bits.__init__(obj, d0, d1)
            return obj

        # Override Vector.cast_data method
        enum.cast_data = classmethod(_cast_data)

        # Override Vector.__new__ method
        def _new(cls, b: Bits | str):
            b = _expect_size(b, cls.size)
            return cls.cast(b)

        enum.__new__ = _new

        # Override Vector.__init__ method (to do nothing)
        enum.__init__ = lambda self, b: None

        # Create name property
        def _name(self):
            try:
                return data2key[self._data]
            except KeyError:
                return f"{name}({vec_n.__str__(self)})"

        enum.name = property(fget=_name)

        # Override VCD methods
        enum.vcd_var = lambda _: "string"
        enum.vcd_val = _name

        return enum


class Enum(metaclass=_EnumMeta):
    """Enum Base Class: Create enums."""


def _struct_init_source(fields: list[tuple[str, type]]) -> str:
    """Return source code for Struct __init__ method w/ fields."""
    lines = []
    s = ", ".join(f"{fn}=None" for fn, _ in fields)
    lines.append(f"def init(self, {s}):\n")
    s = ", ".join(fn for fn, _ in fields)
    lines.append(f"    _init_body(self, {s})\n")
    return "".join(lines)


class _StructMeta(type):
    """Struct Metaclass: Create struct base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Struct":
            return super().__new__(mcs, name, bases, attrs)

        # Scan attributes for field_name: field_type items
        fields = []
        for key, val in attrs.items():
            if key == "__annotations__":
                for field_name, field_type in val.items():
                    fields.append((field_name, field_type))

        if not fields:
            raise ValueError("Empty Struct is not supported")

        # Add struct member base/size attributes
        offset = 0
        offsets = {}
        for field_name, field_type in fields:
            offsets[field_name] = offset
            offset += field_type.size

        # Create Struct class
        size = sum(field_type.size for _, field_type in fields)
        struct = super().__new__(mcs, name, bases + (Bits,), {})

        # Class properties
        struct.size = classproperty(lambda _: size)

        # Override Bits.__init__ method
        def _init_body(obj, *args):
            d0, d1 = 0, 0
            for arg, (fn, ft) in zip(args, fields):
                if arg is not None:
                    # TODO(cjdrake): Check input type?
                    b = _expect_size(arg, ft.size)
                    d0 |= b.data[0] << offsets[fn]
                    d1 |= b.data[1] << offsets[fn]
            Bits.__init__(obj, d0, d1)

        source = _struct_init_source(fields)
        globals_ = {"_init_body": _init_body}
        locals_ = {}
        exec(source, globals_, locals_)
        struct.__init__ = locals_["init"]

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | slice) -> Empty | Scalar | Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        struct.__getitem__ = _getitem

        # Override Bits.__str__ method
        def _str(self):
            args = []
            for fn, ft in fields:
                b = getattr(self, fn)
                if issubclass(ft, Enum):
                    arg = f"{fn}={ft.__name__}.{b.name}"
                else:
                    arg = f"{fn}={b!s}"
                args.append(arg)
            return f'{name}({", ".join(args)})'

        struct.__str__ = _str

        # Override Bits.__repr__ method
        def _repr(self):
            args = []
            for fn, ft in fields:
                b = getattr(self, fn)
                if issubclass(ft, Enum):
                    arg = f"{fn}={ft.__name__}.{b.name}"
                else:
                    arg = f"{fn}={b!r}"
                args.append(arg)
            return f'{name}({", ".join(args)})'

        struct.__repr__ = _repr

        # Create Struct fields
        def _fget(fn: str, ft: type[Bits], self):
            mask = _mask(ft.size)
            d0 = (self._data[0] >> offsets[fn]) & mask
            d1 = (self._data[1] >> offsets[fn]) & mask
            return ft.cast_data(d0, d1)

        for fn, ft in fields:
            setattr(struct, fn, property(fget=partial(_fget, fn, ft)))

        return struct


class Struct(metaclass=_StructMeta):
    """Struct Base Class: Create struct."""


class _UnionMeta(type):
    """Union Metaclass: Create union base classes."""

    def __new__(mcs, name, bases, attrs):
        # Base case for API
        if name == "Union":
            return super().__new__(mcs, name, bases, attrs)

        # Scan attributes for field_name: field_type items
        fields = []
        for key, val in attrs.items():
            if key == "__annotations__":
                for field_name, field_type in val.items():
                    fields.append((field_name, field_type))

        if not fields:
            raise ValueError("Empty Union is not supported")

        # Create Union class
        size = max(field_type.size for _, field_type in fields)
        union = super().__new__(mcs, name, bases + (Bits,), {})

        # Class properties
        union.size = classproperty(lambda _: size)

        # Override Vec __init__ method
        def _init(self, arg: Bits | str):
            if isinstance(arg, str):
                b = _lit2vec(arg)
            else:
                b = arg
            ts = []
            for _, ft in fields:
                if ft not in ts:
                    ts.append(ft)
            if not isinstance(b, tuple(ts)):
                s = ", ".join(t.__name__ for t in ts)
                s = f"Expected arg to be {{{s}}}, or str literal"
                raise TypeError(s)
            Bits.__init__(self, b.data[0], b.data[1])

        union.__init__ = _init

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | slice) -> Empty | Scalar | Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        union.__getitem__ = _getitem

        # Create Union fields
        def _fget(ft: type[Bits], self):
            mask = _mask(ft.size)
            d0 = self._data[0] & mask
            d1 = self._data[1] & mask
            return ft.cast_data(d0, d1)

        for fn, ft in fields:
            setattr(union, fn, property(fget=partial(_fget, ft)))

        return union


class Union(metaclass=_UnionMeta):
    """Union Base Class: Create union."""


def vec(obj=None) -> Empty | Scalar | Vector:
    """Create a Vector using standard input formats.

    vec() or vec(None) will return the empty vec
    vec(False | True) will return a length 1 vec
    vec([False | True, ...]) will return a length n vec
    vec(str) will parse a string literal and return an arbitrary vec

    Args:
        obj: Object that can be converted to a Vector instance.

    Returns:
        A Vec instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    match obj:
        case None | []:
            return _Empty
        case 0 | 1 as x:
            return (_Scalar0, _Scalar1)[x]
        case [0 | 1 as x, *rst]:
            return _bools2vec([x, *rst])
        case str() as lit:
            return _lit2vec(lit)
        case _:
            raise TypeError(f"Invalid input: {obj}")


def bits(obj=None) -> _ShapeIf:
    """Create a Bits object using standard input formats.

    bits() or bits(None) will return the empty vec
    bits(False | True) will return a length 1 vec
    bits([False | True, ...]) will return a length n vec
    bits(str) will parse a string literal and return an arbitrary vec

    Args:
        obj: Object that can be converted to a Bits instance.

    Returns:
        A Bits instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    match obj:
        case None | []:
            return _Empty
        case 0 | 1 as x:
            return (_Scalar0, _Scalar1)[x]
        case [0 | 1 as x, *rst]:
            return _bools2vec([x, *rst])
        case str() as lit:
            return _lit2vec(lit)
        case [str() as lit, *rst]:
            v = _lit2vec(lit)
            return _rank2(v, *rst)
        case [Vector() as v, *rst]:
            return _rank2(v, *rst)
        case [*objs]:
            return stack(*[bits(obj) for obj in objs])
        case _:
            raise TypeError(f"Invalid input: {obj}")


def stack(*objs: _ShapeIf | int | str) -> _ShapeIf:
    """Stack a sequence of Vec/Bits.

    Args:
        objs: a sequence of vec/bits/bool/lit objects.

    Returns:
        A Vec/Bits instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    if len(objs) == 0:
        return _Empty

    # Convert inputs
    bs = []
    for obj in objs:
        if isinstance(obj, _ShapeIf):
            bs.append(obj)
        elif obj in (0, 1):
            bs.append((_Scalar0, _Scalar1)[obj])
        elif isinstance(obj, str):
            v = _lit2vec(obj)
            bs.append(v)
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(bs) == 1:
        return bs[0]

    fst, rst = bs[0], bs[1:]

    size = fst.size
    d0, d1 = fst.data
    for b in rst:
        if b.shape != fst.shape:
            s = f"Expected shape {fst.shape}, got {b.shape}"
            raise TypeError(s)
        d0 |= b.data[0] << size
        d1 |= b.data[1] << size
        size += b.size

    # {Empty, Empty, ...} => Empty
    if fst.shape == (0,):
        return _Empty
    # {Scalar, Scalar, ...} => Vector[K]
    if fst.shape == (1,):
        size = len(bs)
        return _vec_size(size)(d0, d1)
    # {Vector[K], Vector[K], ...} => Array[J,K]
    # {Array[J,K], Array[J,K], ...} => Array[I,J,K]
    shape = (len(bs),) + fst.shape
    return _array_shape(shape)(d0, d1)


def _sel(b: _ShapeIf, key: tuple[tuple[int, int], ...]) -> _ShapeIf:
    (start, stop), key_r = key[0], key[1:]
    assert 0 <= start <= stop <= b.shape[0]

    # Partial select a:b
    if start != 0 or stop != b.shape[0]:
        shape_r: tuple[int, ...] = b.shape[1:]
        size: int = math.prod(shape_r)
        mask = (1 << size) - 1

        def f(i: int) -> _ShapeIf:
            d0 = (b.data[0] >> (size * i)) & mask
            d1 = (b.data[1] >> (size * i)) & mask
            return _array_shape(shape_r)(d0, d1)

        if key_r:
            return stack(*[_sel(f(i), key_r) for i in range(start, stop)])
        return stack(*[f(i) for i in range(start, stop)])

    # Full select 0:n
    if key_r:
        return stack(*[_sel(x, key_r) for x in b])
    return b


def _rank2(fst: Scalar | Vector, *rst: Vector | str) -> Vector | Array:
    d0, d1 = fst.data
    for i, v in enumerate(rst, start=1):
        v = _expect_type(v, Vector[fst.size])
        d0 |= v.data[0] << (fst.size * i)
        d1 |= v.data[1] << (fst.size * i)
    shape = (len(rst) + 1,) + fst.shape
    return _array_shape(shape)(d0, d1)


def _norm_index(n: int, index: int) -> int:
    lo, hi = -n, n
    if not lo <= index < hi:
        s = f"Expected index in [{lo}, {hi}), got {index}"
        raise IndexError(s)
    # Normalize negative start index
    if index < 0:
        return index + hi
    return index


def _norm_slice(n: int, sl: slice) -> tuple[int, int]:
    lo, hi = -n, n
    if sl.step is not None:
        raise ValueError("Slice step is not supported")
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


def _rstr(indent: str, b: Vector | Array) -> str:
    # 1-D Vector
    if len(b.shape) == 1:
        return Bits.__str__(b)
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
