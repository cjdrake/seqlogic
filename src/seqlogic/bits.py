"""SeqLogic Bits module.

Contains classes and functions that implement hardware-oriented ``bits`` data
types and operators.
"""

# PyLint is confused by my hacky classproperty implementation
# pylint: disable=comparison-with-callable
# pylint: disable=no-self-argument

# pylint: disable=protected-access

from __future__ import annotations

import math
import operator
import random
import re
from collections.abc import Callable, Generator
from functools import partial

from ._lbool import (
    _W,
    _X,
    _0,
    _1,
    from_char,
    land,
    lbv,
    lite,
    lmux,
    lnot,
    lor,
    lxnor,
    lxor,
    to_char,
    to_vcd_char,
)
from .util import classproperty, clog2, mask

_VectorSize: dict[int, type[Vector]] = {}


def _get_vec_size(size: int) -> type[Vector]:
    """Return Vector[size] type."""
    assert isinstance(size, int) and size > 1
    try:
        return _VectorSize[size]
    except KeyError:
        name = f"Vector[{size}]"
        vec = type(name, (Vector,), {"_size": size})
        _VectorSize[size] = vec
        return vec


def _vec_size(size: int) -> type[Empty | Scalar | Vector]:
    """Vector[size] class factory."""
    assert size >= 0
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
    assert len(shape) > 1 and all(isinstance(n, int) and n > 1 for n in shape)
    try:
        return _ArrayShape[shape]
    except KeyError:
        name = f'Array[{",".join(str(n) for n in shape)}]'
        size = math.prod(shape)
        array = type(name, (Array,), {"_shape": shape, "_size": size})
        _ArrayShape[shape] = array
        return array


def _expect_type(arg, t: type[Bits]):
    if isinstance(arg, str):
        x = _lit2bv(arg)
    else:
        x = arg
    if not isinstance(x, t):
        raise TypeError(f"Expected arg to be {t.__name__} or str literal")
    return x


def _expect_shift(arg, size: int) -> Bits:
    if isinstance(arg, int):
        return u2bv(arg, size)
    if isinstance(arg, str):
        return _lit2bv(arg)
    if isinstance(arg, Bits):
        return arg
    raise TypeError("Expected arg to be Bits, str literal, or int")


def _expect_size(arg, size: int) -> Bits:
    x = _expect_type(arg, Bits)
    if x.size != size:
        raise TypeError(f"Expected size {size}, got {x.size}")
    return x


def _resolve_type(t0: type[Bits], t1: type[Bits]) -> type[Bits]:
    # T (op) T -> T
    if t0 == t1:
        return t0

    # Otherwise, downgrade to Scalar/Vector
    return _vec_size(t0.size)


class _ShapeIf:
    @classproperty
    def shape(cls) -> tuple[int, ...]:
        raise NotImplementedError()  # pragma: no cover


class Bits:
    r"""Sequence of bits.

    A bit is a 4-state logical value in the set {``0``, ``1``, ``X``, ``-``}:

        * ``0`` is Boolean zero or "False"
        * ``1`` is Boolean one or "True"
        * ``X`` is an uninitialized or metastable value
        * ``-`` is a "don't care" value

    The values ``0`` and ``1`` are "known".
    The values ``X`` and ``-`` are "unknown".

    ``Bits`` is the base class for a family of hardware-oriented data types.
    All ``Bits`` objects have a ``size`` attribute.
    Shaped subclasses (``Empty``, ``Scalar``, ``Vector``, ``Array``) have a
    ``shape`` attribute.
    Composite subclasses (``Struct``, ``Union``) have user-defined attributes.

    ``Bits`` does **NOT** implement the Python ``Sequence`` protocol.

    Children::

                      Bits
                        |
          +------+------+-----+------+-----+
          |      |      |     |      |     |
        Empty Scalar Vector Array Struct Union
                        |
                      Enum

    Do **NOT** construct a Bits object directly.
    Use one of the factory functions:

        * ``bits``
        * ``stack``
        * ``u2bv``
        * ``i2bv``
    """

    @classproperty
    def size(cls) -> int:
        """Number of bits."""
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def cast(cls, x: Bits) -> Bits:
        """Convert Bits object to an instance of this class.

        For example, to cast an ``Array[2,2]`` to a ``Vector[4]``:

        >>> x = bits(["2b00", "2b11"])
        >>> Vector[4].cast(x)
        bits("4b1100")

        Raises:
            TypeError: Object size does not match this class size.
        """
        if x.size != cls.size:
            raise TypeError(f"Expected size {cls.size}, got {x.size}")
        return cls._cast_data(x.data[0], x.data[1])

    @classmethod
    def _cast_data(cls, d0: int, d1: int) -> Bits:
        obj = object.__new__(cls)
        obj._data = (d0, d1)
        return obj

    @classmethod
    def xes(cls) -> Bits:
        """Return an instance filled with ``X`` bits.

        For example:

        >>> Vector[4].xes()
        bits("4bXXXX")
        """
        return cls._cast_data(0, 0)

    @classmethod
    def zeros(cls) -> Bits:
        """Return an instance filled with ``0`` bits.

        For example:

        >>> Vector[4].zeros()
        bits("4b0000")
        """
        return cls._cast_data(cls._dmax, 0)

    @classmethod
    def ones(cls) -> Bits:
        """Return an instance filled with ``1`` bits.

        For example:

        >>> Vector[4].ones()
        bits("4b1111")
        """
        return cls._cast_data(0, cls._dmax)

    @classmethod
    def dcs(cls) -> Bits:
        """Return an instance filled with ``-`` bits.

        For example:

        >>> Vector[4].dcs()
        bits("4b----")
        """
        return cls._cast_data(cls._dmax, cls._dmax)

    @classmethod
    def rand(cls) -> Bits:
        """Return an instance filled with random bits."""
        d1 = random.getrandbits(cls.size)
        return cls._cast_data(cls._dmax ^ d1, d1)

    @classmethod
    def xprop(cls, sel: Bits) -> Bits:
        """Propagate ``X`` in a wildcard pattern (default case).

        If ``sel`` contains an ``X``, propagate ``X``.
        Otherwise, treat as a "don't care", and propagate ``-``.

        For example:

        >>> def f(x: Vector[1]) -> Vector[1]:
        ...     match x:
        ...         case "1b0":
        ...             return bits("1b1")
        ...         case _:
        ...             return Vector[1].xprop(x)

        >>> f(bits("1b0"))  # Match!
        bits("1b1")
        >>> f(bits("1b1"))  # No match; No X prop
        bits("1b-")
        >>> f(bits("1bX"))  # No match; Yes X prop
        bits("1bX")

        Args:
            sel: Bits object, typically a ``match`` subject

        Returns:
            Class instance filled with either ``-`` or ``X``.
        """
        if sel.has_x():
            return cls.xes()
        return cls.dcs()

    @classproperty
    def _dmax(cls) -> int:
        return mask(cls.size)

    @property
    def data(self) -> tuple[int, int]:
        """Internal representation."""
        return self._data

    def __bool__(self) -> bool:
        """Convert to Python ``bool``.

        A ``Bits`` object is ``True`` if its value is known nonzero.

        For example:

        >>> bool(bits("1b0"))
        False
        >>> bool(bits("1b1"))
        True
        >>> bool(bits("4b0000"))
        False
        >>> bool(bits("4b1010"))
        True

        .. warning::
            Be cautious about using any expression that *might* have an unknown
            value as the condition of a Python ``if`` or ``while`` statement.

        Raises:
            ValueError: Contains any unknown bits.
        """
        return self.to_uint() != 0

    def __int__(self) -> int:
        """Convert to Python ``int``.

        Use two's complement representation:

        * If most significant bit is ``1``, result will be negative.
        * If most significant bit is ``0``, result will be non-negative.

        For example:

        >>> int(bits("4b1010"))
        -6
        >>> int(bits("4b0101"))
        5

        Raises:
            ValueError: Contains any unknown bits.
        """
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
        return _not_(self)

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
        return _xor_(self, other)

    def __rxor__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return _xor_(other, self)

    # Note: Drop carry-out
    def __lshift__(self, n: Bits | str | int) -> Bits:
        n = _expect_shift(n, self.size)
        return _lsh(self, n)

    def __rlshift__(self, other: Bits | str) -> Bits:
        other = _expect_type(other, Bits)
        return _lsh(other, self)

    # Note: Drop carry-out
    def __rshift__(self, n: Bits | str | int) -> Bits:
        n = _expect_shift(n, self.size)
        return _rsh(self, n)

    def __rrshift__(self, other: Bits | str) -> Bits:
        other = _expect_type(other, Bits)
        return _rsh(other, self)

    # Note: Keep carry-out
    def __add__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(self, other, _Scalar0)
        return _cat(s, co)

    def __radd__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _add(other, self, _Scalar0)
        return _cat(s, co)

    # Note: Keep carry-out
    def __sub__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _sub(self, other)
        return _cat(s, co)

    def __rsub__(self, other: Bits | str) -> Scalar | Vector:
        other = _expect_size(other, self.size)
        s, co = _sub(other, self)
        return _cat(s, co)

    # Note: Keep carry-out
    def __neg__(self) -> Bits:
        s, co = _neg(self)
        return _cat(s, co)

    def __mul__(self, other: Bits | str) -> Bits:
        return mul(self, other)

    def __rmul__(self, other: Bits | str) -> Bits:
        other = _expect_size(other, self.size)
        return mul(other, self)

    def to_uint(self) -> int:
        """Convert to unsigned integer.

        Returns:
            A non-negative ``int``.

        Raises:
            ValueError: Contains any unknown bits.
        """
        if self.has_unknown():
            raise ValueError("Cannot convert unknown to uint")
        return self._data[1]

    def to_int(self) -> int:
        """Convert to signed integer.

        Returns:
            An ``int``, from two's complement encoding.

        Raises:
            ValueError: Contains any unknown bits.
        """
        if self.size == 0:
            return 0
        sign = self._get_index(self.size - 1)
        if sign == _1:
            return -(_not_(self).to_uint() + 1)
        return self.to_uint()

    def count_xes(self) -> int:
        """Return count of ``X`` bits."""
        d: int = (self._data[0] | self._data[1]) ^ self._dmax
        return d.bit_count()

    def count_zeros(self) -> int:
        """Return count of of ``0`` bits."""
        d: int = self._data[0] & (self._data[1] ^ self._dmax)
        return d.bit_count()

    def count_ones(self) -> int:
        """Return count of ``1`` bits."""
        d: int = (self._data[0] ^ self._dmax) & self._data[1]
        return d.bit_count()

    def count_dcs(self) -> int:
        """Return count of ``-`` bits."""
        d: int = self._data[0] & self._data[1]
        return d.bit_count()

    def count_unknown(self) -> int:
        """Return count of unknown bits."""
        d: int = self._data[0] ^ self._data[1] ^ self._dmax
        return d.bit_count()

    def onehot(self) -> bool:
        """Return True if contains exactly one ``1`` bit."""
        return not self.has_unknown() and self.count_ones() == 1

    def onehot0(self) -> bool:
        """Return True if contains at most one ``1`` bit."""
        return not self.has_unknown() and self.count_ones() <= 1

    def has_x(self) -> bool:
        """Return True if contains at least one ``X`` bit."""
        return bool((self._data[0] | self._data[1]) ^ self._dmax)

    def has_dc(self) -> bool:
        """Return True if contains at least one ``-`` bit."""
        return bool(self._data[0] & self._data[1])

    def has_unknown(self) -> bool:
        """Return True if contains at least one unknown bit."""
        return bool(self._data[0] ^ self._data[1] ^ self._dmax)

    def vcd_var(self) -> str:
        """Return VCD variable type."""
        return "logic"

    def vcd_val(self) -> str:
        """Return VCD variable value."""
        return "".join(to_vcd_char[self._get_index(i)] for i in range(self.size - 1, -1, -1))

    def _get_index(self, i: int) -> lbv:
        d0 = (self._data[0] >> i) & 1
        d1 = (self._data[1] >> i) & 1
        return d0, d1

    def _get_slice(self, i: int, j: int) -> tuple[int, lbv]:
        size = j - i
        m = mask(size)
        d0 = (self._data[0] >> i) & m
        d1 = (self._data[1] >> i) & m
        return size, (d0, d1)

    def _get_key(self, key: int | slice | Bits | str) -> tuple[int, lbv]:
        if isinstance(key, int):
            index = _norm_index(self.size, key)
            return 1, self._get_index(index)
        if isinstance(key, slice):
            start, stop = _norm_slice(self.size, key)
            if start != 0 or stop != self.size:
                return self._get_slice(start, stop)
            return self.size, self._data
        key = _expect_type(key, Bits)
        index = _norm_index(self.size, key.to_uint())
        return 1, self._get_index(index)


class Empty(Bits, _ShapeIf):
    """Null dimensional sequence of bits.

    Degenerate form of a ``Vector`` resulting from an empty slice.

    >>> from seqlogic import Vec
    >>> Vec[0] is Empty
    True

    To get a handle to an ``Empty`` instance:

    >>> empty = bits()

    ``Empty`` implements ``Vector`` methods,
    except for ``__getitem__``:

    >>> empty.size
    0
    >>> empty.shape
    (0,)
    >>> len(empty)
    0
    >>> empty[0]
    Traceback (most recent call last):
        ...
    TypeError: 'Empty' object is not subscriptable
    """

    def __new__(cls, d0: int, d1: int):
        assert d0 == d1 == 0
        return _Empty

    def __reversed__(self):
        yield self

    @classproperty
    def size(cls) -> int:
        return 0

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (0,)

    def __repr__(self) -> str:
        return "bits([])"

    def __str__(self) -> str:
        return "[]"

    def __len__(self) -> int:
        return 0

    # __getitem__ is NOT defined on Empty

    def __iter__(self) -> Generator[Scalar, None, None]:
        yield from ()


_Empty = Empty._cast_data(0, 0)


class Scalar(Bits, _ShapeIf):
    """Zero dimensional (scalar) sequence of bits.

    Degenerate form of a ``Vector`` resulting from a one bit slice.

    >>> from seqlogic import Vec
    >>> Vec[1] is Scalar
    True

    To get a handle to a ``Scalar`` instance:

    >>> f = bits("1b0")
    >>> t = bits("1b1")
    >>> x = bits("1bX")
    >>> dc = bits("1b-")

    For convenience, ``False`` and ``True`` also work:

    >>> bits(False) is f and bits(True) is t
    True

    ``Scalar`` implements ``Vector`` methods,
    including ``__getitem__``:

    >>> t.size
    1
    >>> t.shape
    (1,)
    >>> len(t)
    1
    >>> t[0]
    bits("1b1")
    """

    def __new__(cls, d0: int, d1: int):
        return _scalars[(d0, d1)]

    @classproperty
    def size(cls) -> int:
        return 1

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (1,)

    def __repr__(self) -> str:
        return f'bits("{self.__str__()}")'

    def __str__(self) -> str:
        return f"1b{to_char[self._data]}"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: int | slice | Bits | str) -> Empty | Scalar:
        size, (d0, d1) = self._get_key(key)
        return _vec_size(size)(d0, d1)

    def __iter__(self) -> Generator[Scalar, None, None]:
        yield self


_ScalarX = Scalar._cast_data(*_X)
_Scalar0 = Scalar._cast_data(*_0)
_Scalar1 = Scalar._cast_data(*_1)
_ScalarW = Scalar._cast_data(*_W)

_scalars = {
    _X: _ScalarX,
    _0: _Scalar0,
    _1: _Scalar1,
    _W: _ScalarW,
}
_bool2scalar = (_Scalar0, _Scalar1)


class Vector(Bits, _ShapeIf):
    """One dimensional sequence of bits.

    To create a ``Vector`` instance,
    use binary, decimal, or hexadecimal string literals:

    >>> bits("4b1010")
    bits("4b1010")
    >>> bits("4d10")
    bits("4b1010")
    >>> bits("4ha")
    bits("4b1010")

    ``Vector`` implements ``size`` and ``shape`` attributes,
    as well as ``__len__`` and ``__getitem__`` methods:

    >>> x = bits("8b1111_0000")
    >>> x.size
    8
    >>> x.shape
    (8,)
    >>> len(x)
    8
    >>> x[3]
    bits("1b0")
    >>> x[4]
    bits("1b1")
    >>> x[2:6]
    bits("4b1100")

    A ``Vector`` may be converted into an equal-size multi-dimensional ``Array``
    using the ``reshape`` method:

    >>> x.reshape((2,4))
    bits(["4b0000", "4b1111"])
    """

    _size: int

    def __class_getitem__(cls, size: int) -> type[Empty | Scalar | Vector]:
        if isinstance(size, int) and size >= 0:
            return _vec_size(size)
        raise TypeError(f"Invalid size parameter: {size}")

    def __new__(cls, d0: int, d1: int) -> Vector:
        return cls._cast_data(d0, d1)

    @classproperty
    def size(cls) -> int:
        return cls._size

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return (cls._size,)

    def __repr__(self) -> str:
        return f'bits("{self.__str__()}")'

    def __str__(self) -> str:
        prefix = f"{self.size}b"
        chars = [to_char[self._get_index(0)]]
        for i in range(1, self.size):
            if i % 4 == 0:
                chars.append("_")
            chars.append(to_char[self._get_index(i)])
        return prefix + "".join(reversed(chars))

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: int | slice | Bits | str) -> Empty | Scalar | Vector:
        size, (d0, d1) = self._get_key(key)
        return _vec_size(size)(d0, d1)

    def __iter__(self) -> Generator[Scalar, None, None]:
        for i in range(self._size):
            yield _scalars[self._get_index(i)]

    def reshape(self, shape: tuple[int, ...]) -> Vector | Array:
        if shape == self.shape:
            return self
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return _get_array_shape(shape)(self._data[0], self._data[1])

    def flatten(self) -> Vector:
        return self


class Array(Bits, _ShapeIf):
    """Multi dimensional array of bits.

    To create an ``Array`` instance, use the ``bits`` function:

    >>> x = bits(["4b0100", "4b1110"])

    ``Array`` implements ``size`` and ``shape`` attributes,
    and the ``__getitem__`` method.
    ``Array`` does **NOT** implement a ``__len__`` method.

    >>> x.size
    8
    >>> x.shape
    (2, 4)
    >>> x[0]
    bits("4b0100")
    >>> x[1]
    bits("4b1110")
    >>> x[0,0]
    bits("1b0")

    An ``Array`` may be converted into an equal-size, multi-dimensional ``Array``
    using the ``reshape`` method:

    >>> x.reshape((4,2))
    bits(["2b00", "2b01", "2b10", "2b11"])

    An ``Array`` may be converted into an equal-size, one-dimensional ``Vector``
    using the ``flatten`` method:

    >>> x.flatten()
    bits("8b1110_0100")
    """

    _shape: tuple[int, ...]
    _size: int

    def __class_getitem__(cls, shape: int | tuple[int, ...]) -> type[_ShapeIf]:
        if isinstance(shape, int):
            return _vec_size(shape)
        if isinstance(shape, tuple) and all(isinstance(n, int) and n > 1 for n in shape):
            return _get_array_shape(shape)
        raise TypeError(f"Invalid shape parameter: {shape}")

    def __new__(cls, d0: int, d1: int) -> Array:
        return cls._cast_data(d0, d1)

    @classproperty
    def size(cls) -> int:
        return cls._size

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        return cls._shape

    def __repr__(self) -> str:
        prefix = "bits"
        indent = " " * len(prefix) + "  "
        return f"{prefix}({_array_repr(indent, self)})"

    def __str__(self) -> str:
        indent = " "
        return f"{_array_str(indent, self)}"

    def __getitem__(self, key: int | slice | Bits | tuple[int | slice | Bits, ...]) -> _ShapeIf:
        if isinstance(key, (int, slice, Bits)):
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

    def reshape(self, shape: tuple[int, ...]) -> Vector | Array:
        if shape == self.shape:
            return self
        if math.prod(shape) != self._size:
            s = f"Expected shape with size {self._size}, got {shape}"
            raise ValueError(s)
        if len(shape) == 1:
            return _get_vec_size(shape[0])(self._data[0], self._data[1])
        return _get_array_shape(shape)(self._data[0], self._data[1])

    def flatten(self) -> Vector:
        return _get_vec_size(self._size)(self._data[0], self._data[1])

    @classmethod
    def _norm_key(cls, keys: list[int | slice | Bits]) -> tuple[tuple[int, int], ...]:
        ndim = len(cls._shape)
        klen = len(keys)

        if klen > ndim:
            s = f"Expected ≤ {ndim} key items, got {klen}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(ndim - klen):
            keys.append(slice(None))

        # Normalize key dimensions
        def f(n: int, key: int | slice | Bits) -> tuple[int, int]:
            if isinstance(key, int):
                i = _norm_index(n, key)
                return (i, i + 1)
            if isinstance(key, slice):
                return _norm_slice(n, key)
            if isinstance(key, Bits):
                i = _norm_index(n, key.to_uint())
                return (i, i + 1)
            assert False  # pragma: no cover

        return tuple(f(n, key) for n, key in zip(cls._shape, keys))


# Bitwise
def _not_(x: Bits) -> Bits:
    d0, d1 = lnot(x.data)
    return x._cast_data(d0, d1)


def _or_(x0: Bits, x1: Bits) -> Bits:
    d0, d1 = lor(x0.data, x1.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def _and_(x0: Bits, x1: Bits) -> Bits:
    d0, d1 = land(x0.data, x1.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def _xnor_(x0: Bits, x1: Bits) -> Bits:
    d0, d1 = lxnor(x0.data, x1.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def _xor_(x0: Bits, x1: Bits) -> Bits:
    d0, d1 = lxor(x0.data, x1.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def not_(x: Bits | str) -> Bits:
    """Unary bitwise logical NOT operator.

    Perform logical negation on each bit of the input:

    +-------+--------+
    |   x   | NOT(x) |
    +=======+========+
    | ``0`` |  ``1`` |
    +-------+--------+
    | ``1`` |  ``0`` |
    +-------+--------+
    | ``X`` |  ``X`` |
    +-------+--------+
    | ``-`` |  ``-`` |
    +-------+--------+

    For example:

    >>> not_("4b-10X")
    bits("4b-01X")

    In expressions, you can use the unary ``~`` operator:

    >>> a = bits("4b-10X")
    >>> ~a
    bits("4b-01X")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        ``Bits`` of same type and equal size

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    return _not_(x)


def nor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical NOR operator.

    Perform logical NOR on each bit of the inputs:

    +-------+-----------------------+-------------+-----------------------+
    |   x0  |           x1          | NOR(x0, x1) |          Note         |
    +=======+=======================+=============+=======================+
    | ``0`` |                 ``0`` |       ``1`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``0`` |                 ``1`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``0`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``1`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |       ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``-`` |       ``0`` | ``1`` dominates ``-`` |
    +-------+-----------------------+-------------+-----------------------+
    | ``-`` |        {``0``, ``-``} |       ``-`` | ``-`` dominates ``0`` |
    +-------+-----------------------+-------------+-----------------------+

    For example:

    >>> nor("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b-0-X_000X_-01X_XXXX")

    In expressions, you can use the unary ``~`` and binary ``|`` operators:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> ~(a | b)
    bits("16b-0-X_000X_-01X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    return _not_(or_(x0, *xs))


def or_(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical OR operator.

    Perform logical OR on each bit of the inputs:

    +-------+-----------------------+------------+-----------------------+
    |   x0  |           x1          | OR(x0, x1) |          Note         |
    +=======+=======================+============+=======================+
    | ``0`` |                 ``0`` |      ``0`` |                       |
    +-------+-----------------------+------------+-----------------------+
    | ``0`` |                 ``1`` |      ``1`` |                       |
    +-------+-----------------------+------------+-----------------------+
    | ``1`` |                 ``0`` |      ``1`` |                       |
    +-------+-----------------------+------------+-----------------------+
    | ``1`` |                 ``1`` |      ``1`` |                       |
    +-------+-----------------------+------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |      ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+------------+-----------------------+
    | ``1`` |                 ``-`` |      ``1`` | ``1`` dominates ``-`` |
    +-------+-----------------------+------------+-----------------------+
    | ``-`` |        {``0``, ``-``} |      ``-`` | ``-`` dominates ``0`` |
    +-------+-----------------------+------------+-----------------------+

    For example:

    >>> or_("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b-1-X_111X_-10X_XXXX")

    In expressions, you can use the binary ``|`` operator:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> a | b
    bits("16b-1-X_111X_-10X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _or_(y, x)
    return y


def nand(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical NAND operator.

    Perform logical NAND on each bit of the inputs:

    +-------+-----------------------+--------------+-----------------------+
    |   x0  |           x1          | NAND(x0, x1) |          Note         |
    +=======+=======================+==============+=======================+
    | ``0`` |                 ``0`` |        ``1`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``0`` |                 ``1`` |        ``1`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``1`` |                 ``0`` |        ``1`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``1`` |                 ``1`` |        ``0`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |        ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+--------------+-----------------------+
    | ``0`` |                 ``-`` |        ``1`` | ``0`` dominates ``-`` |
    +-------+-----------------------+--------------+-----------------------+
    | ``-`` |        {``1``, ``-``} |        ``-`` | ``-`` dominates ``1`` |
    +-------+-----------------------+--------------+-----------------------+

    For example:

    >>> nand("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b--1X_-01X_111X_XXXX")

    In expressions, you can use the unary ``~`` and binary ``&`` operators:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> ~(a & b)
    bits("16b--1X_-01X_111X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    return _not_(and_(x0, *xs))


def and_(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical AND operator.

    Perform logical AND on each bit of the inputs:

    +-------+-----------------------+-------------+-----------------------+
    |   x0  |           x1          | AND(x0, x1) |          Note         |
    +=======+=======================+=============+=======================+
    | ``0`` |                 ``0`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``0`` |                 ``1`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``0`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``1`` |       ``1`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |       ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+-------------+-----------------------+
    | ``0`` |                 ``-`` |       ``0`` | ``0`` dominates ``-`` |
    +-------+-----------------------+-------------+-----------------------+
    | ``-`` |        {``1``, ``-``} |       ``-`` | ``-`` dominates ``1`` |
    +-------+-----------------------+-------------+-----------------------+

    For example:

    >>> and_("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b--0X_-10X_000X_XXXX")

    In expressions, you can use the binary ``&`` operator:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> a & b
    bits("16b--0X_-10X_000X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _and_(y, x)
    return y


def xnor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical XNOR operator.

    Perform logical XNOR on each bit of the inputs:

    +-------+-----------------------+--------------+-----------------------+
    |   x0  |           x1          | XNOR(x0, x1) |          Note         |
    +=======+=======================+==============+=======================+
    | ``0`` |                 ``0`` |        ``1`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``0`` |                 ``1`` |        ``0`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``1`` |                 ``0`` |        ``0`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``1`` |                 ``1`` |        ``1`` |                       |
    +-------+-----------------------+--------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |        ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+--------------+-----------------------+
    | ``-`` | {``0``, ``1``. ``-``} |        ``-`` | ``-`` dominates known |
    +-------+-----------------------+--------------+-----------------------+

    For example:

    >>> xnor("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b---X_-10X_-01X_XXXX")

    In expressions, you can use the unary ``~`` and binary ``^`` operators:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> ~(a ^ b)
    bits("16b---X_-10X_-01X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    return _not_(xor(x0, *xs))


def xor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """N-ary bitwise logical XOR operator.

    Perform logical XOR on each bit of the inputs:

    +-------+-----------------------+-------------+-----------------------+
    |   x0  |           x1          | XOR(x0, x1) |          Note         |
    +=======+=======================+=============+=======================+
    | ``0`` |                 ``0`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``0`` |                 ``1`` |       ``1`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``0`` |       ``1`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``1`` |                 ``1`` |       ``0`` |                       |
    +-------+-----------------------+-------------+-----------------------+
    | ``X`` | {``0``, ``1``, ``-``} |       ``X`` |  ``X`` dominates all  |
    +-------+-----------------------+-------------+-----------------------+
    | ``-`` | {``0``, ``1``. ``-``} |       ``-`` | ``-`` dominates known |
    +-------+-----------------------+-------------+-----------------------+

    For example:

    >>> xor("16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b---X_-01X_-10X_XXXX")

    In expressions, you can use the binary ``^`` operator:

    >>> a = bits("16b----_1111_0000_XXXX")
    >>> b = bits("16b-10X_-10X_-10X_-10X")
    >>> a ^ b
    bits("16b---X_-01X_-10X_XXXX")

    Args:
        x0: ``Bits`` or string literal.
        xs: Sequence of ``Bits`` equal size to ``x0``.

    Returns:
        ``Bits`` equal size to ``x0``.

    Raises:
        TypeError: ``x0`` is not a valid ``Bits`` object,
                   or ``xs[i]`` not equal size to ``x0``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _xor_(y, x)
    return y


def _ite(s: Bits, x1: Bits, x0: Bits) -> Bits:
    s0 = mask(x1.size) * s.data[0]
    s1 = mask(x1.size) * s.data[1]
    d0, d1 = lite((s0, s1), x1.data, x0.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def ite(s: Bits | str, x1: Bits | str, x0: Bits | str) -> Bits:
    """Ternary bitwise logical if-then-else (ITE) operator.

    Perform logical ITE on each bit of the inputs:

    +-------+-----------------------+-----------------------+----------------+
    |   s   |           x1          |           x0          | ITE(s, x1, x0) |
    +=======+=======================+=======================+================+
    | ``1`` | {``0``, ``1``, ``-``} |                       |         ``x1`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``0`` |                       | {``0``, ``1``, ``-``} |         ``x0`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``X`` |                       |                       |          ``X`` |
    +-------+-----------------------+-----------------------+----------------+
    |       |                 ``X`` |                       |          ``X`` |
    +-------+-----------------------+-----------------------+----------------+
    |       |                       |                 ``X`` |          ``X`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``-`` |                 ``0`` |                 ``0`` |          ``0`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``-`` |                 ``0`` |        {``1``, ``-``} |          ``-`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``-`` |                 ``1`` |                 ``1`` |          ``1`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``-`` |                 ``1`` |        {``0``, ``-``} |          ``-`` |
    +-------+-----------------------+-----------------------+----------------+
    | ``-`` |                 ``-`` | {``0``, ``1``, ``-``} |          ``-`` |
    +-------+-----------------------+-----------------------+----------------+

    For example:

    >>> ite("1b0", "16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b-10X_-10X_-10X_XXXX")
    >>> ite("1b1", "16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b---X_111X_000X_XXXX")
    >>> ite("1b-", "16b----_1111_0000_XXXX", "16b-10X_-10X_-10X_-10X")
    bits("16b---X_-1-X_--0X_XXXX")

    Args:
        s: ``Bits`` select
        x1: ``Bits`` or string literal.
        x0: ``Bits`` or string literal equal size to ``x1``.

    Returns:
        ``Bits`` equal size to ``x1``.

    Raises:
        TypeError: ``s`` or ``x1`` are not valid ``Bits`` objects,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    s = _expect_size(s, 1)
    x1 = _expect_type(x1, Bits)
    x0 = _expect_size(x0, x1.size)
    return _ite(s, x1, x0)


def _mux(s: Bits, t: type[Bits], xs: dict[int, Bits]) -> Bits:
    m = mask(t.size)
    si = (s._get_index(i) for i in range(s.size))
    s = tuple((m * d0, m * d1) for d0, d1 in si)
    dc = t.dcs()
    d0, d1 = lmux(s, {i: x.data for i, x in xs.items()}, dc.data)
    return t._cast_data(d0, d1)


_MUX_XN_RE = re.compile(r"x(\d+)")


def mux(s: Bits | str, **xs: Bits | str) -> Bits:
    r"""Bitwise logical multiplex (mux) operator.

    Args:
        s: ``Bits`` select.
        xs: ``Bits`` or string literal, all equal size.

    Mux input names are in the form xN,
    where N is a valid int.
    Muxes require at least one input.
    Any inputs not specified will default to "don't care".

    For example:

    >>> mux("2b00", x0="4b0001", x1="4b0010", x2="4b0100", x3="4b1000")
    bits("4b0001")
    >>> mux("2b10", x0="4b0001", x1="4b0010", x2="4b0100", x3="4b1000")
    bits("4b0100")

    Handles X and DC propagation:

    >>> mux("2b1-", x0="4b0001", x1="4b0010", x2="4b0100", x3="4b1000")
    bits("4b--00")
    >>> mux("2b1X", x0="4b0001", x1="4b0010", x2="4b0100", x3="4b1000")
    bits("4bXXXX")

    Returns:
        ``Bits`` equal size to ``xN`` inputs.

    Raises:
        TypeError: ``s`` or ``xN`` are not valid ``Bits`` objects,
                   or ``xN`` mismatching size.
        ValueError: Error parsing string literal.
    """
    s = _expect_type(s, Bits)
    n = 1 << s.size

    # Parse and check inputs
    t = None
    i2x = {}
    for name, value in xs.items():
        if m := _MUX_XN_RE.match(name):
            i = int(m.group(1))
            if not 0 <= i < n:
                raise ValueError(f"Expected x in [x0, ..., x{n - 1}]; got {name}")
            if t is None:
                x = _expect_type(value, Bits)
                t = type(x)
            else:
                x = _expect_size(value, t.size)
                t = _resolve_type(t, type(x))
            i2x[i] = x
        else:
            raise ValueError(f"Invalid input name: {name}")

    if t is None:
        raise ValueError("Expected at least one mux input")

    return _mux(s, t, i2x)


# Unary
def _uor(x: Bits) -> Scalar:
    y = _0
    for i in range(x.size):
        y = lor(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uor(x: Bits | str) -> Scalar:
    """Unary OR reduction operator.

    The identity of OR is ``0``.
    Compute an OR-sum over all the bits of ``x``.

    For example:

    >>> uor("4b1000")
    bits("1b1")

    Empty input returns identity:

    >>> uor(bits())
    bits("1b0")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    return _uor(x)


def _uand(x: Bits) -> Scalar:
    y = _1
    for i in range(x.size):
        y = land(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uand(x: Bits | str) -> Scalar:
    """Unary AND reduction operator.

    The identity of AND is ``1``.
    Compute an AND-sum over all the bits of ``x``.

    For example:

    >>> uand("4b0111")
    bits("1b0")

    Empty input returns identity:

    >>> uand(bits())
    bits("1b1")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    return _uand(x)


def _uxnor(x: Bits) -> Scalar:
    y = _1
    for i in range(x.size):
        y = lxnor(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uxnor(x: Bits | str) -> Scalar:
    """Unary XNOR reduction operator.

    The identity of XOR is ``0``.
    Compute an XNOR-sum (even parity) over all the bits of ``x``.

    For example:

    >>> uxnor("4b1010")
    bits("1b1")

    Empty input returns identity:

    >>> uxnor(bits())
    bits("1b1")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    return _uxnor(x)


def _uxor(x: Bits) -> Scalar:
    y = _0
    for i in range(x.size):
        y = lxor(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uxor(x: Bits | str) -> Scalar:
    """Unary XOR reduction operator.

    The identity of XOR is ``0``.
    Compute an XOR-sum (odd parity) over all the bits of ``x``.

    For example:

    >>> uxor("4b1010")
    bits("1b0")

    Empty input returns identity:

    >>> uxor(bits())
    bits("1b0")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    return _uxor(x)


# Arithmetic
def decode(x: Bits | str) -> Scalar | Vector:
    """Decode dense encoding to sparse, one-hot encoding.

    For example:

    >>> decode("2b00")
    bits("4b0001")
    >>> decode("2b01")
    bits("4b0010")
    >>> decode("2b10")
    bits("4b0100")
    >>> decode("2b11")
    bits("4b1000")

    Empty input returns 1b1:

    >>> decode(bits())
    bits("1b1")

    Args:
        x: ``Bits`` or string literal.

    Returns:
        One hot ``Scalar`` or ``Vector`` w/ ``size`` = ``2**x.size``

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)

    # Output has 2^N bits
    n = 1 << x.size
    vec = _vec_size(n)

    # X/DC propagation
    if x.has_x():
        return vec.xes()
    if x.has_dc():
        return vec.dcs()

    d1 = 1 << x.to_uint()
    return vec(d1 ^ mask(n), d1)


def _add(a: Bits, b: Bits, ci: Scalar) -> tuple[Bits, Scalar]:
    # X/DC propagation
    if a.has_x() or b.has_x() or ci.has_x():
        return a.xes(), _ScalarX
    if a.has_dc() or b.has_dc() or ci.has_dc():
        return a.dcs(), _ScalarW

    dmax = mask(a.size)
    s = a.data[1] + b.data[1] + ci.data[1]
    co = _bool2scalar[s > dmax]
    s &= dmax

    t = _resolve_type(type(a), type(b))
    return t._cast_data(s ^ dmax, s), co


def add(a: Bits | str, b: Bits | str, ci: Scalar | str | None = None) -> Bits:
    """Addition with carry-in, but NO carry-out.

    For example:

    >>> add("4d2", "4d2")
    bits("4b0100")

    >>> add("2d2", "2d2")
    bits("2b00")

    Args:
        a: ``Bits`` or string literal
        b: ``Bits`` or string literal equal size to ``a``.
        ci: ``Scalar`` carry-in, or ``None``.
            ``None`` defaults to carry-in ``0``.

    Returns:
        ``Bits`` sum equal size to ``a`` and ``b``.

    Raises:
        TypeError: ``a``, ``b``, or ``ci`` are not valid ``Bits`` objects,
                   or ``a`` not equal size to ``b``.
        ValueError: Error parsing string literal.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    ci = _Scalar0 if ci is None else _expect_type(ci, Scalar)
    s, _ = _add(a, b, ci)
    return s


def adc(a: Bits | str, b: Bits | str, ci: Scalar | str | None = None) -> Bits:
    """Addition with carry-in, and carry-out.

    For example:

    >>> adc("4d2", "4d2")
    bits("5b0_0100")

    >>> adc("2d2", "2d2")
    bits("3b100")

    Args:
        a: ``Bits`` or string literal
        b: ``Bits`` or string literal equal size to ``a``.
        ci: ``Scalar`` carry-in, or ``None``.
            ``None`` defaults to carry-in ``0``.

    Returns:
        ``Bits`` sum w/ size one larger than ``a`` and ``b``.
        The most significant bit is the carry-out.

    Raises:
        TypeError: ``a``, ``b``, or ``ci`` are not valid ``Bits`` objects,
                   or ``a`` not equal size to ``b``.
        ValueError: Error parsing string literal.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    ci = _Scalar0 if ci is None else _expect_type(ci, Scalar)
    s, co = _add(a, b, ci)
    return cat(s, co)


def _sub(a: Bits, b: Bits) -> tuple[Bits, Scalar]:
    return _add(a, _not_(b), ci=_Scalar1)


def sub(a: Bits | str, b: Bits | str) -> Bits:
    """Twos complement subtraction, with NO carry-out.

    Args:
        a: ``Bits`` or string literal
        b: ``Bits`` or string literal equal size to ``a``.

    Returns:
        ``Bits`` sum equal size to ``a`` and ``b``.

    Raises:
        TypeError: ``a``, or ``b`` are not valid ``Bits`` objects,
                   or ``a`` not equal size to ``b``.
        ValueError: Error parsing string literal.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    s, _ = _sub(a, b)
    return s


def sbc(a: Bits | str, b: Bits | str) -> Bits:
    """Twos complement subtraction, with carry-out.

    Args:
        a: ``Bits`` or string literal
        b: ``Bits`` or string literal equal size to ``a``.

    Returns:
        ``Bits`` sum w/ size one larger than ``a`` and ``b``.
        The most significant bit is the carry-out.

    Raises:
        TypeError: ``a``, or ``b`` are not valid ``Bits`` objects,
                   or ``a`` not equal size to ``b``.
        ValueError: Error parsing string literal.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    s, co = _sub(a, b)
    return cat(s, co)


def _neg(x: Bits) -> tuple[Bits, Scalar]:
    return _add(x.zeros(), _not_(x), ci=_Scalar1)


def neg(x: Bits | str) -> Bits:
    """Twos complement negation, with NO carry-out.

    Args:
        x: ``Bits`` or string literal

    Returns:
        ``Bits`` equal size to ``x``.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    s, _ = _neg(x)
    return s


def ngc(x: Bits | str) -> Bits:
    """Twos complement negation, with carry-out.

    Args:
        x: ``Bits`` or string literal

    Returns:
        ``Bits`` w/ size one larger than ``x``.
        The most significant bit is the carry-out.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: Error parsing string literal.
    """
    x = _expect_type(x, Bits)
    s, co = _neg(x)
    return cat(s, co)


def _mul(a: Bits, b: Bits) -> Empty | Vector:
    n = 2 * a.size

    # X/DC propagation
    if a.has_x() or b.has_x():
        return Vector[n].xes()
    if a.has_dc() or b.has_dc():
        return Vector[n].dcs()

    dmax = mask(n)
    p = a.data[1] * b.data[1]

    return _vec_size(n)(p ^ dmax, p)


def mul(a: Bits | str, b: Bits | str) -> Empty | Vector:
    """Unsigned multiply.

    For example:

    >>> mul("4d2", "4d2")
    bits("8b0000_0100")

    >>> add("2d2", "2d2")
    bits("2b00")

    Args:
        a: ``Bits`` or string literal
        b: ``Bits`` or string literal equal size to ``a``.

    Returns:
        ``Vector`` product w/ size 2 * ``a.size``

    Raises:
        TypeError: ``a`` or ``b`` are not valid ``Bits`` objects,
                   or ``a`` not equal size to ``b``.
        ValueError: Error parsing string literal.
    """
    a = _expect_type(a, Bits)
    b = _expect_size(b, a.size)
    return _mul(a, b)


def _lsh(x: Bits, n: Bits) -> Bits:
    if n.has_x():
        return x.xes()
    if n.has_dc():
        return x.dcs()

    n = n.to_uint()
    if n == 0:
        return x
    if n > x.size:
        raise ValueError(f"Expected n ≤ {x.size}, got {n}")

    _, (sh0, sh1) = x._get_slice(0, x.size - n)
    d0 = mask(n) | sh0 << n
    d1 = sh1 << n
    y = x._cast_data(d0, d1)

    return y


def lsh(x: Bits | str, n: Bits | str | int) -> Bits:
    """Logical left shift by n bits.

    Fill bits with zeros.

    For example:

    >>> lsh("4b1011", 2)
    bits("4b1100")

    Args:
        x: ``Bits`` or string literal.
        n: ``Bits``, string literal, or ``int``
           Non-negative bit shift count.

    Returns:
        ``Bits`` left-shifted by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object,
                   or ``n`` is not a valid bit shift count.
        ValueError: Error parsing string literal,
                    or negative shift amount.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _lsh(x, n)


def _rsh(x: Bits, n: Bits) -> Bits:
    if n.has_x():
        return x.xes()
    if n.has_dc():
        return x.dcs()

    n = n.to_uint()
    if n == 0:
        return x
    if n > x.size:
        raise ValueError(f"Expected n ≤ {x.size}, got {n}")

    sh_size, (sh0, sh1) = x._get_slice(n, x.size)
    d0 = sh0 | (mask(n) << sh_size)
    d1 = sh1
    y = x._cast_data(d0, d1)

    return y


def rsh(x: Bits | str, n: Bits | str | int) -> Bits:
    """Logical right shift by n bits.

    Fill bits with zeros.

    For example:

    >>> rsh("4b1101", 2)
    bits("4b0011")

    Args:
        x: ``Bits`` or string literal.
        n: ``Bits``, string literal, or ``int``
           Non-negative bit shift count.

    Returns:
        ``Bits`` right-shifted by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object,
                   or ``n`` is not a valid bit shift count.
        ValueError: Error parsing string literal,
                    or negative shift amount.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _rsh(x, n)


def _srsh(x: Bits, n: Bits) -> Bits:
    if n.has_x():
        return x.xes()
    if n.has_dc():
        return x.dcs()

    n = n.to_uint()
    if n == 0:
        return x
    if n > x.size:
        raise ValueError(f"Expected n ≤ {x.size}, got {n}")

    sign0, sign1 = x._get_index(x.size - 1)
    si0, si1 = mask(n) * sign0, mask(n) * sign1

    sh_size, (sh0, sh1) = x._get_slice(n, x.size)
    d0 = sh0 | si0 << sh_size
    d1 = sh1 | si1 << sh_size
    y = x._cast_data(d0, d1)

    return y


def srsh(x: Bits | str, n: Bits | str | int) -> Bits:
    """Arithmetic (signed) right shift by n bits.

    Fill bits with most significant bit (sign).

    For example:

    >>> srsh("4b1101", 2)
    bits("4b1111")

    Args:
        x: ``Bits`` or string literal.
        n: ``Bits``, string literal, or ``int``
           Non-negative bit shift count.

    Returns:
        ``Bits`` right-shifted by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object,
                   or ``n`` is not a valid bit shift count.
        ValueError: Error parsing string literal,
                    or negative shift amount.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _srsh(x, n)


# Word operations
def _xt(x: Bits, n: int) -> Vector:
    ext0 = mask(n)
    d0 = x.data[0] | ext0 << x.size
    d1 = x.data[1]
    return _vec_size(x.size + n)(d0, d1)


def xt(x: Bits | str, n: int) -> Bits:
    """Unsigned extend by n bits.

    Fill high order bits with zero.

    For example:

    >>> xt("2b11", 2)
    bits("4b0011")

    Args:
        x: ``Bits`` or string literal.
        n: Non-negative number of bits.

    Returns:
        ``Bits`` zero-extended by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: If n is negative.
    """
    x = _expect_type(x, Bits)

    if n < 0:
        raise ValueError(f"Expected n ≥ 0, got {n}")
    if n == 0:
        return x

    return _xt(x, n)


def _sxt(x: Bits, n: int) -> Vector:
    sign0, sign1 = x._get_index(x.size - 1)
    ext0 = mask(n) * sign0
    ext1 = mask(n) * sign1
    d0 = x.data[0] | ext0 << x.size
    d1 = x.data[1] | ext1 << x.size
    return _vec_size(x.size + n)(d0, d1)


def sxt(x: Bits | str, n: int) -> Bits:
    """Sign extend by n bits.

    Fill high order bits with sign.

    For example:

    >>> sxt("2b11", 2)
    bits("4b1111")

    Args:
        x: ``Bits`` or string literal.
        n: Non-negative number of bits.

    Returns:
        ``Bits`` sign-extended by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object.
        ValueError: If n is negative.
    """
    x = _expect_type(x, Bits)

    if n < 0:
        raise ValueError(f"Expected n ≥ 0, got {n}")
    if n == 0:
        return x

    return _sxt(x, n)


def _lrot(x: Bits, n: Bits) -> Bits:
    if n.has_x():
        return x.xes()
    if n.has_dc():
        return x.dcs()

    n = n.to_uint()
    if n == 0:
        return x
    if n >= x.size:
        raise ValueError(f"Expected n < {x.size}, got {n}")

    _, (co0, co1) = x._get_slice(x.size - n, x.size)
    _, (sh0, sh1) = x._get_slice(0, x.size - n)
    d0 = co0 | sh0 << n
    d1 = co1 | sh1 << n
    return x._cast_data(d0, d1)


def lrot(x: Bits | str, n: Bits | str | int) -> Bits:
    """Rotate left by n bits.

    For example:

    >>> lrot("4b1011", 2)
    bits("4b1110")

    Args:
        x: ``Bits`` or string literal.
        n: ``Bits``, string literal, or ``int``
           Non-negative bit rotate count.

    Returns:
        ``Bits`` left-rotated by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object,
                   or ``n`` is not a valid bit rotate count.
        ValueError: Error parsing string literal,
                    or negative rotate amount.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _lrot(x, n)


def _rrot(x: Bits, n: Bits) -> Bits:
    if n.has_x():
        return x.xes()
    if n.has_dc():
        return x.dcs()

    n = n.to_uint()
    if n == 0:
        return x
    if n >= x.size:
        raise ValueError(f"Expected n < {x.size}, got {n}")

    _, (co0, co1) = x._get_slice(0, n)
    sh_size, (sh0, sh1) = x._get_slice(n, x.size)
    d0 = sh0 | co0 << sh_size
    d1 = sh1 | co1 << sh_size
    return x._cast_data(d0, d1)


def rrot(x: Bits | str, n: Bits | str | int) -> Bits:
    """Rotate right by n bits.

    For example:

    >>> rrot("4b1101", 2)
    bits("4b0111")

    Args:
        x: ``Bits`` or string literal.
        n: ``Bits``, string literal, or ``int``
           Non-negative bit rotate count.

    Returns:
        ``Bits`` right-rotated by n bits.

    Raises:
        TypeError: ``x`` is not a valid ``Bits`` object,
                   or ``n`` is not a valid bit rotate count.
        ValueError: Error parsing string literal,
                    or negative rotate amount.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _rrot(x, n)


def _cat(*xs: Bits) -> Empty | Scalar | Vector:
    if len(xs) == 0:
        return _Empty
    if len(xs) == 1:
        return xs[0]

    size = 0
    d0, d1 = 0, 0
    for x in xs:
        d0 |= x.data[0] << size
        d1 |= x.data[1] << size
        size += x.size
    return _vec_size(size)(d0, d1)


def cat(*objs: Bits | int | str) -> Empty | Scalar | Vector:
    """Concatenate a sequence of Vectors.

    Args:
        objs: a sequence of vec/bool/lit objects.

    Returns:
        A Vec instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    # Convert inputs
    xs = []
    for obj in objs:
        if isinstance(obj, Bits):
            xs.append(obj)
        elif obj in (0, 1):
            xs.append(_bool2scalar[obj])
        elif isinstance(obj, str):
            x = _lit2bv(obj)
            xs.append(x)
        else:
            raise TypeError(f"Invalid input: {obj}")
    return _cat(*xs)


def _rep(x: Bits, n: int) -> Empty | Scalar | Vector:
    xs = [x] * n
    return _cat(*xs)


def rep(obj: Bits | int | str, n: int) -> Empty | Scalar | Vector:
    """Repeat a Vector n times."""
    objs = [obj] * n
    return cat(*objs)


def _pack(x: Bits, n: int) -> Bits:
    if x.size == 0:
        return x

    m = mask(n)

    xd0, xd1 = x.data

    d0 = xd0 & m
    d1 = xd1 & m

    for _ in range(n, x.size, n):
        xd0 >>= n
        xd1 >>= n
        d0 = (d0 << n) | (xd0 & m)
        d1 = (d1 << n) | (xd1 & m)

    return x._cast_data(d0, d1)


def pack(x: Bits | str, n: int = 1) -> Bits:
    """Pack n-bit blocks in right to left order."""
    if n < 1:
        raise ValueError(f"Expected n < 1, got {n}")

    x = _expect_type(x, Bits)
    if x.size % n != 0:
        raise ValueError("Expected x.size to be a multiple of n")

    return _pack(x, n)


def match(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Pattern match operator.

    Similar to ``eq`` operator, but with support for ``-`` wildcards.

    For example:

    >>> match("2b01", "2b0-")
    bits("1b1")
    >>> match("2b--", "2b10")
    bits("1b1")
    >>> match("2b01", "2b10")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)

    # Propagate X
    if x0.has_x() or x1.has_x():
        return _ScalarX

    for i in range(x0.size):
        a0, a1 = x0._get_index(i)
        b0, b1 = x1._get_index(i)
        if a0 ^ b0 and a1 ^ b1:
            return _Scalar0
    return _Scalar1


# Predicates over bitvectors
def _eq(x0: Bits, x1: Bits) -> Scalar:
    return _uand(_xnor_(x0, x1))


def eq(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Equal (==) reduction operator.

    Equivalent to ``uand(xnor(x0, x1))``.

    For example:

    >>> eq("2b01", "2b00")
    bits("1b0")
    >>> eq("2b01", "2b01")
    bits("1b1")
    >>> eq("2b01", "2b10")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _eq(x0, x1)


def _ne(x0: Bits, x1: Bits) -> Scalar:
    return _uor(_xor_(x0, x1))


def ne(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical NotEqual (!=) reduction operator.

    Equivalent to ``uor(xor(x0, x1))``.

    For example:

    >>> ne("2b01", "2b00")
    bits("1b1")
    >>> ne("2b01", "2b01")
    bits("1b0")
    >>> ne("2b01", "2b10")
    bits("1b1")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _ne(x0, x1)


def _cmp(op: Callable, x0: Bits, x1: Bits) -> Scalar:
    # X/DC propagation
    if x0.has_x() or x1.has_x():
        return _ScalarX
    if x0.has_dc() or x1.has_dc():
        return _ScalarW
    return _bool2scalar[op(x0.to_uint(), x1.to_uint())]


def lt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Unsigned LessThan (<) reduction operator.

    Returns ``Scalar`` result of ``x0.to_uint() < x1.to_uint()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> lt("2b01", "2b00")
    bits("1b0")
    >>> lt("2b01", "2b01")
    bits("1b0")
    >>> lt("2b01", "2b10")
    bits("1b1")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.lt, x0, x1)


def le(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Unsigned LessThanOrEqual (≤) reduction operator.

    Returns ``Scalar`` result of ``x0.to_uint() <= x1.to_uint()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> le("2b01", "2b00")
    bits("1b0")
    >>> le("2b01", "2b01")
    bits("1b1")
    >>> le("2b01", "2b10")
    bits("1b1")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.le, x0, x1)


def gt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Unsigned GreaterThan (>) reduction operator.

    Returns ``Scalar`` result of ``x0.to_uint() > x1.to_uint()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> gt("2b01", "2b00")
    bits("1b1")
    >>> gt("2b01", "2b01")
    bits("1b0")
    >>> gt("2b01", "2b10")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.gt, x0, x1)


def ge(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Unsigned GreaterThanOrEqual (≥) reduction operator.

    Returns ``Scalar`` result of ``x0.to_uint() >= x1.to_uint()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> ge("2b01", "2b00")
    bits("1b1")
    >>> ge("2b01", "2b01")
    bits("1b1")
    >>> ge("2b01", "2b10")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.ge, x0, x1)


def _scmp(op: Callable, x0: Bits, x1: Bits) -> Scalar:
    # X/DC propagation
    if x0.has_x() or x1.has_x():
        return _ScalarX
    if x0.has_dc() or x1.has_dc():
        return _ScalarW
    return _bool2scalar[op(x0.to_int(), x1.to_int())]


def slt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Signed LessThan (<) reduction operator.

    Returns ``Scalar`` result of ``x0.to_int() < x1.to_int()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> slt("2b00", "2b11")
    bits("1b0")
    >>> slt("2b00", "2b00")
    bits("1b0")
    >>> slt("2b00", "2b01")
    bits("1b1")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.lt, x0, x1)


def sle(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Signed LessThanOrEqual (≤) reduction operator.

    Returns ``Scalar`` result of ``x0.to_int() <= x1.to_int()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> sle("2b00", "2b11")
    bits("1b0")
    >>> sle("2b00", "2b00")
    bits("1b1")
    >>> sle("2b00", "2b01")
    bits("1b1")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.le, x0, x1)


def sgt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Signed GreaterThan (>) reduction operator.

    Returns ``Scalar`` result of ``x0.to_int() > x1.to_int()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> sgt("2b00", "2b11")
    bits("1b1")
    >>> sgt("2b00", "2b00")
    bits("1b0")
    >>> sgt("2b00", "2b01")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.gt, x0, x1)


def sge(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Binary logical Signed GreaterThanOrEqual (≥) reduction operator.

    Returns ``Scalar`` result of ``x0.to_int() >= x1.to_int()``.
    For performance reasons, use simple ``X``/``-`` propagation:
    ``X`` dominates {``-``, known}, and ``-`` dominates known.

    For example:

    >>> sge("2b00", "2b11")
    bits("1b1")
    >>> sge("2b00", "2b00")
    bits("1b1")
    >>> sge("2b00", "2b01")
    bits("1b0")

    Args:
        x0: ``Bits`` or string literal.
        x1: ``Bits`` or string literal equal size to ``x0``.

    Returns:
        ``Scalar``

    Raises:
        TypeError: ``x0`` or ``x1`` is not a valid ``Bits`` object,
                   or ``x0`` not equal size to ``x1``.
        ValueError: Error parsing string literal.
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.ge, x0, x1)


_LIT_PREFIX_RE = re.compile(r"(?P<Size>[1-9][0-9]*)(?P<Base>[bdh])")


def _parse_lit(lit: str) -> tuple[int, lbv]:
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
            dmax = mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        # Hexadecimal
        if base == "h":
            d1 = int(digits, base=16)
            dmax = mask(size)
            if d1 > dmax:
                s = f"Expected digits in range [0, {dmax}], got {digits}"
                raise ValueError(s)
            return size, (d1 ^ dmax, d1)
        assert False  # pragma: no cover
    raise ValueError(f"Invalid lit: {lit}")


def _lit2bv(lit: str) -> Scalar | Vector:
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


def _bools2vec(x0: int, *xs: int) -> Empty | Scalar | Vector:
    """Convert an iterable of bools to a vec.

    This is a convenience function.
    For data in the form of [0, 1, 0, 1, ...],
    or [False, True, False, True, ...].
    """
    size = 1
    d1 = int(x0)
    for x in xs:
        if x in (0, 1):
            d1 |= x << size
        else:
            raise TypeError(f"Expected x in {{0, 1}}, got {x}")
        size += 1
    return _vec_size(size)(d1 ^ mask(size), d1)


def _rank2(fst: Scalar | Vector, *rst: Scalar | Vector | str) -> Vector | Array:
    d0, d1 = fst.data
    for i, x in enumerate(rst, start=1):
        x = _expect_type(x, Vector[fst.size])
        d0 |= x.data[0] << (fst.size * i)
        d1 |= x.data[1] << (fst.size * i)
    if fst.shape == (1,):
        size = len(rst) + 1
        return _get_vec_size(size)(d0, d1)
    shape = (len(rst) + 1,) + fst.shape
    return _get_array_shape(shape)(d0, d1)


def bits(obj=None) -> Empty | Scalar | Vector | Array:
    """Create a shaped Bits object using standard input formats.

    For example, empty input returns an ``Empty`` instance.

    >>> bits()
    bits([])
    >>> bits(None)
    bits([])

    ``bool``, ``int``, and string literal inputs:

    >>> bits(False)
    bits("1b0")
    >>> bits([False, True, False, True])
    bits("4b1010")
    >>> bits("8d42")
    bits("8b0010_1010")

    Use a ``list`` of inputs to create arbitrary shaped inputs:

    >>> x = bits([["2b00", "2b01"], ["2b10", "2b11"]])
    >>> x
    bits([["2b00", "2b01"],
          ["2b10", "2b11"]])
    >>> x.shape
    (2, 2, 2)

    Args:
        obj: Object that can be converted to a Bits instance.

    Returns:
        Shaped ``Bits`` instance.

    Raises:
        TypeError: If input obj is invalid.
    """
    match obj:
        case None | []:
            return _Empty
        case 0 | 1 as x:
            return _bool2scalar[x]
        case [0 | 1 as fst, *rst]:
            return _bools2vec(fst, *rst)
        case str() as lit:
            return _lit2bv(lit)
        case [str() as lit, *rst]:
            x = _lit2bv(lit)
            return _rank2(x, *rst)
        case [Scalar() as x, *rst]:
            return _rank2(x, *rst)
        case [Vector() as x, *rst]:
            return _rank2(x, *rst)
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
    xs = []
    for obj in objs:
        if isinstance(obj, _ShapeIf):
            xs.append(obj)
        elif obj in (0, 1):
            xs.append(_bool2scalar[obj])
        elif isinstance(obj, str):
            x = _lit2bv(obj)
            xs.append(x)
        else:
            raise TypeError(f"Invalid input: {obj}")

    if len(xs) == 1:
        return xs[0]

    fst, rst = xs[0], xs[1:]

    size = fst.size
    d0, d1 = fst.data
    for x in rst:
        if x.shape != fst.shape:
            s = f"Expected shape {fst.shape}, got {x.shape}"
            raise TypeError(s)
        d0 |= x.data[0] << size
        d1 |= x.data[1] << size
        size += x.size

    # {Empty, Empty, ...} => Empty
    if fst.shape == (0,):
        return _Empty
    # {Scalar, Scalar, ...} => Vector[K]
    if fst.shape == (1,):
        size = len(xs)
        return _vec_size(size)(d0, d1)
    # {Vector[K], Vector[K], ...} => Array[J,K]
    # {Array[J,K], Array[J,K], ...} => Array[I,J,K]
    shape = (len(xs),) + fst.shape
    return _get_array_shape(shape)(d0, d1)


def u2bv(n: int, size: int | None = None) -> Empty | Scalar | Vector:
    """Convert nonnegative int to Vector.

    For example:

    >>> u2bv(42, size=8)
    bits("8b0010_1010")

    Args:
        n: Nonnegative ``int`` to convert.
        size: Optional ``int`` output size.
              Defaults to minimum required size.

    Returns:
        ``Vector``

    Raises:
        ValueError: ``n`` is negative or overflows the output size.
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

    return _vec_size(size)(n ^ mask(size), n)


def i2bv(n: int, size: int | None = None) -> Scalar | Vector:
    """Convert int to Vector.

    For example:

    >>> i2bv(42, size=8)
    bits("8b0010_1010")
    >>> i2bv(-42, size=8)
    bits("8b1101_0110")

    Args:
        n: ``int`` to convert.
        size: Optional ``int`` output size.
              Defaults to minimum required size.

    Returns:
        ``Vector``

    Raises:
        ValueError: ``n`` overflows the output size.
    """
    negative = n < 0

    # Compute required number of bits
    if negative:
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

    x = _vec_size(size)(d1 ^ mask(size), d1)
    if negative:
        s, _ = _neg(x)
        return s
    return x


def _chunk(data: lbv, base: int, size: int) -> lbv:
    m = mask(size)
    return (data[0] >> base) & m, (data[1] >> base) & m


def _sel(x: _ShapeIf, key: tuple[tuple[int, int], ...]) -> _ShapeIf:
    assert len(x.shape) == len(key)

    (start, stop), key_r = key[0], key[1:]
    assert 0 <= start <= stop <= x.shape[0]

    # Partial select m:n
    if start != 0 or stop != x.shape[0]:

        if len(key_r) == 0:
            size = stop - start
            d0, d1 = _chunk(x.data, start, size)
            return _vec_size(size)(d0, d1)

        if len(key_r) == 1:
            vec = _get_vec_size(x.shape[1])
            xs = []
            for i in range(start, stop):
                d0, d1 = _chunk(x.data, vec.size * i, vec.size)
                xs.append(vec(d0, d1))
            return stack(*[_sel(x, key_r) for x in xs])

        array = _get_array_shape(x.shape[1:])
        xs = []
        for i in range(start, stop):
            d0, d1 = _chunk(x.data, array.size * i, array.size)
            xs.append(array(d0, d1))
        return stack(*[_sel(x, key_r) for x in xs])

    # Full select 0:n
    if key_r:
        return stack(*[_sel(xx, key_r) for xx in x])

    return x


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


def _get_sep(indent: str, x: Vector | Array) -> str:
    # 2-D Matrix
    if len(x.shape) == 2:
        return ", "
    # 3-D
    if len(x.shape) == 3:
        return ",\n" + indent
    # N-D
    return ",\n\n" + indent


def _array_repr(indent: str, x: Vector | Array) -> str:
    # 1-D Vector
    if len(x.shape) == 1:
        return f'"{x}"'
    sep = _get_sep(indent, x)
    f = partial(_array_repr, indent + " ")
    return "[" + sep.join(map(f, x)) + "]"


def _array_str(indent: str, x: Vector | Array) -> str:
    # 1-D Vector
    if len(x.shape) == 1:
        return f"{x}"
    sep = _get_sep(indent, x)
    f = partial(_array_str, indent + " ")
    return "[" + sep.join(map(f, x)) + "]"
