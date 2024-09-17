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
import operator
import random
import re
from collections import namedtuple
from collections.abc import Callable, Generator
from functools import cache, partial

from .lbool import (
    _W,
    _X,
    _0,
    _1,
    from_char,
    land,
    lite,
    lmux,
    lnot,
    lor,
    lxnor,
    lxor,
    to_char,
    to_vcd_char,
)
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
        x = _lit2vec(arg)
    else:
        x = arg
    if not isinstance(x, t):
        raise TypeError(f"Expected arg to be {t.__name__} or str literal")
    return x


def _expect_shift(arg, size: int) -> Bits:
    if isinstance(arg, int):
        return u2bv(arg, size)
    if isinstance(arg, str):
        return _lit2vec(arg)
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
    """Shaping interface."""

    @classproperty
    def shape(cls) -> tuple[int, ...]:
        raise NotImplementedError()  # pragma: no cover


class Bits:
    """Base class for Bits

    Bits[size]
    |
    +-- Empty[shape]
    |
    +-- Scalar[shape]
    |
    +-- Vector[shape] -- Enum
    |
    +-- Array[shape]
    |
    +-- Struct
    |
    +-- Union

    Do NOT construct a bit array directly.
    Use one of the factory functions:

        * bits
        * stack
        * u2bv
        * i2bv
    """

    @classproperty
    def size(cls) -> int:
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def cast(cls, x: Bits) -> Bits:
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
        return cls._cast_data(0, 0)

    @classmethod
    def zeros(cls) -> Bits:
        return cls._cast_data(cls._dmax, 0)

    @classmethod
    def ones(cls) -> Bits:
        return cls._cast_data(0, cls._dmax)

    @classmethod
    def dcs(cls) -> Bits:
        return cls._cast_data(cls._dmax, cls._dmax)

    @classmethod
    def rand(cls) -> Bits:
        d1 = random.getrandbits(cls.size)
        return cls._cast_data(cls._dmax ^ d1, d1)

    @classmethod
    def xprop(cls, sel: Bits) -> Bits:
        if sel.has_x():
            return cls.xes()
        return cls.dcs()

    @classproperty
    def _dmax(cls) -> int:
        return _mask(cls.size)

    @property
    def data(self) -> tuple[int, int]:
        return self._data

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
            return -(_not_(self).to_uint() + 1)
        return self.to_uint()

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
        d: int = self._data[0] & self._data[1]
        return d.bit_count()

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

    def _get_key(self, key: int | Bits | slice) -> tuple[int, tuple[int, int]]:
        if isinstance(key, int):
            index = _norm_index(self.size, key)
            return 1, self._get_index(index)
        if isinstance(key, Bits):
            index = _norm_index(self.size, key.to_uint())
            return 1, self._get_index(index)
        if isinstance(key, slice):
            start, stop = _norm_slice(self.size, key)
            if start != 0 or stop != self.size:
                return self._get_slice(start, stop)
            return self.size, self._data
        raise TypeError("Expected key to be int or slice")


class Empty(Bits, _ShapeIf):
    """Empty sequence of bits."""

    def __new__(cls, d0: int, d1: int):
        assert d0 == d1 == 0
        return _Empty

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
    """Zero dimensional (scalar) sequence of bits."""

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

    def __getitem__(self, key: int | Bits | slice) -> Empty | Scalar:
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
    """One dimensional (vector) sequence of bits."""

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

    def __getitem__(self, key: int | Bits | slice) -> Empty | Scalar | Vector:
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
    """N dimensional (array) sequence of bits."""

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

    def __getitem__(self, key: int | Bits | slice | tuple[int | slice | Bits, ...]) -> _ShapeIf:
        if isinstance(key, (int, Bits, slice)):
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
    def _norm_key(cls, keys: list[int | Bits | slice]) -> tuple[tuple[int, int], ...]:
        ndim = len(cls._shape)
        klen = len(keys)

        if klen > ndim:
            s = f"Expected ≤ {ndim} key items, got {klen}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(ndim - klen):
            keys.append(slice(None))

        # Normalize key dimensions
        def f(n: int, key: int | Bits | slice) -> tuple[int, int]:
            if isinstance(key, int):
                i = _norm_index(n, key)
                return (i, i + 1)
            if isinstance(key, Bits):
                i = _norm_index(n, key.to_uint())
                return (i, i + 1)
            if isinstance(key, slice):
                return _norm_slice(n, key)
            assert False  # pragma: no cover

        return tuple(f(n, key) for n, key in zip(cls._shape, keys))


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
                        s = f"Expected lit len {size}, got {size_i}"
                        raise ValueError(s)
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
        vec = _vec_size(size)
        enum = super().__new__(mcs, name, bases + (vec,), enum_attrs)

        # Instantiate members
        for (d0, d1), key in data2key.items():
            setattr(enum, key, enum._cast_data(d0, d1))

        # Override Vector._cast_data method
        def _cast_data(cls, d0: int, d1: int) -> Bits:
            data = (d0, d1)
            try:
                obj = getattr(cls, data2key[data])
            except KeyError:
                obj = object.__new__(cls)
                obj._data = data
            return obj

        # Override Vector._cast_data method
        enum._cast_data = classmethod(_cast_data)

        # Override Vector.__new__ method
        def _new(cls, arg: Bits | str):
            x = _expect_size(arg, cls.size)
            return cls.cast(x)

        enum.__new__ = _new

        # Override Vector.__repr__ method
        def _repr(self):
            try:
                return f"{name}.{data2key[self._data]}"
            except KeyError:
                return f'{name}("{vec.__str__(self)}")'

        enum.__repr__ = _repr

        # Override Vector.__str__ method
        def _str(self):
            try:
                return f"{name}.{data2key[self._data]}"
            except KeyError:
                return f"{name}({vec.__str__(self)})"

        enum.__str__ = _str

        # Create name property
        def _name(self):
            try:
                return data2key[self._data]
            except KeyError:
                return f"{name}({vec.__str__(self)})"

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

        # Get field_name: field_type items
        try:
            fields = list(attrs["__annotations__"].items())
        except KeyError as e:
            raise ValueError("Empty Struct is not supported") from e

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
                    x = _expect_size(arg, ft.size)
                    d0 |= x.data[0] << offsets[fn]
                    d1 |= x.data[1] << offsets[fn]
            obj._data = (d0, d1)

        source = _struct_init_source(fields)
        globals_ = {"_init_body": _init_body}
        locals_ = {}
        exec(source, globals_, locals_)
        struct.__init__ = locals_["init"]

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | Bits | slice) -> Empty | Scalar | Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        struct.__getitem__ = _getitem

        # Override Bits.__str__ method
        def _str(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!s},")
            parts.append(")")
            return "\n".join(parts)

        struct.__str__ = _str

        # Override Bits.__repr__ method
        def _repr(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!r},")
            parts.append(")")
            return "\n".join(parts)

        struct.__repr__ = _repr

        # Create Struct fields
        def _fget(fn: str, ft: type[Bits], self):
            mask = _mask(ft.size)
            d0 = (self._data[0] >> offsets[fn]) & mask
            d1 = (self._data[1] >> offsets[fn]) & mask
            return ft._cast_data(d0, d1)

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

        # Get field_name: field_type items
        try:
            fields = list(attrs["__annotations__"].items())
        except KeyError as e:
            raise ValueError("Empty Union is not supported") from e

        # Create Union class
        size = max(field_type.size for _, field_type in fields)
        union = super().__new__(mcs, name, bases + (Bits,), {})

        # Class properties
        union.size = classproperty(lambda _: size)

        # Override Bits.__init__ method
        def _init(self, arg: Bits | str):
            if isinstance(arg, str):
                x = _lit2vec(arg)
            else:
                x = arg
            ts = []
            for _, ft in fields:
                if ft not in ts:
                    ts.append(ft)
            if not isinstance(x, tuple(ts)):
                s = ", ".join(t.__name__ for t in ts)
                s = f"Expected arg to be {{{s}}}, or str literal"
                raise TypeError(s)
            self._data = x.data

        union.__init__ = _init

        # Override Bits.__getitem__ method
        def _getitem(self, key: int | Bits | slice) -> Empty | Scalar | Vector:
            size, (d0, d1) = self._get_key(key)
            return _vec_size(size)(d0, d1)

        union.__getitem__ = _getitem

        # Override Bits.__str__ method
        def _str(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!s},")
            parts.append(")")
            return "\n".join(parts)

        union.__str__ = _str

        # Override Bits.__repr__ method
        def _repr(self):
            parts = [f"{name}("]
            for fn, _ in fields:
                x = getattr(self, fn)
                parts.append(f"    {fn}={x!r},")
            parts.append(")")
            return "\n".join(parts)

        union.__repr__ = _repr

        # Create Union fields
        def _fget(ft: type[Bits], self):
            mask = _mask(ft.size)
            d0 = self._data[0] & mask
            d1 = self._data[1] & mask
            return ft._cast_data(d0, d1)

        for fn, ft in fields:
            setattr(union, fn, property(fget=partial(_fget, ft)))

        return union


class Union(metaclass=_UnionMeta):
    """Union Base Class: Create union."""


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
    """Bitwise NOT.

    f(x) -> y:
        X => X | 00 => 00
        0 => 1 | 01 => 10
        1 => 0 | 10 => 01
        - => - | 11 => 11

    Returns:
        Bits of equal size w/ inverted data.
    """
    x = _expect_type(x, Bits)
    return _not_(x)


def nor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise NOR.

    f(x0, x1) -> y:
        0 0 => 1
        1 - => 0
        X - => X
        - 0 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ NOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return _not_(or_(x0, *xs))


def or_(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise OR.

    f(x0, x1) -> y:
        0 0 => 0
        1 - => 1
        X - => X
        - 0 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ OR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _or_(y, x)
    return y


def nand(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise NAND.

    f(x0, x1) -> y:
        1 1 => 0
        0 - => 1
        X - => X
        - 1 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ NAND result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return _not_(and_(x0, *xs))


def and_(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise AND.

    f(x0, x1) -> y:
        1 1 => 1
        0 - => 0
        X - => X
        - 1 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ AND result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _and_(y, x)
    return y


def xnor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise XNOR.

    f(x0, x1) -> y:
        0 0 => 1
        0 1 => 0
        1 0 => 0
        1 1 => 1
        X - => X
        - 0 => -
        - 1 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ XNOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    return _not_(xor(x0, *xs))


def xor(x0: Bits | str, *xs: Bits | str) -> Bits:
    """Bitwise XOR.

    f(x0, x1) -> y:
       0 0 => 0
        0 1 => 1
        1 0 => 1
        1 1 => 0
        X - => X
        - 0 => -
        - 1 => -

    Args:
        x0: Bits
        x1: Bits of equal size.

    Returns:
        Bits of equal size, w/ XOR result.

    Raises:
        ValueError: Bits sizes do not match.
    """
    x0 = _expect_type(x0, Bits)
    y = x0
    for x in xs:
        x = _expect_size(x, x0.size)
        y = _xor_(y, x)
    return y


def _ite(s: Bits, x1: Bits, x0: Bits) -> Bits:
    s0 = _mask(x1.size) * s.data[0]
    s1 = _mask(x1.size) * s.data[1]
    d0, d1 = lite((s0, s1), x1.data, x0.data)
    t = _resolve_type(type(x0), type(x1))
    return t._cast_data(d0, d1)


def ite(s: Bits | str, x1: Bits | str, x0: Bits | str) -> Bits:
    """If-Then-Else operator.

    Args:
        s: One-bit select
        x1: Bits
        x0: Bits of equal length.

    Returns:
        If-Then-Else result s ? x1 : x0
    """
    s = _expect_size(s, 1)
    x1 = _expect_type(x1, Bits)
    x0 = _expect_size(x0, x1.size)
    return _ite(s, x1, x0)


def _mux(s: Bits, t: type[Bits], xs: dict[int, Bits]) -> Bits:
    m = _mask(t.size)
    si = (s._get_index(i) for i in range(s.size))
    s = tuple((m * d0, m * d1) for d0, d1 in si)
    dc = t.dcs()
    d0, d1 = lmux(s, {i: x.data for i, x in xs.items()}, dc.data)
    return t._cast_data(d0, d1)


_MUX_XN_RE = re.compile(r"x(\d+)")


def mux(s: Bits | str, **xs: Bits | str) -> Bits:
    """Mux operator.

    Args:
        s: Mux select
        **xs: Mux inputs, e.g. x0="4b0001", x1="4b0010", ...

    Mux input names are in the form xN,
    where N is a valid int.
    Muxes require at least one input.
    Any inputs not specified will default to "don't care".

    Returns:
        Mux output, selected from inputs.
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
    """Unary OR reduction.

    Returns:
        Scalar w/ OR reduction.
    """
    x = _expect_type(x, Bits)
    return _uor(x)


def _uand(x: Bits) -> Scalar:
    y = _1
    for i in range(x.size):
        y = land(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uand(x: Bits | str) -> Scalar:
    """Unary AND reduction.

    Returns:
        Scalar w/ AND reduction.
    """
    x = _expect_type(x, Bits)
    return _uand(x)


def _uxnor(x: Bits) -> Scalar:
    y = _1
    for i in range(x.size):
        y = lxnor(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uxnor(x: Bits | str) -> Scalar:
    """Unary XNOR reduction.

    Returns:
        Scalar w/ XNOR reduction.
    """
    x = _expect_type(x, Bits)
    return _uxnor(x)


def _uxor(x: Bits) -> Scalar:
    y = _0
    for i in range(x.size):
        y = lxor(y, x._get_index(i))
    return Scalar(y[0], y[1])


def uxor(x: Bits | str) -> Scalar:
    """Unary XOR reduction.

    Returns:
        Scalar w/ XOR reduction.
    """
    x = _expect_type(x, Bits)
    return _uxor(x)


# Arithmetic
def decode(x: Bits | str) -> Scalar | Vector:
    """Decode dense encoding to sparse, one-hot encoding."""
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
    return vec(d1 ^ _mask(n), d1)


def _add(a: Bits, b: Bits, ci: Scalar) -> tuple[Bits, Scalar]:
    # X/DC propagation
    if a.has_x() or b.has_x() or ci.has_x():
        return a.xes(), _ScalarX
    if a.has_dc() or b.has_dc() or ci.has_dc():
        return a.dcs(), _ScalarW

    dmax = _mask(a.size)
    s = a.data[1] + b.data[1] + ci.data[1]
    co = _bool2scalar[s > dmax]
    s &= dmax

    t = _resolve_type(type(a), type(b))
    return t._cast_data(s ^ dmax, s), co


def add(a: Bits | str, b: Bits | str, ci: Scalar | str | None = None) -> Bits:
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
    ci = _Scalar0 if ci is None else _expect_type(ci, Scalar)
    s, _ = _add(a, b, ci)
    return s


def adc(a: Bits | str, b: Bits | str, ci: Scalar | str | None = None) -> AddResult:
    """Addition with carry-in, and carry-out.

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
    ci = _Scalar0 if ci is None else _expect_type(ci, Scalar)
    s, co = _add(a, b, ci)
    return AddResult(s, co)


def _sub(a: Bits, b: Bits) -> AddResult:
    s, co = _add(a, _not_(b), ci=_Scalar1)
    return AddResult(s, co)


def sub(a: Bits | str, b: Bits | str) -> Bits:
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
    s, _ = _sub(a, b)
    return s


def sbc(a: Bits | str, b: Bits | str) -> AddResult:
    """Twos complement subtraction, with carry-out.

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
    return _sub(a, b)


def _neg(x: Bits) -> AddResult:
    s, co = _add(x.zeros(), _not_(x), ci=_Scalar1)
    return AddResult(s, co)


def neg(x: Bits | str) -> Bits:
    """Twos complement negation.

    Computed using 0 - x.

    Returns:
        2-tuple of (sum, carry-out).
    """
    x = _expect_type(x, Bits)
    s, _ = _neg(x)
    return s


def ngc(x: Bits | str) -> AddResult:
    """Twos complement negation, with carry-out.

    Computed using 0 - x.

    Returns:
        2-tuple of (sum, carry-out).
    """
    x = _expect_type(x, Bits)
    return _neg(x)


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
    d0 = _mask(n) | sh0 << n
    d1 = sh1 << n
    y = x._cast_data(d0, d1)

    return y


def lsh(x: Bits | str, n: Bits | str | int) -> Bits:
    """Left shift by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Bits left-shifted by n bits.

    Raises:
        ValueError: If n is invalid.
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
    d0 = sh0 | (_mask(n) << sh_size)
    d1 = sh1
    y = x._cast_data(d0, d1)

    return y


def rsh(x: Bits | str, n: Bits | str | int) -> Bits:
    """Right shift by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Bits right-shifted by n bits.

    Raises:
        ValueError: If n is invalid.
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
    si0, si1 = _mask(n) * sign0, _mask(n) * sign1

    sh_size, (sh0, sh1) = x._get_slice(n, x.size)
    d0 = sh0 | si0 << sh_size
    d1 = sh1 | si1 << sh_size
    y = x._cast_data(d0, d1)

    return y


def srsh(x: Bits | str, n: Bits | str | int) -> tuple[Bits, Empty | Scalar | Vector]:
    """Signed (arithmetic) right shift by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Bits arithmetically right-shifted by n bits.

    Raises:
        ValueError: If n is invalid.
    """
    x = _expect_type(x, Bits)
    n = _expect_shift(n, x.size)
    return _srsh(x, n)


# Word operations
def _xt(x: Bits, n: int) -> Vector:
    ext0 = _mask(n)
    d0 = x.data[0] | ext0 << x.size
    d1 = x.data[1]
    return _vec_size(x.size + n)(d0, d1)


def xt(x: Bits | str, n: int) -> Bits:
    """Unsigned extend by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Vector zero-extended by n bits.

    Raises:
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
    ext0 = _mask(n) * sign0
    ext1 = _mask(n) * sign1
    d0 = x.data[0] | ext0 << x.size
    d1 = x.data[1] | ext1 << x.size
    return _vec_size(x.size + n)(d0, d1)


def sxt(x: Bits | str, n: int) -> Bits:
    """Sign extend by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Vector sign-extended by n bits.

    Raises:
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
    """Left rotate by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Bits left-rotated by n bits.

    Raises:
        ValueError: If n is invalid/inconsistent.
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
    """Right rotate by n bits.

    Args:
        n: Non-negative number of bits.

    Returns:
        Bits right-rotated by n bits.

    Raises:
        ValueError: If n is invalid/inconsistent.
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
            x = _lit2vec(obj)
            xs.append(x)
        else:
            raise TypeError(f"Invalid input: {obj}")
    return _cat(*xs)


def rep(obj: Bits | int | str, n: int) -> Empty | Scalar | Vector:
    """Repeat a Vector n times."""
    objs = [obj] * n
    return cat(*objs)


# Predicates over bitvectors
def _eq(x0: Bits, x1: Bits) -> Scalar:
    return _uand(_xnor_(x0, x1))


def eq(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Equal operator.

    Args:
        x1: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of self == other
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _eq(x0, x1)


def _ne(x0: Bits, x1: Bits) -> Scalar:
    return _uor(_xor_(x0, x1))


def ne(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Not Equal operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of self != other
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
    """Unsigned less than operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of unsigned(x0) < unsigned(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.lt, x0, x1)


def le(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Unsigned less than or equal operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of unsigned(x0) ≤ unsigned(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.le, x0, x1)


def gt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Unsigned greater than operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of unsigned(x0) > unsigned(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _cmp(operator.gt, x0, x1)


def ge(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Unsigned greater than or equal operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of unsigned(x0) ≥ unsigned(x1)
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
    """Signed less than operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of signed(x0) < signed(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.lt, x0, x1)


def sle(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Signed less than or equal operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of signed(x0) ≤ signed(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.le, x0, x1)


def sgt(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Signed greater than operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of signed(x0) > signed(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.gt, x0, x1)


def sge(x0: Bits | str, x1: Bits | str) -> Scalar:
    """Signed greater than or equal operator.

    Args:
        x0: Bits
        x1: Bits of equal length.

    Returns:
        Scalar result of signed(x0) ≥ signed(x1)
    """
    x0 = _expect_type(x0, Bits)
    x1 = _expect_size(x1, x0.size)
    return _scmp(operator.ge, x0, x1)


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
    return _vec_size(size)(d1 ^ _mask(size), d1)


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
            return _bool2scalar[x]
        case [0 | 1 as fst, *rst]:
            return _bools2vec(fst, *rst)
        case str() as lit:
            return _lit2vec(lit)
        case [str() as lit, *rst]:
            x = _lit2vec(lit)
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
            x = _lit2vec(obj)
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


def i2bv(n: int, size: int | None = None) -> Scalar | Vector:
    """Convert int to Vector.

    Args:
        n: An integer.
        size: Optional output length.

    Returns:
        A Vec instance.

    Raises:
        ValueError: If n overflows the output length.
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

    x = _vec_size(size)(d1 ^ _mask(size), d1)
    if negative:
        return _neg(x).s
    return x


def _chunk(data: tuple[int, int], base: int, size: int) -> tuple[int, int]:
    mask = _mask(size)
    return (data[0] >> base) & mask, (data[1] >> base) & mask


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
