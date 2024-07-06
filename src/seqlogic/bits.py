"""Bit Array data type."""

# Simplify access to friend object attributes
# pylint: disable = protected-access

# PyLint/PyRight are confused by MetaClass behavior
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

from __future__ import annotations

import math
from collections.abc import Generator
from functools import partial

from .util import classproperty
from .vec import Vec, _bools2vec, _lit2vec, _Vec0, _Vec1, _VecE

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
