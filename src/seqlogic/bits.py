"""Bit Array data type."""

# pylint: disable = protected-access

from __future__ import annotations

import math
from collections.abc import Collection, Generator
from functools import cached_property

from . import lbool


class bits:
    """Bit array data type.

    Do NOT instantiate this type directly.
    Use the factory functions instead.
    """

    def __class_getitem__(cls, key: int | tuple[int, ...]):
        pass  # pragma: no cover

    def __init__(self, shape: tuple[int, ...], w: lbool.vec):
        """TODO(cjdrake): Write docstring."""
        assert math.prod(shape) == len(w)
        self._shape = shape
        self._w = w

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, key: int | bits | slice | tuple[int | bits | slice, ...]) -> bits:
        if self._shape == (0,):
            raise IndexError("Cannot index an empty vector")
        match key:
            case int() | bits() | slice():
                return _sel(self, self._norm_key([key]))
            case tuple():
                return _sel(self, self._norm_key(list(key)))
            case _:
                s = "Expected key to be int, bits, slice, or tuple"
                raise TypeError(s)

    def __iter__(self) -> Generator[bits, None, None]:
        for i in range(self._shape[0]):
            yield self.__getitem__(i)

    def __str__(self) -> str:
        name = "bits"
        indent = " " * len(name) + "  "
        return f"{name}({self._str(indent)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        match other:
            case bits():
                return self._w.data == other._w.data and self._shape == other.shape
            case _:
                return False

    def __invert__(self) -> bits:
        return self.lnot()

    def __or__(self, other: bits) -> bits:
        return self.lor(other)

    def __and__(self, other: bits) -> bits:
        return self.land(other)

    def __xor__(self, other: bits) -> bits:
        return self.lxor(other)

    def __lshift__(self, n: int | bits) -> bits:
        return self.lsh(n)[0]

    def __rshift__(self, n: int | bits) -> bits:
        return self.rsh(n)[0]

    def __add__(self, other: bits) -> bits:
        v = self._w.__add__(other._w)
        return bits((len(v),), v)

    def __sub__(self, other: bits) -> bits:
        v = self._w.__sub__(other._w)
        return bits((len(v),), v)

    def __neg__(self) -> bits:
        v = self._w.__neg__()
        return bits((len(v),), v)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return bit array shape."""
        return self._shape

    def reshape(self, shape: tuple[int, ...]) -> bits:
        """Return an equivalent bit array with modified shape."""
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return bits(shape, self._w)

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return len(self._w)

    @property
    def flat(self) -> Generator[bits, None, None]:
        """Return a flat iterator to the items."""
        for v in self._w:
            yield bits((len(v),), v)

    def flatten(self) -> bits:
        """Return a vector with equal data, flattened to 1D shape."""
        if self.ndim == 1:
            return self
        return bits((len(self._w),), self._w)

    def _check_shape(self, other: bits):
        if self._shape != other.shape:
            s = f"Expected shape {self._shape}, got {other.shape}"
            raise ValueError(s)

    def lnot(self) -> bits:
        """Return output of "lifted" NOT function."""
        v = self._w.lnot()
        return bits((len(v),), v)

    def lnor(self, other: bits) -> bits:
        """Return output of "lifted" NOR function."""
        self._check_shape(other)
        v = self._w.lnor(other._w)
        return bits((len(v),), v)

    def lor(self, other: bits) -> bits:
        """Return output of "lifted" OR function."""
        self._check_shape(other)
        v = self._w.lor(other._w)
        return bits((len(v),), v)

    def ulor(self) -> bits:
        """Return unary "lifted" OR of bits."""
        v = self._w.ulor()
        return bits((len(v),), v)

    def lnand(self, other: bits) -> bits:
        """Return output of "lifted" NAND function."""
        self._check_shape(other)
        v = self._w.lnand(other._w)
        return bits((len(v),), v)

    def land(self, other: bits) -> bits:
        """Return output of "lifted" AND function."""
        self._check_shape(other)
        v = self._w.land(other._w)
        return bits((len(v),), v)

    def uland(self) -> bits:
        """Return unary "lifted" AND of bits."""
        v = self._w.uland()
        return bits((len(v),), v)

    def lxnor(self, other: bits) -> bits:
        """Return output of "lifted" XNOR function."""
        self._check_shape(other)
        v = self._w.lxnor(other._w)
        return bits((len(v),), v)

    def lxor(self, other: bits) -> bits:
        """Return output of "lifted" XOR function."""
        self._check_shape(other)
        v = self._w.lxor(other._w)
        return bits((len(v),), v)

    def ulxor(self) -> bits:
        """Return unary "lifted" XOR of bits."""
        v = self._w.ulxor()
        return bits((len(v),), v)

    def to_uint(self) -> int:
        """Convert vector to unsigned integer."""
        return self._w.to_uint()

    def to_int(self) -> int:
        """Convert vector to signed integer."""
        return self._w.to_int()

    def zext(self, n: int) -> bits:
        """Return bit array zero extended by n bits.

        Zero extension is defined for 1-D vectors.
        Vectors of higher dimensions will be flattened, then zero extended.
        """
        v = self.flatten()._w.zext(n)
        return bits((len(v),), v)

    def sext(self, n: int) -> bits:
        """Return bit array sign extended by n bits.

        Sign extension is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then sign extended.
        """
        v = self.flatten()._w.sext(n)
        return bits((len(v),), v)

    def lsh(self, n: int | bits, ci: bits | None = None) -> tuple[bits, bits]:
        """Return bit array left shifted by n bits.

        Left shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case bits():
                if n._w.has_illogical():
                    return illogicals((v.size,)), E
                elif n._w.has_unknown():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or bits")
        if ci is None:
            y, co = self._w.lsh(n)
        else:
            y, co = self._w.lsh(n, ci._w)
        return bits((len(y),), y), bits((len(co),), co)

    def rsh(self, n: int | bits, ci: bits | None = None) -> tuple[bits, bits]:
        """Return bit array right shifted by n bits.

        Right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case bits():
                if n._w.has_illogical():
                    return illogicals((v.size,)), E
                elif n._w.has_unknown():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or bits")
        if ci is None:
            y, co = self._w.rsh(n)
        else:
            y, co = self._w.rsh(n, ci._w)
        return bits((len(y),), y), bits((len(co),), co)

    def arsh(self, n: int | bits) -> tuple[bits, bits]:
        """Return bit array arithmetically right shifted by n bits.

        Arithmetic right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case bits():
                if n._w.has_illogical():
                    return illogicals((v.size,)), E
                elif n._w.has_unknown():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or bits")
        y, co = self._w.arsh(n)
        return bits((len(y),), y), bits((len(co),), co)

    def add(self, other: bits, ci: object) -> tuple[bits, bits, bits]:
        """Return the sum of two bit arrays, carry out, and overflow.

        The implementation propagates Xes according to the
        ripple carry addition algorithm.
        """
        match ci:
            case bits():
                pass
            case _:
                ci = (F, T)[bool(ci)]
        s, co, ovf = self._w.add(other._w, ci._w)
        return bits((len(s),), s), bits((len(co),), co), bits((len(ovf),), ovf)

    def _to_lit(self) -> str:
        return str(self._w)

    def _str(self, indent: str) -> str:
        """Help __str__ method recursion."""
        # Empty
        if self._shape == (0,):
            return "[]"
        # Scalar
        if self._shape == (1,):
            return "[" + str(self._w)[2:] + "]"
        # 1D Vector
        if self.ndim == 1:
            return self._to_lit()

        # Tensor, ie N-dimensional vector
        if self.ndim == 2:
            sep = ", "
        elif self.ndim == 3:
            sep = ",\n" + indent
        else:
            sep = ",\n\n" + indent
        return "[" + sep.join(v._str(indent + " ") for v in self) + "]"

    def _norm_index(self, index: int, i: int) -> int:
        lo, hi = -self._shape[i], self._shape[i]
        if not lo <= index < hi:
            s = f"Expected index in [{lo}, {hi}), got {index}"
            raise IndexError(s)
        # Normalize negative start index
        if index < 0:
            index += hi
        return index

    def _norm_slice(self, sl: slice, i: int) -> slice:
        lo, hi = -self._shape[i], self._shape[i]
        # Normalize start
        if sl.start is None:
            sl = slice(0, sl.stop, sl.step)
        elif sl.start < lo:
            sl = slice(lo, sl.stop, sl.step)
        if sl.start < 0:
            sl = slice(sl.start + hi, sl.stop, sl.step)
        # Normalize stop
        if sl.stop is None or sl.stop > hi:
            sl = slice(sl.start, hi, sl.step)
        elif sl.stop < 0:
            sl = slice(sl.start, sl.stop + hi, sl.step)
        # Normalize step
        if sl.step is None:
            sl = slice(sl.start, sl.stop, 1)
        return sl

    def _norm_key(self, key: list[int | bits | slice]) -> tuple[int | slice, ...]:
        ndim = len(key)
        if ndim > self.ndim:
            s = f"Expected ≤ {self.ndim} slice dimensions, got {ndim}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(self.ndim - ndim):
            key.append(slice(None))

        # Normalize key dimensions
        nkey = []
        for i, dim in enumerate(key):
            match dim:
                case int() as index:
                    nkey.append(self._norm_index(index, i))
                case bits() as v:
                    nkey.append(self._norm_index(v.to_uint(), i))
                case slice() as sl:
                    nkey.append(self._norm_slice(sl, i))
                case _:  # pragma: no cover
                    assert False

        return tuple(nkey)


def _rank2(fst: bits, rst) -> bits:
    shape = (len(rst) + 1,) + fst.shape
    size = len(fst._w)
    data = fst._w.data
    for i, v in enumerate(rst, start=1):
        match v:
            case str() as lit:
                w = lbool._lit2vec(lit)
                if len(w) != fst.size:
                    s = f"Expected str literal to have size {fst.size}, got {len(w)}"
                    raise TypeError(s)
                data |= w.data << (fst._w.nbits * i)
            case bits() if v.shape == fst.shape:
                data |= v._w.data << (fst._w.nbits * i)
            case _:
                s = ",".join(str(dim) for dim in fst.shape)
                s = f"Expected item to be str or bits[{s}]"
                raise TypeError(s)
        size += len(fst._w)
    return bits(shape, lbool.vec(size, data))


def foo(obj=None) -> bits:
    """Create a bit array."""
    match obj:
        # Empty
        case None:
            return E
        # Rank 0 int
        case 0 | 1 as x:
            return bits((1,), lbool._bools2vec([x]))
        # Rank 1 str
        case str() as lit:
            v = lbool._lit2vec(lit)
            return bits((len(v),), v)
        # Rank 1 [0 | 1, ...]
        case [0 | 1 as x, *rst]:
            v = lbool._bools2vec([x, *rst])
            return bits((len(v),), v)
        # Rank 2 str
        case [str() as lit, *rst]:
            v = lbool._lit2vec(lit)
            b = bits((len(v),), v)
            return _rank2(b, rst)
        # Rank 2 logic_vector
        case [bits() as b, *rst]:
            return _rank2(b, rst)
        # Rank 3+
        case [*objs]:
            return cat([foo(obj) for obj in objs])
        # Unimplemented
        case _:
            raise TypeError(f"Invalid input: {type(obj)}")


def uint2bits(num: int, n: int | None = None) -> bits:
    """Convert nonnegative int to logic_vector."""
    v = lbool.uint2vec(num, n)
    return bits((len(v),), v)


def int2bits(num: int, n: int | None = None) -> bits:
    """Convert int to logic_vector."""
    v = lbool.int2vec(num, n)
    return bits((len(v),), v)


def cat(objs: Collection[int | bits], flatten: bool = False) -> bits:
    """Join a sequence of bits."""
    # Empty
    if len(objs) == 0:
        return E

    # Convert inputs
    bs: list[bits] = []
    for obj in objs:
        match obj:
            case 0 | 1 as x:
                bs.append(bits((1,), lbool._bools2vec([x])))
            case bits() as b:
                bs.append(b)
            case _:
                raise TypeError(f"Invalid input: {type(obj)}")

    if len(bs) == 1:
        return bs[0]

    fst, rst = bs[0], bs[1:]
    scalar = fst.shape == (1,)
    regular = True
    dims = [fst.shape[0]]
    size = len(fst._w)
    data = fst._w.data

    pos = fst._w.nbits
    for v in rst:
        if v.shape[0] != fst.shape[0]:
            regular = False
        if v.shape[1:] != fst.shape[1:]:
            s = f"Expected shape {fst.shape[1:]}, got {v.shape[1:]}"
            raise ValueError(s)
        dims.append(v.shape[0])
        size += len(v._w)
        data |= v._w.data << pos
        pos += v._w.nbits

    if not scalar and regular and not flatten:
        shape = (len(dims),) + fst.shape
    else:
        shape = (sum(dims),) + fst.shape[1:]

    return bits(shape, lbool.vec(size, data))


def rep(obj: int | bits, n: int, flatten: bool = False) -> bits:
    """Repeat a bit array n times."""
    return cat([obj] * n, flatten)


def _sel(b: bits, key: tuple[int | slice, ...]) -> bits:
    assert 0 <= b.ndim == len(key)

    shape = b.shape[1:]
    n = math.prod(shape)
    nbits = 2 * n
    mask = (1 << nbits) - 1

    def f(data: int, i: int) -> int:
        return (data >> (nbits * i)) & mask

    match key[0]:
        case int() as i:
            data = f(b._w.data, i)
            if shape:
                return _sel(bits(shape, lbool.vec(n, data)), key[1:])
            return bits((1,), lbool.vec(1, data))
        case slice() as sl:
            datas = (f(b._w.data, i) for i in range(sl.start, sl.stop, sl.step))
            if shape:
                return cat([_sel(bits(shape, lbool.vec(n, data)), key[1:]) for data in datas])
            return cat([bits((1,), lbool.vec(1, data)) for data in datas])
        case _:  # pragma: no cover
            assert False


# The empty vector is a singleton
E = bits((0,), lbool.vec(0, 0))


def illogicals(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with ILLOGICAL."""
    return bits(shape, lbool.illogicals(math.prod(shape)))


def zeros(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with zeros."""
    return bits(shape, lbool.zeros(math.prod(shape)))


def ones(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with ones."""
    return bits(shape, lbool.ones(math.prod(shape)))


def xes(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with Xes."""
    return bits(shape, lbool.xes(math.prod(shape)))


# One bit values
W = illogicals((1,))
F = zeros((1,))
T = ones((1,))
X = xes((1,))
