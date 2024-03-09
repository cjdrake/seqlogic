"""Bit Array Data Type."""

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

    def __init__(self, w: lbool.vec, shape: tuple[int, ...] | None = None):
        """TODO(cjdrake): Write docstring."""
        self._w = w
        if shape is None:
            self._shape = (len(w),)
        else:
            assert math.prod(shape) == len(w)
            self._shape = shape

    def __str__(self) -> str:
        indent = "      "
        return f"bits({self._str(indent)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self._shape[0]

    def __iter__(self) -> Generator[bits, None, None]:
        for i in range(self._shape[0]):
            yield self.__getitem__(i)

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
        return bits(self._w.__add__(other._w))

    def __sub__(self, other: bits) -> bits:
        return bits(self._w.__sub__(other._w))

    def __neg__(self) -> bits:
        return bits(self._w.__neg__())

    @property
    def shape(self) -> tuple[int, ...]:
        """Return bit array shape."""
        return self._shape

    def reshape(self, shape: tuple[int, ...]) -> bits:
        """Return an equivalent bit array with modified shape."""
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return bits(self._w, shape)

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
        for x in self._w:
            yield bits(x)

    def flatten(self) -> bits:
        """Return a vector with equal data, flattened to 1D shape."""
        return bits(self._w)

    def _check_shape(self, other: bits):
        if self._shape != other.shape:
            s = f"Expected shape {self._shape}, got {other.shape}"
            raise ValueError(s)

    def lnot(self) -> bits:
        """Return output of "lifted" NOT function."""
        return bits(self._w.lnot())

    def lnor(self, other: bits) -> bits:
        """Return output of "lifted" NOR function."""
        self._check_shape(other)
        return bits(self._w.lnor(other._w))

    def lor(self, other: bits) -> bits:
        """Return output of "lifted" OR function."""
        self._check_shape(other)
        return bits(self._w.lor(other._w))

    def ulor(self) -> bits:
        """Return unary "lifted" OR of bits."""
        return bits(self._w.ulor())

    def lnand(self, other: bits) -> bits:
        """Return output of "lifted" NAND function."""
        self._check_shape(other)
        return bits(self._w.lnand(other._w))

    def land(self, other: bits) -> bits:
        """Return output of "lifted" AND function."""
        self._check_shape(other)
        return bits(self._w.land(other._w))

    def uland(self) -> bits:
        """Return unary "lifted" AND of bits."""
        return bits(self._w.uland())

    def lxnor(self, other: bits) -> bits:
        """Return output of "lifted" XNOR function."""
        self._check_shape(other)
        return bits(self._w.lxnor(other._w))

    def lxor(self, other: bits) -> bits:
        """Return output of "lifted" XOR function."""
        self._check_shape(other)
        return bits(self._w.lxor(other._w))

    def ulxor(self) -> bits:
        """Return unary "lifted" XOR of bits."""
        return bits(self._w.ulxor())

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
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return bits(v._w.zext(n))

    def sext(self, n: int) -> bits:
        """Return bit array sign extended by n bits.

        Sign extension is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then sign extended.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return bits(v._w.sext(n))

    def lsh(self, n: int | bits, ci: bits | None = None) -> tuple[bits, bits]:
        """Return bit array left shifted by n bits.

        Left shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
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
        return bits(y), bits(co)

    def rsh(self, n: int | bits, ci: bits | None = None) -> tuple[bits, bits]:
        """Return bit array right shifted by n bits.

        Right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
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
        return bits(y), bits(co)

    def arsh(self, n: int | bits) -> tuple[bits, bits]:
        """Return bit array arithmetically right shifted by n bits.

        Arithmetic right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
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
        return bits(y), bits(co)

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
        return bits(s), bits(co), bits(ovf)

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
    return bits(lbool.vec(size, data), shape)


def foo(obj=None) -> bits:
    """Create a bit array."""
    match obj:
        # Empty
        case None:
            return E
        # Rank 0 int
        case 0 | 1 as x:
            return bits(lbool._bools2vec([x]))
        # Rank 1 str
        case str() as lit:
            return bits(lbool._lit2vec(lit))
        # Rank 1 [0 | 1, ...]
        case [0 | 1 as x, *rst]:
            return bits(lbool._bools2vec([x, *rst]))
        # Rank 2 str
        case [str() as lit, *rst]:
            v = bits(lbool._lit2vec(lit))
            return _rank2(v, rst)
        # Rank 2 logic_vector
        case [bits() as v, *rst]:
            return _rank2(v, rst)
        # Rank 3+
        case [*objs]:
            return cat([foo(obj) for obj in objs])
        # Unimplemented
        case _:
            raise TypeError(f"Invalid input: {type(obj)}")


def uint2bits(num: int, n: int | None = None) -> bits:
    """Convert nonnegative int to logic_vector."""
    return bits(lbool.uint2vec(num, n))


def int2bits(num: int, n: int | None = None) -> bits:
    """Convert int to logic_vector."""
    return bits(lbool.int2vec(num, n))


def cat(objs: Collection[int | bits], flatten: bool = False) -> bits:
    """Join a sequence of bits."""
    # Empty
    if len(objs) == 0:
        return E

    # Convert inputs
    vs: list[bits] = []
    for obj in objs:
        match obj:
            case 0 | 1 as x:
                vs.append(bits(lbool._bools2vec([x])))
            case bits() as v:
                vs.append(v)
            case _:
                raise TypeError(f"Invalid input: {type(obj)}")

    if len(vs) == 1:
        return vs[0]

    fst, rst = vs[0], vs[1:]
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

    return bits(lbool.vec(size, data), shape)


def rep(obj: int | bits, n: int, flatten: bool = False) -> bits:
    """Repeat a bit array n times."""
    return cat([obj] * n, flatten)


def _sel(v: bits, key: tuple[int | slice, ...]) -> bits:
    assert 0 <= v.ndim == len(key)

    shape = v.shape[1:]
    n = math.prod(shape)
    nbits = 2 * n
    mask = (1 << nbits) - 1

    def f(data: int, i: int) -> int:
        return (data >> (nbits * i)) & mask

    match key[0]:
        case int() as i:
            data = f(v._w.data, i)
            if shape:
                return _sel(bits(lbool.vec(n, data), shape), key[1:])
            return bits(lbool.vec(1, data))
        case slice() as sl:
            datas = (f(v._w.data, i) for i in range(sl.start, sl.stop, sl.step))
            if shape:
                return cat([_sel(bits(lbool.vec(n, data), shape), key[1:]) for data in datas])
            return cat([bits(lbool.vec(1, data)) for data in datas])
        case _:  # pragma: no cover
            assert False


# The empty vector is a singleton
E = bits(lbool.vec(0, 0))


def illogicals(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with ILLOGICAL."""
    return bits(lbool.illogicals(math.prod(shape)), shape)


def zeros(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with zeros."""
    return bits(lbool.zeros(math.prod(shape)), shape)


def ones(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with ones."""
    return bits(lbool.ones(math.prod(shape)), shape)


def xes(shape: tuple[int, ...]) -> bits:
    """Return a new logic_vector of given shape, filled with Xes."""
    return bits(lbool.xes(math.prod(shape)), shape)


# One bit values
W = illogicals((1,))
F = zeros((1,))
T = ones((1,))
X = xes((1,))
