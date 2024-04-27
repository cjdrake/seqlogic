"""Bit Array data type."""

# pylint: disable = protected-access

from __future__ import annotations

import math
from collections.abc import Collection, Generator
from functools import cached_property

from . import lbool
from .lbool import Vec, bools2vec, int2vec, lit2vec, uint2vec


class Bits:
    """Bit array data type.

    Do NOT instantiate this type directly.
    That requires you to understand the data encoding.
    Use the factory functions instead:

    * bits
    * uint2bits
    * int2bits
    * illogicals
    * zeros
    * ones
    * xes
    """

    def __class_getitem__(cls, key: int | tuple[int, ...]):
        pass  # pragma: no cover

    def __init__(self, shape: tuple[int, ...], data: int):
        """Initialize.

        Do NOT instantiate this type directly.

        Args:
            shape: a tuple of int dimension sizes.
            data: lbool items packed into an int.
        """
        self._shape = shape
        self._data = data

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, key: int | Bits | slice | tuple[int | Bits | slice, ...]) -> Bits:
        if self._shape == (0,):
            raise IndexError("Cannot index an empty vector")
        match key:
            case int() | Bits() | slice():
                return _sel(self, self._norm_key([key]))
            case tuple():
                return _sel(self, self._norm_key(list(key)))
            case _:
                s = "Expected key to be int, Bits, slice, or tuple"
                raise TypeError(s)

    def __iter__(self) -> Generator[Bits, None, None]:
        for i in range(self._shape[0]):
            yield self.__getitem__(i)

    def __str__(self) -> str:
        name = "bits"
        indent = " " * len(name) + "  "
        return f"{name}({self._str(indent)})"

    def __repr__(self) -> str:
        d = f"0b{self._data:0{self._v.nbits}b}"
        return f"bits({repr(self._shape)}, {d})"

    def __bool__(self) -> bool:
        return bool(self._v)

    def __int__(self) -> int:
        return int(self._v)

    # Comparison
    def _eq(self, other: Bits) -> bool:
        return self._shape == other.shape and self._data == other.data

    def __eq__(self, other) -> bool:
        match other:
            case Bits():
                return self._eq(other)
            case _:
                return False

    def __hash__(self) -> int:
        return hash(self._shape) ^ hash(self._data)

    # Bitwise Arithmetic
    def __invert__(self) -> Bits:
        return self.lnot()

    def __or__(self, other: Bits) -> Bits:
        return self.lor(other)

    def __and__(self, other: Bits) -> Bits:
        return self.land(other)

    def __xor__(self, other: Bits) -> Bits:
        return self.lxor(other)

    def __lshift__(self, n: int | Bits) -> Bits:
        return self.lsh(n)[0]

    def __rshift__(self, n: int | Bits) -> Bits:
        return self.rsh(n)[0]

    def __add__(self, other: Bits) -> Bits:
        return self.add(other, ci=F)[0]

    def __sub__(self, other: Bits) -> Bits:
        return self.sub(other)[0]

    def __neg__(self) -> Bits:
        return self.neg()[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return bit array shape."""
        return self._shape

    @property
    def data(self) -> int:
        """Return bit array data."""
        return self._data

    @cached_property
    def _v(self) -> Vec:
        return Vec[self.size](self._data)

    def reshape(self, shape: tuple[int, ...]) -> Bits:
        """Return an equivalent bit array with modified shape."""
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return Bits(shape, self._data)

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return math.prod(self._shape)

    @property
    def flat(self) -> Generator[Bits, None, None]:
        """Return a flat iterator to the items."""
        for v in self._v:
            yield _v2b(v)

    def flatten(self) -> Bits:
        """Return a vector with equal data, flattened to 1D shape."""
        if self.ndim == 1:
            return self
        return Bits((self.size,), self._data)

    def lnot(self) -> Bits:
        """Bitwise lifted NOT.

        Returns:
            bit array of equal size and inverted data.
        """
        return Bits(self._shape, self._v.not_().data)

    def lnor(self, other: Bits) -> Bits:
        """Bitwise lifted NOR.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains NOR result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.nor(other._v))

    def lor(self, other: Bits) -> Bits:
        """Bitwise lifted OR.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains OR result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.or_(other._v))

    def ulor(self) -> Bits[1]:
        """Unary lifted OR reduction.

        Returns:
            One-bit array, data contains OR reduction.
        """
        return _v2b(self._v.uor())

    def lnand(self, other: Bits) -> Bits:
        """Bitwise lifted NAND.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains NAND result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.nand(other._v))

    def land(self, other: Bits) -> Bits:
        """Bitwise lifted AND.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains AND result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.and_(other._v))

    def uland(self) -> Bits[1]:
        """Unary lifted AND reduction.

        Returns:
            One-bit array, data contains AND reduction.
        """
        return _v2b(self._v.uand())

    def lxnor(self, other: Bits) -> Bits:
        """Bitwise lifted XNOR.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains XNOR result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.xnor(other._v))

    def lxor(self, other: Bits) -> Bits:
        """Bitwise lifted XOR.

        Args:
            other: bit array of equal size.

        Returns:
            bit array of equal size, data contains XOR result.

        Raises:
            ValueError: bit array shapes do not match.
        """
        self._check_shape(other)
        return _v2b(self._v.xor(other._v))

    def ulxnor(self) -> Bits[1]:
        """Unary lifted XNOR reduction.

        Returns:
            One-bit array, data contains XOR reduction.
        """
        return _v2b(self._v.uxnor())

    def ulxor(self) -> Bits[1]:
        """Unary lifted XOR reduction.

        Returns:
            One-bit array, data contains XOR reduction.
        """
        return _v2b(self._v.uxor())

    def to_uint(self) -> int:
        """Convert to unsigned integer.

        Returns:
            An unsigned int.

        Raises:
            ValueError: bit array is partially unknown.
        """
        return self._v.to_uint()

    def to_int(self) -> int:
        """Convert to signed integer.

        Returns:
            A signed int, from two's complement encoding.

        Raises:
            ValueError: bit array is partially unknown.
        """
        return self._v.to_int()

    def ult(self, other: Bits) -> bool:
        """Unsigned less than.

        Args:
            other: bit array of equal size.

        Returns:
            Boolean result of unsigned(self) < unsigned(other)

        Raises:
            ValueError: bit array sizes do not match.
        """
        return self._v.ult(other._v)

    def slt(self, other: Bits) -> bool:
        """Signed less than.

        Args:
            other: bit array of equal size.

        Returns:
            Boolean result of signed(self) < signed(other)

        Raises:
            ValueError: bit array sizes do not match.
        """
        return self._v.slt(other._v)

    def zext(self, n: int) -> Bits:
        """Return bit array zero extended by n bits.

        Zero extension is defined for 1-D vectors.
        Vectors of higher dimensions will be flattened, then zero extended.
        """
        return _v2b(self.flatten()._v.zext(n))

    def sext(self, n: int) -> Bits:
        """Return bit array sign extended by n bits.

        Sign extension is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then sign extended.
        """
        return _v2b(self.flatten()._v.sext(n))

    def lsh(self, n: int | Bits, ci: Bits[1] | None = None) -> tuple[Bits, Bits]:
        """Return bit array left shifted by n bits.

        Left shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case Bits():
                if n._v.has_x():
                    return illogicals((v.size,)), E
                elif n._v.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or Bits")
        if ci is None:
            y, co = self._v.lsh(n)
        else:
            y, co = self._v.lsh(n, ci._v)
        return _v2b(y), _v2b(co)

    def rsh(self, n: int | Bits, ci: Bits[1] | None = None) -> tuple[Bits, Bits]:
        """Return bit array right shifted by n bits.

        Right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case Bits():
                if n._v.has_x():
                    return illogicals((v.size,)), E
                elif n._v.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or Bits")
        if ci is None:
            y, co = self._v.rsh(n)
        else:
            y, co = self._v.rsh(n, ci._v)
        return _v2b(y), _v2b(co)

    def arsh(self, n: int | Bits) -> tuple[Bits, Bits]:
        """Return bit array arithmetically right shifted by n bits.

        Arithmetic right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self.flatten()
        match n:
            case int():
                pass
            case Bits():
                if n._v.has_x():
                    return illogicals((v.size,)), E
                elif n._v.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or Bits")
        y, co = self._v.arsh(n)
        return _v2b(y), _v2b(co)

    def add(self, other: Bits, ci: Bits[1]) -> tuple[Bits, Bits[1], Bits[1]]:
        """Twos complement additions.

        Args:
            other: bit array of equal size.

        Returns:
            3-tuple of (sum, carry-out, overflow).

        Raises:
            ValueError: bit array lengths are invalid/inconsistent.
        """
        s, co, ovf = self._v.add(other._v, ci._v)
        return _v2b(s), _v2b(co), _v2b(ovf)

    def sub(self, other: Bits) -> tuple[Bits, Bits[1], Bits[1]]:
        """Twos complement subtraction.

        Args:
            other: bit array of equal size.

        Raises:
            ValueError: bit array lengths are invalid/inconsistent.
        """
        s, co, ovf = self._v.sub(other._v)
        return _v2b(s), _v2b(co), _v2b(ovf)

    def neg(self) -> tuple[Bits, Bits[1], Bits[1]]:
        """Twos complement negation.

        Computed using 0 - self.

        Returns:
            3-tuple of (sum, carry-out, overflow).
        """
        s, co, ovf = self._v.neg()
        return _v2b(s), _v2b(co), _v2b(ovf)

    def _check_shape(self, other: Bits):
        if self._shape != other.shape:
            s = f"Expected shape {self._shape}, got {other.shape}"
            raise ValueError(s)

    def _to_lit(self) -> str:
        return str(self._v)

    def _str(self, indent: str) -> str:
        """Help __str__ method recursion."""
        # Empty
        if self._shape == (0,):
            return "[]"
        # Scalar
        if self._shape == (1,):
            return "[" + str(self._v)[2:] + "]"
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

    def _norm_key(self, key: list[int | Bits | slice]) -> tuple[int | slice, ...]:
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
                case Bits() as v:
                    nkey.append(self._norm_index(v.to_uint(), i))
                case slice() as sl:
                    nkey.append(self._norm_slice(sl, i))
                case _:  # pragma: no cover
                    assert False

        return tuple(nkey)


def _v2b(v: Vec) -> Bits:
    return Bits((len(v),), v.data)


def _rank2(fst: Bits, rst) -> Bits:
    shape = (len(rst) + 1,) + fst.shape
    size = len(fst._v)
    data = fst.data
    for i, b in enumerate(rst, start=1):
        match b:
            case str() as lit:
                v = lit2vec(lit)
                if len(v) != fst.size:
                    s = f"Expected str literal to have size {fst.size}, got {len(v)}"
                    raise TypeError(s)
                data |= v.data << (fst._v.nbits * i)
            case Bits() if b.shape == fst.shape:
                data |= b.data << (fst._v.nbits * i)
            case _:
                s = ",".join(str(dim) for dim in fst.shape)
                s = f"Expected item to be str or Bits[{s}]"
                raise TypeError(s)
        size += len(fst._v)
    return Bits(shape, data)


def bits(obj=None) -> Bits:
    """Create a bit array."""
    match obj:
        # Empty
        case None:
            return E
        # Rank 0 int
        case 0 | 1 as x:
            v = bools2vec([x])
            return Bits((1,), v.data)
        # Rank 1 str
        case str() as lit:
            return _v2b(lit2vec(lit))
        # Rank 1 [0 | 1, ...]
        case [0 | 1 as x, *rst]:
            return _v2b(bools2vec([x, *rst]))
        # Rank 2 str
        case [str() as lit, *rst]:
            return _rank2(_v2b(lit2vec(lit)), rst)
        # Rank 2 logic_vector
        case [Bits() as b, *rst]:
            return _rank2(b, rst)
        # Rank 3+
        case [*objs]:
            return cat([bits(obj) for obj in objs])
        # Unimplemented
        case _:
            raise TypeError(f"Invalid input: {type(obj)}")


def uint2bits(num: int, n: int | None = None) -> Bits:
    """Convert nonnegative int to bit array.

    Args:
        num: A nonnegative integer.
        n: Optional output length.

    Returns:
        A bit array instance.

    Raises:
        ValueError: If num is negative or overflows the output length.
    """
    return _v2b(uint2vec(num, n))


def int2bits(num: int, n: int | None = None) -> Bits:
    """Convert int to bit array.

    Args:
        num: An integer.
        n: Optional output length.

    Returns:
        A bit array instance.

    Raises:
        ValueError: If num overflows the output length.
    """
    return _v2b(int2vec(num, n))


def cat(objs: Collection[int | Bits], flatten: bool = False) -> Bits:
    """Join a sequence of bits."""
    # Empty
    if len(objs) == 0:
        return E

    # Convert inputs
    bs: list[Bits] = []
    for obj in objs:
        match obj:
            case 0 | 1 as x:
                v = bools2vec([x])
                bs.append(Bits((1,), v.data))
            case Bits() as b:
                bs.append(b)
            case _:
                raise TypeError(f"Invalid input: {type(obj)}")

    if len(bs) == 1:
        return bs[0]

    fst, rst = bs[0], bs[1:]
    scalar = fst.shape == (1,)
    regular = True
    dims = [fst.shape[0]]
    size = len(fst._v)
    data = fst.data

    pos = fst._v.nbits
    for b in rst:
        if b.shape[0] != fst.shape[0]:
            regular = False
        if b.shape[1:] != fst.shape[1:]:
            s = f"Expected shape {fst.shape[1:]}, got {b.shape[1:]}"
            raise ValueError(s)
        dims.append(b.shape[0])
        size += len(b._v)
        data |= b.data << pos
        pos += b._v.nbits

    if not scalar and regular and not flatten:
        shape = (len(dims),) + fst.shape
    else:
        shape = (sum(dims),) + fst.shape[1:]

    return Bits(shape, data)


def rep(obj: int | Bits, n: int, flatten: bool = False) -> Bits:
    """Repeat a bit array n times."""
    return cat([obj] * n, flatten)


def _sel(b: Bits, key: tuple[int | slice, ...]) -> Bits:
    assert 0 <= b.ndim == len(key)

    shape = b.shape[1:]
    n = math.prod(shape)
    nbits = 2 * n
    mask = (1 << nbits) - 1

    def f(data: int, i: int) -> int:
        return (data >> (nbits * i)) & mask

    match key[0]:
        case int() as i:
            data = f(b.data, i)
            if shape:
                return _sel(Bits(shape, data), key[1:])
            return Bits((1,), data)
        case slice() as sl:
            datas = (f(b.data, i) for i in range(sl.start, sl.stop, sl.step))
            if shape:
                return cat([_sel(Bits(shape, data), key[1:]) for data in datas])
            return cat([Bits((1,), data) for data in datas])
        case _:  # pragma: no cover
            assert False


# The empty vector is a singleton
E = Bits((0,), 0)


def illogicals(shape: tuple[int, ...]) -> Bits:
    """Return a new logic_vector of given shape, filled with ILLOGICAL."""
    n = math.prod(shape)
    v = lbool.xes(n)
    return Bits(shape, v.data)


def zeros(shape: tuple[int, ...]) -> Bits:
    """Return a new logic_vector of given shape, filled with zeros."""
    n = math.prod(shape)
    v = lbool.zeros(n)
    return Bits(shape, v.data)


def ones(shape: tuple[int, ...]) -> Bits:
    """Return a new logic_vector of given shape, filled with ones."""
    n = math.prod(shape)
    v = lbool.ones(n)
    return Bits(shape, v.data)


def xes(shape: tuple[int, ...]) -> Bits:
    """Return a new logic_vector of given shape, filled with Xes."""
    n = math.prod(shape)
    v = lbool.dcs(n)
    return Bits(shape, v.data)


# One bit values
W = illogicals((1,))
F = zeros((1,))
T = ones((1,))
X = xes((1,))
