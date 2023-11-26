"""
Logic Vector Data Type
"""

# pylint: disable = protected-access

import math
import re
from collections.abc import Collection, Generator
from functools import cached_property
from typing import Self, TypeAlias, Union

from . import pcn
from .logic import logic
from .pcn import PcItem, PcList

_Logic: TypeAlias = Union[logic, "logicvec"]

# __getitem__ input key type
_Key: TypeAlias = Union[int, "logicvec", slice, tuple[Union[int, "logicvec", slice], ...]]


_NUM_RE = re.compile(
    r"((?P<BinSize>[0-9]+)b(?P<BinDigits>[X01x_]+))|"
    r"((?P<HexSize>[0-9]+)h(?P<HexDigits>[0-9a-fA-F_]+))"
)


class logicvec:
    """
    Logic vector data type

    Do NOT instantiate this type directly.
    Use the factory functions instead.
    """

    def __init__(self, shape: tuple[int, ...], data: int):
        self._shape = shape
        assert 0 <= data < (1 << self.nbits)
        self._data = data

    def __str__(self) -> str:
        indent = "     "
        return f"vec({self._str(indent)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self._shape[0]

    def __iter__(self) -> Generator[_Logic, None, None]:
        for i in range(self._shape[0]):
            yield self.__getitem__(i)

    def __getitem__(self, key: _Key) -> _Logic:
        if self._shape == (0,):
            raise IndexError("Cannot index an empty vector")
        return _sel(self, self._norm_key(key))

    def __eq__(self, other) -> bool:
        match other:
            case logicvec():
                return self._shape == other.shape and self._data == other.data
            case _:
                return False

    def __invert__(self) -> Self:
        return self.not_()

    def __or__(self, other: Self) -> Self:
        return self.or_(other)

    def __and__(self, other: Self) -> Self:
        return self.and_(other)

    def __xor__(self, other: Self) -> Self:
        return self.xor(other)

    def __lshift__(self, n: int) -> Self:
        return self.lsh(n)[0]

    def __rshift__(self, n: int) -> Self:
        return self.rsh(n)[0]

    def __add__(self, other: Self) -> Self:
        s, _, _ = self.add(other, ci=logic.F)
        return s

    def __sub__(self, other: Self) -> Self:
        s, _, _ = self.add(~other, ci=logic.T)
        return s

    def __neg__(self) -> Self:
        s = []
        c = [logic.T]
        for i, x in enumerate((~self).flat):
            s.append(x ^ c[i])
            c.append(x & c[i])
        data = PcList(0)
        for i, x in enumerate(s):
            data = pcn.set_item(data, i, x.value)
        return logicvec((self.size,), data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def data(self) -> int:
        return self._data

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Return an equivalent logic_vector with modified shape."""
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return logicvec(shape, self._data)

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self._shape)

    @cached_property
    def size(self) -> int:
        """Number of elements in the vector"""
        return math.prod(self._shape)

    @cached_property
    def nbits(self) -> int:
        """Number of bits of data"""
        return self.size << 1

    @property
    def flat(self) -> Generator[logic, None, None]:
        """Return a flat iterator to the logic items."""
        for i in range(self.size):
            yield logic(pcn.get_item(self._data, i))

    def flatten(self) -> Self:
        """Return a vector with equal data, flattened to 1D shape."""
        return logicvec((self.size,), self._data)

    def not_(self) -> Self:
        """Return output of NOT function."""
        x_0 = self._bits(0)
        x_01 = x_0 << 1
        x_1 = self._bits(1)
        x_10 = x_1 >> 1

        y0 = x_10
        y1 = x_01

        return logicvec(self._shape, y0 | y1)

    def _check_shape(self, other: Self):
        if self._shape != other.shape:
            s = f"Expected shape {self._shape}, got {other.shape}"
            raise ValueError(s)

    def nor(self, other: Self) -> Self:
        """Return output of NOR function.

        y1 = x0[0] & x1[0]
        y0 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_01 = x0_0 << 1
        x0_1 = self._bits(1)
        x0_10 = x0_1 >> 1

        x1_0 = other._bits(0)
        x1_01 = x1_0 << 1
        x1_1 = other._bits(1)
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_01

        return logicvec(self._shape, y0 | y1)

    def or_(self, other: Self) -> Self:
        """Return output of OR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[0]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_01 = x0_0 << 1
        x0_1 = self._bits(1)

        x1_0 = other._bits(0)
        x1_01 = x1_0 << 1
        x1_1 = other._bits(1)

        y0 = x0_0 & x1_0
        y1 = x0_01 & x1_1 | x0_1 & x1_01 | x0_1 & x1_1

        return logicvec(self._shape, y0 | y1)

    def uor(self) -> logic:
        """Return unary OR of bits."""
        y = logic.F
        for x in self.flat:
            y |= x
        return y

    def nand(self, other: Self) -> Self:
        """Return output of NAND function.

        y1 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[1] & x1[1]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_01 = x0_0 << 1
        x0_1 = self._bits(1)
        x0_10 = x0_1 >> 1

        x1_0 = other._bits(0)
        x1_01 = x1_0 << 1
        x1_1 = other._bits(1)
        x1_10 = x1_1 >> 1

        y0 = x0_10 & x1_10
        y1 = x0_01 & x1_01 | x0_01 & x1_1 | x0_1 & x1_01

        return logicvec(self._shape, y0 | y1)

    def and_(self, other: Self) -> Self:
        """Return output of AND function.

        y1 = x0[1] & x1[1]
        y0 = x0[0] & x1[0] | x0[0] & x1[1] | x0[1] & x1[0]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_1 = self._bits(1)
        x0_10 = x0_1 >> 1

        x1_0 = other._bits(0)
        x1_1 = other._bits(1)
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_1 & x1_1

        return logicvec(self._shape, y0 | y1)

    def uand(self) -> logic:
        """Return unary AND of bits."""
        y = logic.T
        for x in self.flat:
            y &= x
        return y

    def xnor(self, other: Self) -> Self:
        """Return output of XNOR function.

        y1 = x0[0] & x1[0] | x0[1] & x1[1]
        y0 = x0[0] & x1[1] | x0[1] & x1[0]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_01 = x0_0 << 1
        x0_1 = self._bits(1)
        x0_10 = x0_1 >> 1

        x1_0 = other._bits(0)
        x1_01 = x1_0 << 1
        x1_1 = other._bits(1)
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_10 | x0_10 & x1_0
        y1 = x0_01 & x1_01 | x0_1 & x1_1

        return logicvec(self._shape, y0 | y1)

    def xor(self, other: Self) -> Self:
        """Return output of XOR function.

        y1 = x0[0] & x1[1] | x0[1] & x1[0]
        y0 = x0[0] & x1[0] | x0[1] & x1[1]
        """
        self._check_shape(other)

        x0_0 = self._bits(0)
        x0_01 = x0_0 << 1
        x0_1 = self._bits(1)
        x0_10 = x0_1 >> 1

        x1_0 = other._bits(0)
        x1_01 = x1_0 << 1
        x1_1 = other._bits(1)
        x1_10 = x1_1 >> 1

        y0 = x0_0 & x1_0 | x0_10 & x1_10
        y1 = x0_01 & x1_1 | x0_1 & x1_01

        return logicvec(self._shape, y0 | y1)

    def uxor(self) -> logic:
        """Return unary XOR of bits."""
        y = logic.F
        for x in self.flat:
            y ^= x
        return y

    def to_uint(self) -> int:
        """Convert vector to unsigned integer."""
        y = 0
        for i, x in enumerate(self.flat):
            match x:
                case logic.F:
                    pass
                case logic.T:
                    y |= 1 << i
                case _:
                    raise ValueError("Cannot convert unknown to uint")
        return y

    def to_int(self) -> int:
        """Convert vector to signed integer."""
        if self._shape == (0,):
            return 0
        sign = logic(pcn.get_item(self._data, self.size - 1))
        if sign is logic.T:
            return -(self.not_().to_uint() + 1)
        return self.to_uint()

    def zext(self, n: int) -> Self:
        """Return vector zero extended by n bits.

        Zero extension is defined for 1-D vectors.
        Vectors of higher dimensions will be flattened, then zero extended.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return cat([v, uint2vec(0, n)], flatten=True)

    def sext(self, n: int) -> Self:
        """Return vector sign extended by n bits.

        Sign extension is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then sign extended.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return cat([v, rep(v[-1], n)], flatten=True)

    def lsh(self, n: int, ci: Self | None = None) -> tuple[Self, Self]:
        """Return vector left shifted by n bits.

        Left shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, _empty
        if ci is None:
            ci = uint2vec(0, n)
        elif ci.shape != (n,):
            raise ValueError(f"Expected ci to have shape ({n},)")
        return cat([ci, v[:-n]], flatten=True), v[-n:]

    def rsh(self, n: int, ci: Self | None = None) -> tuple[Self, Self]:
        """Return vector right shifted by n bits.

        Right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, _empty
        if ci is None:
            ci = uint2vec(0, n)
        elif ci.shape != (n,):
            raise ValueError(f"Expected ci to have shape ({n},)")
        return cat([v[n:], ci], flatten=True), v[:n]

    def arsh(self, n: int) -> tuple[Self, Self]:
        """Return vector arithmetically right shifted by n bits.

        Arithmetic right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, _empty
        return cat([v[n:], rep(v[-1], n)], flatten=True), v[:n]

    def add(self, other: Self, ci: object) -> tuple[Self, logic, logic]:
        """Return the sum of two vectors, carry out, and overflow.

        The implementation propagates Xes according to the
        ripple carry addition algorithm.
        """
        match ci:
            case logic():
                pass
            case _:
                ci = (logic.F, logic.T)[bool(ci)]
        s = []
        c = [ci]
        for i, (a, b) in enumerate(zip(self.flat, other.flat)):
            s.append(a ^ b ^ c[i])
            c.append(a & b | a & c[i] | b & c[i])
        co = c[-1]
        v = c[-2] ^ c[-1] if self.size else logic.F
        data = PcList(0)
        for i, x in enumerate(s):
            data = pcn.set_item(data, i, x.value)
        return logicvec((self.size,), data), co, v

    def countbits(self, ctl: Collection[logic]) -> int:
        """Return the number of bits in the ctl set."""
        return sum(1 for x in self.flat if x in ctl)

    def countones(self) -> int:
        """Return the number of ones in the vector."""
        return self.countbits({logic.T})

    def onehot(self) -> bool:
        """Return True if the vector has exactly one hot bit."""
        return self.countones() == 1

    def onehot0(self) -> bool:
        """Return True if the vector has at most one hot bit."""
        return self.countones() <= 1

    def isunknown(self) -> bool:
        """Return True if the vector contains at least one Null or X bit."""
        for x in self.flat:
            if x in {logic.N, logic.X}:
                return True
        return False

    def _to_lit(self) -> str:
        prefix = f"{self.size}b"
        chars = []
        for i in range(self.size):
            if i % 4 == 0 and i != 0:
                chars.append("_")
            chars.append(str(logic(pcn.get_item(self._data, i))))
        return prefix + "".join(reversed(chars))

    def _str(self, indent: str) -> str:
        """Helper funtion for __str__"""
        # Empty
        if self._shape == (0,):
            return "[]"
        # Scalar
        if self._shape == (1,):
            return f"[{logic(self._data)}]"
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

    @cached_property
    def _data_mask(self) -> tuple[PcList, PcList]:
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
        zero_mask = pcn.zeros(self.size)
        one_mask = PcList(zero_mask << 1)
        return (zero_mask, one_mask)

    def _bits(self, n: int) -> PcList:
        return PcList(self._data & self._data_mask[n])

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

    def _norm_key(self, key: _Key) -> tuple[int | slice, ...]:
        # First, convert key to a list
        match key:
            case int() | logicvec() | slice():
                lkey = [key]
            case tuple():
                lkey = list(key)
            case _:
                s = "Expected key to be int, slice, logicvec, or tuple"
                raise TypeError(s)

        ndim = len(lkey)
        if ndim > self.ndim:
            s = f"Expected ≤ {self.ndim} slice dimensions, got {ndim}"
            raise ValueError(s)

        # Append ':' to the end
        for _ in range(self.ndim - ndim):
            lkey.append(slice(None))

        # Normalize key dimensions
        lnkey: list[int | slice] = []
        for i, dim in enumerate(lkey):
            match dim:
                case int() as index:
                    lnkey.append(self._norm_index(index, i))
                case logicvec() as v:
                    lnkey.append(self._norm_index(v.to_uint(), i))
                case slice() as sl:
                    lnkey.append(self._norm_slice(sl, i))
                case _:  # pragma: no cover
                    assert False

        return tuple(lnkey)


def _parse_str_lit(lit: str) -> tuple[int, PcList]:
    if m := _NUM_RE.match(lit):
        # Binary
        if m.group("BinSize"):
            size = int(m.group("BinSize"))
            digits = m.group("BinDigits").replace("_", "")
            num_digits = len(digits)
            if num_digits != size:
                raise ValueError(f"Expected {size} digits, got {num_digits}")
            data = PcList(0)
            for i, digit in enumerate(reversed(digits)):
                data = pcn.set_item(data, i, pcn.from_char[digit])
            return size, data
        # Hexadecimal
        elif m.group("HexSize"):
            size = int(m.group("HexSize"))
            digits = m.group("HexDigits").replace("_", "")
            num_digits = len(digits)
            if 4 * num_digits != size:
                s = f"Expected size to match # digits, got {size} ≠ {4*num_digits}"
                raise ValueError(s)
            data = PcList(0)
            for i, digit in enumerate(reversed(digits)):
                data = pcn.set_byte(data, i, pcn.from_hexchar[digit])
            return size, data
        else:  # pragma: no cover
            assert False
    else:
        raise ValueError(f"Expected str literal, got {lit}")


def _rank1(fst: logic, rst) -> logicvec:
    shape = (len(rst) + 1,)
    data = pcn.set_item(PcList(0), 0, fst.value)
    for i, x in enumerate(rst, start=1):
        match x:
            case logic():
                data = pcn.set_item(data, i, x.value)
            case 0 | 1:
                data = pcn.set_item(data, i, pcn.from_int[x])
            case _:
                raise TypeError("Expected item to be logic, or in (0, 1)")
    return logicvec(shape, data)


def _rank2(fst: logicvec, rst) -> logicvec:
    shape = (len(rst) + 1,) + fst.shape
    data = fst.data
    for i, v in enumerate(rst, start=1):
        match v:
            case str() as lit:
                size, d = _parse_str_lit(lit)
                if size != fst.size:
                    s = f"Expected str literal to have size {fst.size}, got {size}"
                    raise TypeError(s)
                data |= d << (fst.nbits * i)
            case logicvec() if v.shape == fst.shape:
                data |= v.data << (fst.nbits * i)
            case _:
                s = ",".join(str(dim) for dim in fst.shape)
                s = f"Expected item to be str or logicvec[{s}]"
                raise TypeError(s)
    return logicvec(shape, data)


def vec(obj=None) -> logicvec:
    """
    Create a logic_vector.
    """
    match obj:
        # Empty
        case None:
            return _empty
        # Rank 0 Logic
        case logic() as x:
            return logicvec((1,), x.value)
        # Rank 0 int
        case 0 | 1 as x:
            return logicvec((1,), pcn.from_int[x])
        # Rank 1 str
        case str() as lit:
            size, data = _parse_str_lit(lit)
            return logicvec((size,), data)
        # Rank 1 [logic(), ...]
        case [logic() as x, *rst]:
            return _rank1(x, rst)
        # Rank 1 [0 | 1, ...]
        case [0 | 1 as x, *rst]:
            return _rank1(logic(pcn.from_int[x]), rst)
        # Rank 2 str
        case [str() as lit, *rst]:
            size, data = _parse_str_lit(lit)
            return _rank2(logicvec((size,), data), rst)
        # Rank 2 logic_vector
        case [logicvec() as v, *rst]:
            return _rank2(v, rst)
        # Rank 3+
        case [*objs]:
            return cat([vec(obj) for obj in objs])
        # Unimplemented
        case _:
            raise TypeError(f"Invalid input: {type(obj)}")


def uint2vec(num: int, size: int | None = None) -> logicvec:
    """Convert a nonnegative int to a logic_vector."""
    if num < 0:
        raise ValueError(f"Expected num ≥ 0, got {num}")

    if num == 0:
        index = 1
        data = pcn.set_item(PcList(0), 0, pcn.ZERO)
    else:
        index = 0
        data = PcList(0)
        while num:
            data = pcn.set_item(data, index, pcn.from_int[num & 1])
            index += 1
            num >>= 1

    if size is not None:
        if size < index:
            s = f"Overflow: num = {num} requires length ≥ {index}, got {size}"
            raise ValueError(s)
        for i in range(index, size):
            data = pcn.set_item(data, i, pcn.ZERO)
    else:
        size = index

    return logicvec((size,), data)


def cat(objs: Collection[_Logic], flatten: bool = False) -> logicvec:
    """Join a sequence of logicvecs."""
    # Empty
    if len(objs) == 0:
        return _empty

    # Convert inputs
    vs: list[logicvec] = []
    for obj in objs:
        match obj:
            case logic() as x:
                vs.append(logicvec((1,), x.value))
            case 0 | 1 as x:
                vs.append(logicvec((1,), pcn.from_int[x]))
            case logicvec() as v:
                vs.append(v)
            case _:
                raise TypeError(f"Invalid input: {type(obj)}")

    if len(vs) == 1:
        return vs[0]

    fst, rst = vs[0], vs[1:]
    scalar = fst.shape == (1,)
    regular = True
    dims = [fst.shape[0]]
    data = fst.data

    for i, v in enumerate(rst, start=1):
        if v.shape[0] != fst.shape[0]:
            regular = False
        if v.shape[1:] != fst.shape[1:]:
            s = f"Expected shape {fst.shape[1:]}, got {v.shape[1:]}"
            raise ValueError(s)
        dims.append(v.shape[0])
        data |= v.data << (fst.nbits * i)

    if not scalar and regular and not flatten:
        shape = (len(dims),) + fst.shape
    else:
        shape = (sum(dims),) + fst.shape[1:]

    return logicvec(shape, data)


def rep(v: _Logic, n: int, flatten: bool = False) -> logicvec:
    """Repeat a logicvec n times."""
    return cat([v] * n, flatten)


def _sel(v: logicvec, key: tuple[int | slice, ...]) -> _Logic:
    assert 0 <= v.ndim == len(key)

    shape = v.shape[1:]
    nbits = math.prod(shape) << 1
    mask = (1 << nbits) - 1

    def f(data: int, n: int) -> int:
        return (data >> (nbits * n)) & mask

    match key[0]:
        case int() as index:
            d = f(v.data, index)
            if shape:
                return _sel(logicvec(shape, d), key[1:])
            return logic(d)
        case slice() as sl:
            ds = [f(v.data, i) for i in range(sl.start, sl.stop, sl.step)]
            if shape:
                return cat([_sel(logicvec(shape, d), key[1:]) for d in ds])
            return cat([logic(d) for d in ds])
        case _:  # pragma: no cover
            assert False


# The empty vector is a singleton
_empty = logicvec((0,), 0)


def _consts(shape: tuple[int, ...], x: PcItem) -> logicvec:
    data = PcList(0)
    for i in range(math.prod(shape)):
        data = pcn.set_item(data, i, x)
    return logicvec(shape, data)


def nulls(shape: tuple[int, ...]) -> logicvec:
    """Return a new logic_vector of given shape, filled with NULLs."""
    return _consts(shape, pcn.NULL)


def zeros(shape: tuple[int, ...]) -> logicvec:
    """Return a new logic_vector of given shape, filled with zeros."""
    return _consts(shape, pcn.ZERO)


def ones(shape: tuple[int, ...]) -> logicvec:
    """Return a new logic_vector of given shape, filled with ones."""
    return _consts(shape, pcn.ONE)


def xes(shape: tuple[int, ...]) -> logicvec:
    """Return a new logic_vector of given shape, filled with Xes."""
    return _consts(shape, pcn.DC)
