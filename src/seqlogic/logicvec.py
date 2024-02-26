"""Logic Vector Data Type."""

# pylint: disable = protected-access

from __future__ import annotations

import math
import re
from collections.abc import Collection, Generator
from functools import cached_property

from . import pcn
from .pcn import PcVec

_NUM_RE = re.compile(
    r"((?P<BinSize>[0-9]+)b(?P<BinDigits>[X01x_]+))|"
    r"((?P<HexSize>[0-9]+)h(?P<HexDigits>[0-9a-fA-F_]+))"
)


class logicvec:
    """Logic vector data type.

    Do NOT instantiate this type directly.
    Use the factory functions instead.
    """

    def __init__(self, w: PcVec, shape: tuple[int, ...] | None = None):
        """TODO(cjdrake): Write docstring."""
        self._w = w
        if shape is None:
            self._shape = (len(w),)
        else:
            assert math.prod(shape) == len(w)
            self._shape = shape

    def __str__(self) -> str:
        indent = "     "
        return f"vec({self._str(indent)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self._shape[0]

    def __iter__(self) -> Generator[logicvec, None, None]:
        for i in range(self._shape[0]):
            yield self.__getitem__(i)

    def __getitem__(
        self, key: int | logicvec | slice | tuple[int | logicvec | slice, ...]
    ) -> logicvec:
        if self._shape == (0,):
            raise IndexError("Cannot index an empty vector")
        match key:
            case int() | logicvec() | slice():
                return _sel(self, self._norm_key([key]))
            case tuple():
                return _sel(self, self._norm_key(list(key)))
            case _:
                s = "Expected key to be int, logicvec, slice, or tuple"
                raise TypeError(s)

    def __eq__(self, other) -> bool:
        match other:
            case logicvec():
                return self._w.data == other._w.data and self._shape == other.shape
            case _:
                return False

    def __invert__(self) -> logicvec:
        return self.lnot()

    def __or__(self, other: logicvec) -> logicvec:
        return self.lor(other)

    def __and__(self, other: logicvec) -> logicvec:
        return self.land(other)

    def __xor__(self, other: logicvec) -> logicvec:
        return self.lxor(other)

    def __lshift__(self, n: int | logicvec) -> logicvec:
        return self.lsh(n)[0]

    def __rshift__(self, n: int | logicvec) -> logicvec:
        return self.rsh(n)[0]

    def __add__(self, other: logicvec) -> logicvec:
        return logicvec(self._w.__add__(other._w))

    def __sub__(self, other: logicvec) -> logicvec:
        return logicvec(self._w.__sub__(other._w))

    def __neg__(self) -> logicvec:
        return logicvec(self._w.__neg__())

    @property
    def shape(self) -> tuple[int, ...]:
        """Return logicvec shape."""
        return self._shape

    def reshape(self, shape: tuple[int, ...]) -> logicvec:
        """Return an equivalent logic_vector with modified shape."""
        if math.prod(shape) != self.size:
            s = f"Expected shape with size {self.size}, got {shape}"
            raise ValueError(s)
        return logicvec(self._w, shape)

    @cached_property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Number of elements in the vector."""
        return len(self._w)

    @property
    def flat(self) -> Generator[logicvec, None, None]:
        """Return a flat iterator to the items."""
        for x in self._w:
            yield logicvec(x)

    def flatten(self) -> logicvec:
        """Return a vector with equal data, flattened to 1D shape."""
        return logicvec(self._w)

    def _check_shape(self, other: logicvec):
        if self._shape != other.shape:
            s = f"Expected shape {self._shape}, got {other.shape}"
            raise ValueError(s)

    def lnot(self) -> logicvec:
        """Return output of "lifted" NOT function."""
        return logicvec(self._w.lnot())

    def lnor(self, other: logicvec) -> logicvec:
        """Return output of "lifted" NOR function."""
        self._check_shape(other)
        return logicvec(self._w.lnor(other._w))

    def lor(self, other: logicvec) -> logicvec:
        """Return output of "lifted" OR function."""
        self._check_shape(other)
        return logicvec(self._w.lor(other._w))

    def ulor(self) -> logicvec:
        """Return unary "lifted" OR of bits."""
        return logicvec(self._w.ulor())

    def lnand(self, other: logicvec) -> logicvec:
        """Return output of "lifted" NAND function."""
        self._check_shape(other)
        return logicvec(self._w.lnand(other._w))

    def land(self, other: logicvec) -> logicvec:
        """Return output of "lifted" AND function."""
        self._check_shape(other)
        return logicvec(self._w.land(other._w))

    def uland(self) -> logicvec:
        """Return unary "lifted" AND of bits."""
        return logicvec(self._w.uland())

    def lxnor(self, other: logicvec) -> logicvec:
        """Return output of "lifted" XNOR function."""
        self._check_shape(other)
        return logicvec(self._w.lxnor(other._w))

    def lxor(self, other: logicvec) -> logicvec:
        """Return output of "lifted" XOR function."""
        self._check_shape(other)
        return logicvec(self._w.lxor(other._w))

    def ulxor(self) -> logicvec:
        """Return unary "lifted" XOR of bits."""
        return logicvec(self._w.ulxor())

    def to_uint(self) -> int:
        """Convert vector to unsigned integer."""
        return self._w.to_uint()

    def to_int(self) -> int:
        """Convert vector to signed integer."""
        return self._w.to_int()

    def zext(self, n: int) -> logicvec:
        """Return vector zero extended by n bits.

        Zero extension is defined for 1-D vectors.
        Vectors of higher dimensions will be flattened, then zero extended.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return logicvec(v._w.zext(n))

    def sext(self, n: int) -> logicvec:
        """Return vector sign extended by n bits.

        Sign extension is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then sign extended.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        return logicvec(v._w.sext(n))

    def lsh(self, n: int | logicvec, ci: logicvec | None = None) -> tuple[logicvec, logicvec]:
        """Return vector left shifted by n bits.

        Left shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        match n:
            case int():
                pass
            case logicvec():
                if n._w.has_null():
                    return nulls((v.size,)), E
                elif n._w.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or logicvec")
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, E
        if ci is None:
            ci = uint2vec(0, n)
        elif ci.shape != (n,):
            raise ValueError(f"Expected ci to have shape ({n},)")
        y, co = self._w.lsh(n, ci._w)
        return logicvec(y), logicvec(co)

    def rsh(self, n: int | logicvec, ci: logicvec | None = None) -> tuple[logicvec, logicvec]:
        """Return vector right shifted by n bits.

        Right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        match n:
            case int():
                pass
            case logicvec():
                if n._w.has_null():
                    return nulls((v.size,)), E
                elif n._w.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or logicvec")
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, E
        if ci is None:
            ci = uint2vec(0, n)
        elif ci.shape != (n,):
            raise ValueError(f"Expected ci to have shape ({n},)")
        y, co = self._w.rsh(n, ci._w)
        return logicvec(y), logicvec(co)

    def arsh(self, n: int | logicvec) -> tuple[logicvec, logicvec]:
        """Return vector arithmetically right shifted by n bits.

        Arithmetic right shift is defined for 1-D vectors.
        Vectors of higher dimension will be flattened, then shifted.
        """
        v = self
        if self.ndim != 1:
            v = self.flatten()
        match n:
            case int():
                pass
            case logicvec():
                if n._w.has_null():
                    return nulls((v.size,)), E
                elif n._w.has_dc():
                    return xes((v.size,)), E
                else:
                    n = n.to_uint()
            case _:
                raise TypeError("Expected n to be int or logicvec")
        if not 0 <= n <= v.size:
            raise ValueError(f"Expected 0 ≤ n ≤ {v.size}, got {n}")
        if n == 0:
            return v, E
        y, co = self._w.arsh(n)
        return logicvec(y), logicvec(co)

    def add(self, other: logicvec, ci: object) -> tuple[logicvec, logicvec, logicvec]:
        """Return the sum of two vectors, carry out, and overflow.

        The implementation propagates Xes according to the
        ripple carry addition algorithm.
        """
        match ci:
            case logicvec():
                pass
            case _:
                ci = (F, T)[bool(ci)]
        s, co, ovf = self._w.add(other._w, ci._w)
        return logicvec(s), logicvec(co), logicvec(ovf)

    def _to_lit(self) -> str:
        prefix = f"{self.size}b"
        chars = []
        for i, x in enumerate(self.flat):
            if i % 4 == 0 and i != 0:
                chars.append("_")
            chars.append(pcn.to_char[x._w.data])
        return prefix + "".join(reversed(chars))

    def _str(self, indent: str) -> str:
        """Help __str__ method recursion."""
        # Empty
        if self._shape == (0,):
            return "[]"
        # Scalar
        if self._shape == (1,):
            return "[" + pcn.to_char[self._w.data] + "]"
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

    def _norm_key(self, key: list[int | logicvec | slice]) -> tuple[int | slice, ...]:
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
                case logicvec() as v:
                    nkey.append(self._norm_index(v.to_uint(), i))
                case slice() as sl:
                    nkey.append(self._norm_slice(sl, i))
                case _:  # pragma: no cover
                    assert False

        return tuple(nkey)


def _parse_str_lit(lit: str) -> PcVec:
    if m := _NUM_RE.match(lit):
        # Binary
        if m.group("BinSize"):
            size = int(m.group("BinSize"))
            digits = m.group("BinDigits").replace("_", "")
            ndigits = len(digits)
            if ndigits != size:
                s = f"Expected {size} digits, got {ndigits}"
                raise ValueError(s)
            return pcn.from_pcitems(pcn.from_char[c] for c in reversed(digits))
        # Hexadecimal
        elif m.group("HexSize"):
            size = int(m.group("HexSize"))
            digits = m.group("HexDigits").replace("_", "")
            ndigits = len(digits)
            if 4 * ndigits != size:
                s = f"Expected size to match # digits, got {size} ≠ {4 * ndigits}"
                raise ValueError(s)
            return pcn.from_quads(pcn.from_hexchar[c] for c in reversed(digits))
        else:  # pragma: no cover
            assert False
    else:
        raise ValueError(f"Expected str literal, got {lit}")


def _rank1(fst: int, rst) -> logicvec:
    pcitems = [fst]
    for x in rst:
        match x:
            case 0 | 1:
                pcitems.append(pcn.from_int[x])
            case _:
                raise TypeError("Expected item to be logic, or in (0, 1)")
    return logicvec(pcn.from_pcitems(pcitems))


def _rank2(fst: logicvec, rst) -> logicvec:
    shape = (len(rst) + 1,) + fst.shape
    size = len(fst._w)
    data = fst._w.data
    for i, v in enumerate(rst, start=1):
        match v:
            case str() as lit:
                w = _parse_str_lit(lit)
                if len(w) != fst.size:
                    s = f"Expected str literal to have size {fst.size}, got {len(w)}"
                    raise TypeError(s)
                data |= w.data << (fst._w.nbits * i)
            case logicvec() if v.shape == fst.shape:
                data |= v._w.data << (fst._w.nbits * i)
            case _:
                s = ",".join(str(dim) for dim in fst.shape)
                s = f"Expected item to be str or logicvec[{s}]"
                raise TypeError(s)
        size += len(fst._w)
    return logicvec(PcVec(size, data), shape)


def vec(obj=None) -> logicvec:
    """Create a logic_vector."""
    match obj:
        # Empty
        case None:
            return E
        # Rank 0 int
        case 0 | 1 as x:
            return logicvec(pcn.from_pcitems([pcn.from_int[x]]))
        # Rank 1 str
        case str() as lit:
            return logicvec(_parse_str_lit(lit))
        # Rank 1 [0 | 1, ...]
        case [0 | 1 as x, *rst]:
            return _rank1(pcn.from_int[x], rst)
        # Rank 2 str
        case [str() as lit, *rst]:
            return _rank2(logicvec(_parse_str_lit(lit)), rst)
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

    pcitems = []
    if num == 0:
        index = 1
        pcitems.append(pcn.ZERO)
    else:
        index = 0
        while num:
            pcitems.append(pcn.from_int[num & 1])
            index += 1
            num >>= 1

    if size is None:
        size = index
    else:
        if size < index:
            s = f"Overflow: num = {num} requires length ≥ {index}, got {size}"
            raise ValueError(s)
        for _ in range(index, size):
            pcitems.append(pcn.ZERO)

    return logicvec(pcn.from_pcitems(pcitems))


def cat(objs: Collection[int | logicvec], flatten: bool = False) -> logicvec:
    """Join a sequence of logicvecs."""
    # Empty
    if len(objs) == 0:
        return E

    # Convert inputs
    vs: list[logicvec] = []
    for obj in objs:
        match obj:
            case 0 | 1 as x:
                vs.append(logicvec(pcn.from_pcitems([pcn.from_int[x]])))
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

    return logicvec(PcVec(size, data), shape)


def rep(obj: int | logicvec, n: int, flatten: bool = False) -> logicvec:
    """Repeat a logicvec n times."""
    return cat([obj] * n, flatten)


def _sel(v: logicvec, key: tuple[int | slice, ...]) -> logicvec:
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
                return _sel(logicvec(PcVec(n, data), shape), key[1:])
            return logicvec(PcVec(1, data))
        case slice() as sl:
            datas = (f(v._w.data, i) for i in range(sl.start, sl.stop, sl.step))
            if shape:
                return cat([_sel(logicvec(PcVec(n, data), shape), key[1:]) for data in datas])
            return cat([logicvec(PcVec(1, data)) for data in datas])
        case _:  # pragma: no cover
            assert False


# The empty vector is a singleton
E = logicvec(PcVec(0, 0))


def _consts(shape: tuple[int, ...], x: int) -> logicvec:
    num = math.prod(shape)
    return logicvec(pcn.from_pcitems([x] * num), shape)


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


# One bit values
N = nulls((1,))
F = zeros((1,))
T = ones((1,))
X = xes((1,))
