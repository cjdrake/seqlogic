"""Expression Symbolic Logic."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Generator
from typing import Any

from bvwx import (
    Array,
    Scalar,
    adc,
    add,
    and_,
    bits,
    cat,
    div,
    eq,
    ge,
    gt,
    impl,
    ite,
    land,
    le,
    lor,
    lrot,
    lsh,
    lt,
    lxor,
    mod,
    mul,
    mux,
    ne,
    neg,
    ngc,
    not_,
    or_,
    rep,
    rrot,
    rsh,
    sbc,
    srsh,
    sub,
    sxt,
    uand,
    uor,
    uxor,
    xor,
    xt,
)

type BitsOp = Callable[..., Array]

_OPS: list[BitsOp] = [
    not_,
    or_,
    and_,
    xor,
    impl,
    ite,
    mux,
    lor,
    land,
    lxor,
    uor,
    uand,
    uxor,
    add,
    adc,
    sub,
    sbc,
    neg,
    ngc,
    mul,
    div,
    mod,
    lsh,
    rsh,
    srsh,
    xt,
    sxt,
    lrot,
    rrot,
    cat,
    rep,
    lt,
    le,
    eq,
    ne,
    gt,
    ge,
]

_GLOBALS: dict[str, BitsOp] = {f.__name__: f for f in _OPS}


def _b2c(arg: int) -> BitsConst:
    if arg in (0, 1):
        return BitsConst(bits(arg))
    else:
        raise ValueError(f"Expected arg: int in {{0, 1}}, got {arg}")


def _expect_bits(arg: ExprLike) -> Expr:
    """Any Bits-like object that defines its own size"""
    if isinstance(arg, int):
        return _b2c(arg)
    if isinstance(arg, str):
        return BitsConst(bits(arg))
    if isinstance(arg, Array):
        return BitsConst(arg)
    if isinstance(arg, Expr):
        return arg
    raise TypeError("Expected arg to be: Expr, Array, or str literal, or {0, 1}")


def _expect_scalar(arg: ScalarLike) -> Expr:
    """Any Scalar-like object"""
    if isinstance(arg, int):
        return _b2c(arg)
    if isinstance(arg, str):
        return BitsConst(bits(arg))
    if isinstance(arg, Scalar):
        return BitsConst(arg)
    if isinstance(arg, Expr):
        return arg
    raise TypeError("Expected arg to be: Expr, Scalar, str literal, or {0, 1}")


def _expect_bits_size(arg: ExprLike) -> Expr:
    """Any Bits-Like object that may or may not define its own size"""
    if isinstance(arg, int):
        return IntConst(arg)
    if isinstance(arg, str):
        return BitsConst(bits(arg))
    if isinstance(arg, Array):
        return BitsConst(arg)
    if isinstance(arg, Expr):
        return arg
    raise TypeError("Expected arg to be: Expr, Scalar, or str literal, or int")


def _expect_uint(arg: UintLike) -> Expr:
    """Any Uint-Like object that may or may not define its own size"""
    if isinstance(arg, int):
        return IntConst(arg)
    if isinstance(arg, str):
        return BitsConst(bits(arg))
    if isinstance(arg, Array):
        return BitsConst(arg)
    if isinstance(arg, Expr):
        return arg
    raise TypeError("Expected arg to be: Expr, Scalar, or str literal, or int")


class Expr(ABC):
    """Symbolic expression."""

    def __str__(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    def __invert__(self) -> Not:
        return Not(self)

    def __or__(self, other: ExprLike) -> Or:
        return Or(self, other)

    def __ror__(self, other: ConstLike) -> Or:
        return Or(other, self)

    def __and__(self, other: ExprLike) -> And:
        return And(self, other)

    def __rand__(self, other: ConstLike) -> And:
        return And(other, self)

    def __xor__(self, other: ExprLike) -> Xor:
        return Xor(self, other)

    def __rxor__(self, other: ConstLike) -> Xor:
        return Xor(other, self)

    def __lshift__(self, other: UintLike) -> Lsh:
        return Lsh(self, other)

    def __rlshift__(self, other: ConstLike) -> Lsh:
        return Lsh(other, self)

    def __rshift__(self, other: UintLike) -> Rsh:
        return Rsh(self, other)

    def __rrshift__(self, other: ConstLike) -> Rsh:
        return Rsh(other, self)

    def iter_vars(self) -> Generator[Variable, None, None]:
        raise NotImplementedError()  # pragma: no cover

    @property
    def support(self) -> frozenset[Variable]:
        return frozenset(self.iter_vars())

    def to_func(self) -> tuple[BitsOp, list[Variable]]:
        vs = sorted(self.support, key=lambda v: v.name)
        args = ", ".join(v.name for v in vs)
        source = f"def f({args}):\n    return {self}\n"
        _locals: dict[str, Any] = {}
        exec(source, _GLOBALS, _locals)
        return _locals["f"], vs


# Type Aliases
type ConstLike = str | int
type ExprLike = Expr | Array | str | int
type ScalarLike = Expr | Scalar | str | int
type UintLike = Expr | Array | str | int


class _Atom(Expr):
    """Atomic expression (leaf) node."""


class Const[T](Expr):
    """Constant node."""

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    def iter_vars(self) -> Generator[Variable, None, None]:
        yield from ()


class BitsConst(Const[Array]):
    """Const node."""

    def __str__(self) -> str:
        return f'"{self._value}"'


class IntConst(Const[int]):
    """Integer node."""

    def __str__(self) -> str:
        return str(self._value)


class Variable(_Atom):
    """Variable node."""

    # TODO(cjdrake): Disambiguate from Hierarchy._name
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name

    def iter_vars(self) -> Generator[Variable, None, None]:
        yield self


class _Op(Expr):
    """Variable node."""

    def __init__(self, *xs: Expr):
        self._xs = xs

    @property
    def xs(self) -> tuple[Expr, ...]:
        return self._xs

    def iter_vars(self) -> Generator[Variable, None, None]:
        for x in self._xs:
            yield from x.iter_vars()


class _PrefixOp(_Op):
    """Prefix operator: f(x[0], x[1], ..., x[n-1])"""

    name = NotImplemented

    def __str__(self) -> str:
        s = ", ".join(str(x) for x in self._xs)
        return f"{self.name}({s})"


class _UnaryOp(_PrefixOp):
    """Unary operator: f(x)"""

    def __init__(self, x: ExprLike):
        x = _expect_bits(x)
        super().__init__(x)


class _BinOp1(_PrefixOp):
    """Binary operator: f(x0, x1)"""

    def __init__(self, x0: ExprLike, x1: ExprLike):
        x0 = _expect_bits(x0)
        x1 = _expect_bits_size(x1)
        super().__init__(x0, x1)


class _BinOp2(_PrefixOp):
    """Add or Adc operator node."""

    def __init__(self, a: ExprLike, b: ExprLike):
        a = _expect_bits(a)
        b = _expect_bits(b)
        super().__init__(a, b)


class _BinOp3(_PrefixOp):
    """Shift operator node."""

    def __init__(self, x: ExprLike, n: UintLike):
        x = _expect_bits(x)
        n = _expect_uint(n)
        super().__init__(x, n)


class _NaryOp1(_PrefixOp):
    """Nary operator: f(x0, ...)"""

    def __init__(self, x0: ExprLike, *xs: ExprLike):
        x0 = _expect_bits(x0)
        xs_ = [_expect_bits_size(x) for x in xs]
        super().__init__(x0, *xs_)


class _NaryOp2(_PrefixOp):
    """Nary operator: f(...)"""

    def __init__(self, *xs: ScalarLike):
        xs_ = [_expect_scalar(x) for x in xs]
        super().__init__(*xs_)


class _NaryOp3(_PrefixOp):
    """Nary operator: f(...)"""

    def __init__(self, *xs: ExprLike):
        xs_ = [_expect_bits(x) for x in xs]
        super().__init__(*xs_)


class Not(_UnaryOp):
    """NOT operator node."""

    name = "not_"


class Or(_NaryOp1):
    """OR operator node."""

    name = "or_"


class And(_NaryOp1):
    """AND operator node."""

    name = "and_"


class Xor(_NaryOp1):
    """XOR operator node."""

    name = "xor"


class Impl(_BinOp1):
    """Implies operator node."""

    name = "impl"


class ITE(_PrefixOp):
    """If-Then-Else operator node."""

    name = "ite"

    def __init__(self, s: ScalarLike, x1: ExprLike, x0: ExprLike):
        s = _expect_scalar(s)
        x1 = _expect_bits(x1)
        x0 = _expect_bits_size(x0)
        super().__init__(s, x1, x0)


class Mux(Expr):
    """Multiplexer operator node."""

    def __init__(self, s: ExprLike, **xs: ExprLike):
        self._s = _expect_bits(s)
        self._xs = {k: _expect_bits_size(v) for k, v in xs.items()}

    def __str__(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self._xs.items())
        return f"mux({self._s}, {kwargs})"

    def iter_vars(self) -> Generator[Variable, None, None]:
        yield from self._s.iter_vars()
        for x in self._xs.values():
            yield from x.iter_vars()


class Lor(_NaryOp2):
    """Logical OR"""

    name = "lor"


class Land(_NaryOp2):
    """Logical AND"""

    name = "land"


class Lxor(_NaryOp2):
    """Logical XOR"""

    name = "lxor"


class Uor(_UnaryOp):
    """Unary OR reduction operator node."""

    name = "uor"


class Uand(_UnaryOp):
    """Unary AND reduction operator node."""

    name = "uand"


class Uxor(_UnaryOp):
    """Unary XOR reduction operator node."""

    name = "uxor"


class _AddOp(_PrefixOp):
    """Add or Adc operator node."""

    def __init__(
        self,
        a: ExprLike,
        b: ExprLike,
        ci: ScalarLike | None = None,
    ):
        xs: list[Expr] = []
        xs.append(_expect_bits(a))
        xs.append(_expect_bits(b))
        if ci is not None:
            xs.append(_expect_scalar(ci))
        super().__init__(*xs)


class Add(_AddOp):
    """ADD operator node."""

    name = "add"


class Adc(_AddOp):
    """ADC operator node."""

    name = "adc"


class _SubOp(_PrefixOp):
    """Sub or Sbc operator node."""

    def __init__(self, a: ExprLike, b: ExprLike):
        a = _expect_bits(a)
        b = _expect_bits_size(b)
        super().__init__(a, b)


class Sub(_SubOp):
    """SUB operator node."""

    name = "sub"


class Sbc(_SubOp):
    """SBC operator node."""

    name = "sbc"


class Neg(_UnaryOp):
    """NEG operator node."""

    name = "neg"


class Ngc(_UnaryOp):
    """NGC operator node."""

    name = "ngc"


class Mul(_BinOp2):
    """Multiply operator node."""

    name = "mul"


class Div(_BinOp2):
    """Divide operator node."""

    name = "div"


class Mod(_BinOp2):
    """Modulo operator node."""

    name = "mod"


class Lsh(_BinOp3):
    """Left shift operator node."""

    name = "lsh"


class Rsh(_BinOp3):
    """Right shift operator node."""

    name = "rsh"


class Srsh(_BinOp3):
    """Signed right shift operator node."""

    name = "srsh"


class Xt(_BinOp3):
    """Zero extend operator node."""

    name = "xt"


class Sxt(_BinOp3):
    """Sign extend operator node."""

    name = "sxt"


class Lrot(_BinOp3):
    """Left rotate operator node."""

    name = "lrot"


class Rrot(_BinOp3):
    """Right rotate operator node."""

    name = "rrot"


class Cat(_NaryOp3):
    """Concatenate operator node."""

    name = "cat"


class Rep(_PrefixOp):
    """Repeat operator node."""

    name = "rep"

    def __init__(self, x: ExprLike, n: int):
        if isinstance(n, int):
            _n = IntConst(n)
        else:
            raise TypeError(f"Invalid input: {n}")
        x = _expect_bits(x)
        super().__init__(x, _n)


class EQ(_BinOp1):
    """Equal (==) operator node."""

    name = "eq"


class NE(_BinOp1):
    """NotEqual (!=) operator node."""

    name = "ne"


class LT(_BinOp1):
    """LessThan (<) operator node."""

    name = "lt"


class LE(_BinOp1):
    """Less Than Or Equal (≤) operator node."""

    name = "le"


class GT(_BinOp1):
    """GreaterThan (>) operator node."""

    name = "gt"


class GE(_BinOp1):
    """Greater Than Or Equal (≥) operator node."""

    name = "ge"


class GetItem(_Op):
    """GetItem operator node."""

    def __init__(self, v: Variable, key: int | slice):
        if not isinstance(key, (int, slice)):
            raise TypeError(f"Invalid key: {key}")
        self._xs = (v, Const(key))

    def __str__(self) -> str:
        v = self._xs[0]
        key = self._xs[1].value
        match key:
            case int() as i:
                return f"{v}[{i}]"
            case slice() as sl:
                assert not (sl.start is None and sl.stop is None)
                if sl.start is None:
                    return f"{v}[:{sl.stop}]"
                if sl.stop is None:
                    return f"{v}[{sl.start}:]"
                return f"{v}[{sl.start}:{sl.stop}]"
            case _:  # pragma: no cover
                assert False


class GetAttr(_Op):
    """GetAttr operator node."""

    def __init__(self, v: Variable, key: str):
        if not isinstance(key, str):
            raise TypeError(f"Invalid key: {key}")
        self._xs = (v, Const(key))

    def __str__(self) -> str:
        v = self._xs[0]
        key = self._xs[1].value
        return f"{v}.{key}"
