"""Expression Symbolic Logic."""

# pylint: disable=exec-used

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

from .bits import (
    Bits,
    _lit2vec,
    adc,
    add,
    and_,
    cat,
    eq,
    ge,
    gt,
    ite,
    le,
    lrot,
    lsh,
    lt,
    mux,
    nand,
    ne,
    neg,
    ngc,
    nor,
    not_,
    or_,
    rep,
    rev,
    rrot,
    rsh,
    sbc,
    srsh,
    sub,
    sxt,
    uand,
    uor,
    uxnor,
    uxor,
    xnor,
    xor,
    xt,
)


def _arg_xbs(obj: Expr | Bits | str) -> Expr:
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, Bits):
        return BitsConst(obj)
    if isinstance(obj, str):
        v = _lit2vec(obj)
        return BitsConst(v)
    raise TypeError(f"Invalid input: {obj}")


def _arg_xbsi(obj: Expr | Bits | str | int) -> Expr:
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, Bits):
        return BitsConst(obj)
    if isinstance(obj, str):
        v = _lit2vec(obj)
        return BitsConst(v)
    if isinstance(obj, int):
        return IntConst(obj)
    raise TypeError(f"Invalid input: {obj}")


class Expr(ABC):
    """Symbolic expression."""

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __invert__(self) -> Not:
        return Not(self)

    def __or__(self, other: Expr | Bits | str) -> Or:
        return Or(self, other)

    def __ror__(self, other: Bits | str) -> Or:
        return Or(other, self)

    def __and__(self, other: Expr | Bits | str) -> And:
        return And(self, other)

    def __rand__(self, other: Bits | str) -> And:
        return And(other, self)

    def __xor__(self, other: Expr | Bits | str) -> Xor:
        return Xor(self, other)

    def __rxor__(self, other: Bits | str) -> Xor:
        return Xor(other, self)

    def __lshift__(self, other: Expr | Bits | str) -> Lsh:
        return Lsh(self, other)

    def __rlshift__(self, other: Bits | str) -> Lsh:
        return Lsh(other, self)

    def __rshift__(self, other: Expr | Bits | str) -> Rsh:
        return Rsh(self, other)

    def __rrshift__(self, other: Bits | str) -> Rsh:
        return Rsh(other, self)

    def iter_vars(self):
        raise NotImplementedError()

    @property
    def support(self) -> set[Variable]:
        return set(self.iter_vars())

    def to_func(self) -> tuple[Callable, list[Variable]]:
        vs = sorted(self.support, key=lambda v: v.name)
        args = ", ".join(v.name for v in vs)
        source = f"def f({args}):\n    return {self}\n"
        globals_ = {
            "not_": not_,
            "nor": nor,
            "or_": or_,
            "nand": nand,
            "and_": and_,
            "xnor": xnor,
            "xor": xor,
            "ite": ite,
            "mux": mux,
            "uor": uor,
            "uand": uand,
            "uxnor": uxnor,
            "uxor": uxor,
            "add": add,
            "adc": adc,
            "sub": sub,
            "sbc": sbc,
            "neg": neg,
            "ngc": ngc,
            "lsh": lsh,
            "rsh": rsh,
            "srsh": srsh,
            "xt": xt,
            "sxt": sxt,
            "lrot": lrot,
            "rrot": rrot,
            "cat": cat,
            "rep": rep,
            "rev": rev,
            "lt": lt,
            "le": le,
            "eq": eq,
            "ne": ne,
            "gt": gt,
            "ge": ge,
        }
        locals_ = {}
        exec(source, globals_, locals_)
        return locals_["f"], vs


class _Atom(Expr):
    """Atomic expression (leaf) node."""


class Const(Expr):
    """Constant node."""

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def iter_vars(self):
        yield from ()


class BitsConst(Const):
    """Const node."""

    def __str__(self) -> str:
        return f'"{self._value}"'


class IntConst(Const):
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
    def name(self):
        return self._name

    def iter_vars(self):
        yield self


class _Op(Expr):
    """Variable node."""

    def __init__(self, *xs: Expr):
        self._xs = xs

    def iter_vars(self):
        for x in self._xs:
            yield from x.iter_vars()


class _PrefixOp(_Op):
    """Prefix operator: f(x[0], x[1], ..., x[n-1])"""

    name = NotImplemented

    def __str__(self) -> str:
        s = ", ".join(str(x) for x in self._xs)
        return f"{self.name}({s})"


class _NaryOp(_PrefixOp):
    """N-ary operator: f(x0, x1, ..., x[n-1])"""

    def __init__(self, *objs: Expr | Bits | str):
        xs = tuple(_arg_xbs(obj) for obj in objs)
        super().__init__(*xs)


class _UnaryOp(_NaryOp):
    """Unary operator: f(x)"""

    def __init__(self, x: Expr | Bits | str):
        super().__init__(x)


class _BinaryOp(_NaryOp):
    """Binary operator: f(x0, x1)"""

    def __init__(self, x0: Expr | Bits | str, x1: Expr | Bits | str):
        super().__init__(x0, x1)


class _TernaryOp(_NaryOp):
    """Ternary operator: f(x0, x1, x2)"""

    def __init__(
        self,
        x0: Expr | Bits | str,
        x1: Expr | Bits | str,
        x2: Expr | Bits | str,
    ):
        super().__init__(x0, x1, x2)


class Not(_UnaryOp):
    """NOT operator node."""

    name = "not_"


class Nor(_NaryOp):
    """NOR operator node."""

    name = "nor"


class Or(_NaryOp):
    """OR operator node."""

    name = "or_"


class Nand(_NaryOp):
    """NAND operator node."""

    name = "nand"


class And(_NaryOp):
    """AND operator node."""

    name = "and_"


class Xnor(_NaryOp):
    """XNOR operator node."""

    name = "xnor"


class Xor(_NaryOp):
    """XOR operator node."""

    name = "xor"


class ITE(_TernaryOp):
    """If-Then-Else operator node."""

    name = "ite"


class Mux(Expr):
    """Multiplexer operator node."""

    def __init__(self, s: Expr | Bits | str, **xs: Expr | Bits | str):
        self._s = _arg_xbs(s)
        self._xs = {name: _arg_xbs(value) for name, value in xs.items()}

    def __str__(self) -> str:
        kwargs = ", ".join(f"{name}={value}" for name, value in self._xs.items())
        return f"mux({self._s}, {kwargs})"

    def iter_vars(self):
        yield from self._s.iter_vars()
        for x in self._xs.values():
            yield from x.iter_vars()


class Uor(_UnaryOp):
    """Unary OR reduction operator node."""

    name = "uor"


class Uand(_UnaryOp):
    """Unary AND reduction operator node."""

    name = "uand"


class Uxnor(_UnaryOp):
    """Unary XNOR reduction operator node."""

    name = "uxnor"


class Uxor(_UnaryOp):
    """Unary XOR reduction operator node."""

    name = "uxor"


class _AddOp(_PrefixOp):
    """Add or Adc operator node."""

    def __init__(
        self,
        a: Expr | Bits | str,
        b: Expr | Bits | str,
        ci: Expr | Bits | str | None = None,
    ):
        xs = [_arg_xbs(a), _arg_xbs(b)]
        if ci is not None:
            xs.append(_arg_xbs(ci))
        super().__init__(*xs)


class Add(_AddOp):
    """ADD operator node."""

    name = "add"


class Adc(_AddOp):
    """ADC operator node."""

    name = "adc"


class _SubOp(_PrefixOp):
    """Sub or Sbc operator node."""

    def __init__(self, a: Expr | Bits | str, b: Expr | Bits | str):
        super().__init__(_arg_xbs(a), _arg_xbs(b))


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


class _ShOp(_PrefixOp):
    """Shift operator node."""

    def __init__(self, x: Expr | Bits | str, n: Expr | Bits | str | int):
        super().__init__(_arg_xbs(x), _arg_xbsi(n))


class Lsh(_ShOp):
    """Left shift operator node."""

    name = "lsh"


class Rsh(_ShOp):
    """Right shift operator node."""

    name = "rsh"


class Srsh(_ShOp):
    """Signed right shift operator node."""

    name = "srsh"


class Xt(_PrefixOp):
    """Zero extend operator node."""

    name = "xt"

    def __init__(self, x: Expr | Bits | str, n: int):
        if isinstance(n, int):
            n = IntConst(n)
        else:
            raise TypeError(f"Invalid input: {n}")
        super().__init__(_arg_xbs(x), n)


class Sxt(_PrefixOp):
    """Sign extend operator node."""

    name = "sxt"

    def __init__(self, x: Expr | Bits | str, n: int):
        if isinstance(n, int):
            n = IntConst(n)
        else:
            raise TypeError(f"Invalid input: {n}")
        super().__init__(_arg_xbs(x), n)


class Lrot(_ShOp):
    """Left rotate operator node."""

    name = "lrot"


class Rrot(_ShOp):
    """Right rotate operator node."""

    name = "rrot"


class Cat(_NaryOp):
    """Concatenate operator node."""

    name = "cat"


class Rep(_PrefixOp):
    """Repeat operator node."""

    name = "rep"

    def __init__(self, x: Expr | Bits | str, n: int):
        if isinstance(n, int):
            n = IntConst(n)
        else:
            raise TypeError(f"Invalid input: {n}")
        super().__init__(_arg_xbs(x), n)


class Rev(_UnaryOp):
    """Reverse operator node."""

    name = "rev"


class EQ(_BinaryOp):
    """Equal (==) operator node."""

    name = "eq"


class NE(_BinaryOp):
    """NotEqual (!=) operator node."""

    name = "ne"


class LT(_BinaryOp):
    """LessThan (<) operator node."""

    name = "lt"


class LE(_BinaryOp):
    """Less Than Or Equal (≤) operator node."""

    name = "le"


class GT(_BinaryOp):
    """GreaterThan (>) operator node."""

    name = "gt"


class GE(_BinaryOp):
    """Greater Than Or Equal (≥) operator node."""

    name = "ge"


class GetItem(_Op):
    """GetItem operator node."""

    def __init__(self, x: Expr, obj: int | slice):
        if isinstance(obj, (int, slice)):
            key = Const(obj)
        else:
            raise TypeError(f"Invalid input: {obj}")
        super().__init__(x, key)

    def __str__(self) -> str:
        x = self._xs[0]
        key = self._xs[1].value
        match key:
            case int() as i:
                return f"{x}[{i}]"
            case slice() as sl:
                assert not (sl.start is None and sl.stop is None)
                if sl.start is None:
                    return f"{x}[:{sl.stop}]"
                if sl.stop is None:
                    return f"{x}[{sl.start}:]"
                return f"{x}[{sl.start}:{sl.stop}]"
            case _:
                assert False


class GetAttr(_Op):
    """GetAttr operator node."""

    def __init__(self, v: Variable, obj: str):
        if isinstance(obj, str):
            name = Const(obj)
        else:
            raise TypeError(f"Invalid input: {obj}")
        super().__init__(v, name)

    def __str__(self) -> str:
        v = self._xs[0]
        name = self._xs[1].value
        return f"{v}.{name}"
