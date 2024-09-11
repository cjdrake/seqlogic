"""Expression Symbolic Logic."""

# pylint: disable=exec-used

from __future__ import annotations

from collections.abc import Callable

from .bits import (
    Bits,
    _lit2vec,
    add,
    and_,
    eq,
    ge,
    gt,
    ite,
    le,
    lt,
    nand,
    ne,
    neg,
    nor,
    not_,
    or_,
    sub,
    xnor,
    xor,
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


def _arg_bs(obj: Bits | str) -> Expr:
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, Bits):
        return BitsConst(obj)
    if isinstance(obj, str):
        v = _lit2vec(obj)
        return BitsConst(v)
    raise TypeError(f"Invalid input: {obj}")


class Expr:
    """Symbolic expression."""

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, key: int | slice) -> GetItem:
        return GetItem(self, Const(key))

    def __invert__(self) -> Not:
        return Not(self)

    def __or__(self, other: Expr | Bits | str) -> Or:
        return Or(self, _arg_xbs(other))

    def __ror__(self, other: Bits | str) -> Or:
        return Or(_arg_bs(other), self)

    def __and__(self, other: Expr | Bits | str) -> And:
        return And(self, _arg_xbs(other))

    def __rand__(self, other: Bits | str) -> And:
        return And(_arg_bs(other), self)

    def __xor__(self, other: Expr | Bits | str) -> Xor:
        return Xor(self, _arg_xbs(other))

    def __rxor__(self, other: Bits | str) -> Xor:
        return Xor(_arg_bs(other), self)

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
            "add": add,
            "sub": sub,
            "neg": neg,
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

    def __init__(self, *objs: Expr | Bits | str):
        self._xs = tuple(_arg_xbs(obj) for obj in objs)

    def iter_vars(self):
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

    def __init__(self, x: Expr | Bits):
        super().__init__(x)


class _BinaryOp(_PrefixOp):
    """Binary operator: f(x0, x1)"""

    def __init__(self, x0: Expr | Bits, x1: Expr | Bits):
        super().__init__(x0, x1)


class _TernaryOp(_PrefixOp):
    """Ternary operator: f(x0, x1, x2)"""

    def __init__(self, x0: Expr | Bits, x1: Expr | Bits, x2: Expr | Bits):
        super().__init__(x0, x1, x2)


class Not(_UnaryOp):
    """NOT operator node."""

    name = "not_"


class Nor(_PrefixOp):
    """NOR operator node."""

    name = "nor"


class Or(_PrefixOp):
    """OR operator node."""

    name = "or_"


class Nand(_PrefixOp):
    """NAND operator node."""

    name = "nand"


class And(_PrefixOp):
    """AND operator node."""

    name = "and_"


class Xnor(_PrefixOp):
    """XNOR operator node."""

    name = "xnor"


class Xor(_PrefixOp):
    """XOR operator node."""

    name = "xor"


class ITE(_TernaryOp):
    """If-Then-Else operator node."""

    name = "ite"


class Add(_BinaryOp):
    """ADD operator node."""

    name = "add"


class Sub(_BinaryOp):
    """SUB operator node."""

    name = "sub"


class Neg(_UnaryOp):
    """NEG operator node."""

    name = "neg"


class GetItem(_Op):
    """GetItem operator node."""

    def __init__(self, x: Expr, key: Const):
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

    def __init__(self, v: Variable, obj: Const | str):
        if isinstance(obj, Const):
            name = obj
        elif isinstance(obj, str):
            name = Const(obj)
        else:
            raise TypeError(f"Invalid input: {obj}")
        super().__init__(v, name)

    def __str__(self) -> str:
        v = self._xs[0]
        name = self._xs[1].value
        return f"{v}.{name}"


class LT(_BinaryOp):
    """LessThan (<) operator node."""

    name = "lt"


class LE(_BinaryOp):
    """Less Than Or Equal (≤) operator node."""

    name = "le"


class EQ(_BinaryOp):
    """Equal (==) operator node."""

    name = "eq"


class NE(_BinaryOp):
    """NotEqual (!=) operator node."""

    name = "ne"


class GT(_BinaryOp):
    """GreaterThan (>) operator node."""

    name = "gt"


class GE(_BinaryOp):
    """Greater Than Or Equal (≥) operator node."""

    name = "ge"
