"""Expression Symbolic Logic."""

from __future__ import annotations

from abc import ABC
from enum import Enum, auto

from .bits import Bits


class Op(Enum):
    """Expression opcode."""

    # Bitwise
    NOT = auto()
    OR = auto()
    AND = auto()
    XOR = auto()

    # Word
    GETITEM = auto()
    GETATTR = auto()

    # Comparison
    LT = auto()
    LE = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    GE = auto()


class Expr(ABC):
    """Symbolic expression."""

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, key: int | slice) -> GetItem:
        return GetItem(self, Constant(key))

    def __invert__(self) -> Not:
        return Not(self)

    def __or__(self, other: Expr) -> Or:
        return Or(self, other)

    def __and__(self, other: Expr) -> And:
        return And(self, other)

    def __xor__(self, other: Expr) -> Xor:
        return Xor(self, other)

    def iter_vars(self):
        raise NotImplementedError()


class Atom(Expr):
    """Atomic expression (leaf) node."""


class Constant(Expr):
    """Constant node."""

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def iter_vars(self):
        yield from ()


class BitsConst(Constant):
    """Const node."""

    def __str__(self) -> str:
        return f'"{self._value}"'


class Variable(Atom):
    """Variable node."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name

    def iter_vars(self):
        yield self


class Operator(Expr):
    """Variable node."""

    def __init__(self, *xs: Expr):
        self._xs = xs

    def iter_vars(self):
        for x in self._xs:
            yield from x.iter_vars()


class GetItem(Operator):
    """GetItem operator node."""

    def __init__(self, v: Variable, key: Constant):
        super().__init__(v, key)

    def __str__(self) -> str:
        v, key = self._xs
        match key.value:
            case int() as i:
                return f"{v}[{i}]"
            case slice() as sl:
                assert not (sl.start is None and sl.stop is None)
                if sl.start is None:
                    return f"{v}[:{sl.stop}]"
                if sl.stop is None:
                    return f"{v}[{sl.start}:]"
                return f"{v}[{sl.start}:{sl.stop}]"
            case _:
                assert False


class GetAttr(Operator):
    """GetAttr operator node."""

    def __init__(self, v: Variable, name: Constant):
        super().__init__(v, name)

    def __str__(self) -> str:
        v, name = self._xs
        return f"{v}.{name.value}"


class Not(Operator):
    """NOT operator node."""

    def __init__(self, x: Expr):
        super().__init__(x)

    def __str__(self) -> str:
        x = self._xs[0]
        return f"~{x}"


class Or(Operator):
    """OR operator node."""

    def __str__(self) -> str:
        s = " | ".join(str(x) for x in self._xs)
        return f"({s})"


class And(Operator):
    """AND operator node."""

    def __str__(self) -> str:
        s = " & ".join(str(x) for x in self._xs)
        return f"({s})"


class Xor(Operator):
    """XOR operator node."""

    def __str__(self) -> str:
        s = " ^ ".join(str(x) for x in self._xs)
        return f"({s})"


class LessThan(Operator):
    """LessThan (<) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.lt({x1})"


class LessEqual(Operator):
    """Less Than Or Equal (≤) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.le({x1})"


class Equal(Operator):
    """Equal (==) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.eq({x1})"


class NotEqual(Operator):
    """NotEqual (!=) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.ne({x1})"


class GreaterThan(Operator):
    """GreaterThan (>) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.gt({x1})"


class GreaterEqual(Operator):
    """Greater Than Or Equal (≥) operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"{x0}.ge({x1})"


def f(arg):
    match arg:
        case tuple() as args:
            return parse(*args)
        case Bits() as b:
            return BitsConst(b)
        case Expr() as x:
            return x
        case _:
            raise ValueError("Invalid argument")


def parse(*args):
    """Return a symbolic expression."""
    match args:
        case [Op.NOT, x]:
            return Not(f(x))
        case [Op.OR, *xs]:
            return Or(*[f(x) for x in xs])
        case [Op.AND, *xs]:
            return And(*[f(x) for x in xs])
        case [Op.XOR, *xs]:
            return Xor(*[f(x) for x in xs])
        case [Op.GETITEM, Variable() as v, (int() | slice()) as key]:
            return GetItem(v, Constant(key))
        case [Op.GETATTR, Variable() as v, str() as name]:
            return GetAttr(v, Constant(name))
        case [Op.LT, x0, x1]:
            return LessThan(f(x0), f(x1))
        case [Op.LE, x0, x1]:
            return LessEqual(f(x0), f(x1))
        case [Op.EQ, x0, x1]:
            return Equal(f(x0), f(x1))
        case [Op.NE, x0, x1]:
            return NotEqual(f(x0), f(x1))
        case [Op.GT, x0, x1]:
            return GreaterThan(f(x0), f(x1))
        case [Op.GE, x0, x1]:
            return GreaterEqual(f(x0), f(x1))
        case _:
            raise ValueError("Invalid expression")
