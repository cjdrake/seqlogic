"""Expression Symbolic Logic."""

from __future__ import annotations

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


class Expr:
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

    @property
    def support(self) -> set[Variable]:
        return set(self.iter_vars())


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


class Operator(Expr):
    """Variable node."""

    def __init__(self, *xs: Expr):
        self._xs = xs

    def iter_vars(self):
        for x in self._xs:
            yield from x.iter_vars()


class PrefixOp(Operator):
    """Prefix operator: f(x[0], x[1], ..., x[n-1])"""

    name = NotImplemented

    def __str__(self) -> str:
        s = ", ".join(str(x) for x in self._xs)
        return f"{self.name}({s})"


class GetItem(Operator):
    """GetItem operator node."""

    def __init__(self, x: Expr, key: Constant):
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


class GetAttr(Operator):
    """GetAttr operator node."""

    def __init__(self, v: Variable, name: Constant):
        super().__init__(v, name)

    def __str__(self) -> str:
        v = self._xs[0]
        name = self._xs[1].value
        return f"{v}.{name}"


class Not(PrefixOp):
    """NOT operator node."""

    name = "not_"

    def __init__(self, x: Expr):
        super().__init__(x)


class Or(PrefixOp):
    """OR operator node."""

    name = "or_"


class And(PrefixOp):
    """AND operator node."""

    name = "and_"


class Xor(PrefixOp):
    """XOR operator node."""

    name = "xor"


class LessThan(PrefixOp):
    """LessThan (<) operator node."""

    name = "lt"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


class LessEqual(PrefixOp):
    """Less Than Or Equal (≤) operator node."""

    name = "le"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


class Equal(PrefixOp):
    """Equal (==) operator node."""

    name = "eq"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


class NotEqual(PrefixOp):
    """NotEqual (!=) operator node."""

    name = "ne"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


class GreaterThan(PrefixOp):
    """GreaterThan (>) operator node."""

    name = "gt"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


class GreaterEqual(PrefixOp):
    """Greater Than Or Equal (≥) operator node."""

    name = "ge"

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)


def f(arg) -> Expr:
    match arg:
        case tuple() as args:
            return parse(*args)
        case Bits() as b:
            return BitsConst(b)
        case Expr() as x:
            return x
        case _:
            raise ValueError("Invalid argument")


def parse(*args) -> Expr:
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
        case [Op.GETITEM, Expr() as x, (int() | slice()) as key]:
            return GetItem(x, Constant(key))
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
