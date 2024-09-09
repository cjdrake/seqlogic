"""Expression Symbolic Logic."""

# pylint: disable=exec-used

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto

from .bits import Bits, add, and_, eq, ge, gt, ite, le, lt, ne, neg, not_, or_, sub, xor


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

    def to_func(self) -> tuple[Callable, list[Variable]]:
        vs = sorted(self.support, key=lambda v: v.name)
        args = ", ".join(v.name for v in vs)
        source = f"def f({args}):\n    return {self}\n"
        globals_ = {
            "not_": not_,
            "or_": or_,
            "and_": and_,
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

    def __init__(self, *objs: Expr | Bits):
        xs = []
        for obj in objs:
            match obj:
                case Expr():
                    xs.append(obj)
                case Bits():
                    xs.append(BitsConst(obj))
                case _:
                    assert False
        self._xs = tuple(xs)

    def iter_vars(self):
        for x in self._xs:
            yield from x.iter_vars()


class PrefixOp(Operator):
    """Prefix operator: f(x[0], x[1], ..., x[n-1])"""

    name = NotImplemented

    def __str__(self) -> str:
        s = ", ".join(str(x) for x in self._xs)
        return f"{self.name}({s})"


class UnaryOp(PrefixOp):
    """Unary operator: f(x)"""

    def __init__(self, x: Expr | Bits):
        super().__init__(x)


class BinaryOp(PrefixOp):
    """Binary operator: f(x0, x1)"""

    def __init__(self, x0: Expr | Bits, x1: Expr | Bits):
        super().__init__(x0, x1)


class TernaryOp(PrefixOp):
    """Ternary operator: f(x0, x1, x2)"""

    def __init__(self, x0: Expr | Bits, x1: Expr | Bits, x2: Expr | Bits):
        super().__init__(x0, x1, x2)


class Not(UnaryOp):
    """NOT operator node."""

    name = "not_"


class Or(PrefixOp):
    """OR operator node."""

    name = "or_"


class And(PrefixOp):
    """AND operator node."""

    name = "and_"


class Xor(PrefixOp):
    """XOR operator node."""

    name = "xor"


class IfThenElse(TernaryOp):
    """If-Then-Else operator node."""

    name = "ite"


class Add(BinaryOp):
    """ADD operator node."""

    name = "add"


class Sub(BinaryOp):
    """SUB operator node."""

    name = "sub"


class Neg(UnaryOp):
    """NEG operator node."""

    name = "neg"


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

    def __init__(self, v: Variable, obj: Constant | str):
        match obj:
            case Constant():
                name = obj
            case str():
                name = Constant(obj)
            case _:
                assert False
        super().__init__(v, name)

    def __str__(self) -> str:
        v = self._xs[0]
        name = self._xs[1].value
        return f"{v}.{name}"


class LessThan(BinaryOp):
    """LessThan (<) operator node."""

    name = "lt"


class LessEqual(BinaryOp):
    """Less Than Or Equal (≤) operator node."""

    name = "le"


class Equal(BinaryOp):
    """Equal (==) operator node."""

    name = "eq"


class NotEqual(BinaryOp):
    """NotEqual (!=) operator node."""

    name = "ne"


class GreaterThan(BinaryOp):
    """GreaterThan (>) operator node."""

    name = "gt"


class GreaterEqual(BinaryOp):
    """Greater Than Or Equal (≥) operator node."""

    name = "ge"


class Op(Enum):
    """Expression opcode."""

    # Bitwise
    NOT = auto()
    OR = auto()
    AND = auto()
    XOR = auto()
    ITE = auto()

    # Arithmetic
    ADD = auto()
    SUB = auto()
    NEG = auto()

    # Word
    GETITEM = auto()
    GETATTR = auto()

    # Predicate
    LT = auto()
    LE = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    GE = auto()


def f(arg) -> Expr:
    match arg:
        case [*xs]:
            return parse(*xs)
        case Bits() as x:
            return BitsConst(x)
        case Expr() as x:
            return x
        case _:
            raise ValueError("Invalid argument")


def parse(*args) -> Expr:
    """Return a symbolic expression."""
    match args:
        # Bitwise
        case [Op.NOT, x]:
            return Not(f(x))
        case [Op.OR, *xs]:
            return Or(*[f(x) for x in xs])
        case [Op.AND, *xs]:
            return And(*[f(x) for x in xs])
        case [Op.XOR, *xs]:
            return Xor(*[f(x) for x in xs])
        # Arithmetic
        case [Op.ADD, x0, x1]:
            return Add(f(x0), f(x1))
        case [Op.SUB, x0, x1]:
            return Sub(f(x0), f(x1))
        case [Op.NEG, x]:
            return Neg(f(x))
        # Word
        case [Op.GETITEM, Expr() as x, (int() | slice()) as key]:
            return GetItem(x, Constant(key))
        case [Op.GETATTR, Variable() as v, str() as name]:
            return GetAttr(v, Constant(name))
        # Predicate
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
        # Error
        case _:
            raise ValueError("Invalid expression")
