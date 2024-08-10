"""Expression Symbolic Logic."""

from __future__ import annotations

from abc import ABC

from .bits import Bits


class Expr(ABC):
    """Symbolic expression."""

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, key: int | slice) -> GetItem:
        return GetItem(self, Key(key))

    def getattr(self, name: str) -> GetAttr:
        return GetAttr(self, Name(name))

    def __invert__(self) -> Not:
        return Not(self)

    def __or__(self, other: Expr) -> Or:
        return Or(self, other)

    def __and__(self, other: Expr) -> And:
        return And(self, other)

    def __xor__(self, other: Expr) -> Xor:
        return Xor(self, other)

    def lt(self, other: Expr | Bits) -> LessThan:
        if isinstance(other, Bits):
            return LessThan(self, Const(other))
        if isinstance(other, Expr):
            return LessThan(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def le(self, other: Expr | Bits) -> LessEqual:
        if isinstance(other, Bits):
            return LessEqual(self, Const(other))
        if isinstance(other, Expr):
            return LessEqual(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def eq(self, other: Expr | Bits) -> Equal:
        if isinstance(other, Bits):
            return Equal(self, Const(other))
        if isinstance(other, Expr):
            return Equal(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def ne(self, other: Expr | Bits) -> NotEqual:
        if isinstance(other, Bits):
            return NotEqual(self, Const(other))
        if isinstance(other, Expr):
            return NotEqual(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def gt(self, other: Expr | Bits) -> GreaterThan:
        if isinstance(other, Bits):
            return GreaterThan(self, Const(other))
        if isinstance(other, Expr):
            return GreaterThan(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def ge(self, other: Expr | Bits) -> GreaterEqual:
        if isinstance(other, Bits):
            return GreaterEqual(self, Const(other))
        if isinstance(other, Expr):
            return GreaterEqual(self, other)
        raise TypeError("Expected other to be Expr or Bits")

    def iter_vars(self):
        raise NotImplementedError()


class Atom(Expr):
    """Atomic expression (leaf) node."""


class Key(Atom):
    """GetItem operator key node."""

    def __init__(self, key: int | slice):
        self._x = key

    @property
    def x(self) -> int | slice:
        return self._x

    def iter_vars(self):
        yield from ()


class Name(Atom):
    """GetAttr operator name node."""

    def __init__(self, name: str):
        self._x = name

    @property
    def x(self) -> str:
        return self._x

    def iter_vars(self):
        yield from ()


class Const(Atom):
    """Const node."""

    def __init__(self, b: Bits):
        self._x = b

    def __str__(self) -> str:
        return f'"{self._x}"'

    @property
    def x(self) -> str:
        return self._x

    def iter_vars(self):
        yield from ()


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

    def __init__(self, x: Expr, key: Key):
        super().__init__(x, key)

    def __str__(self) -> str:
        x, key = self._xs
        match key.x:
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

    def __init__(self, x: Expr, name: Name):
        super().__init__(x, name)

    def __str__(self) -> str:
        x, name = self._xs
        return f"{x}.{name.x}"


class Not(Operator):
    """NOT operator node."""

    def __init__(self, x: Expr):
        super().__init__(x)

    def __str__(self) -> str:
        x = self._xs[0]
        return f"~{x}"


class Or(Operator):
    """OR operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"({x0} | {x1})"


class And(Operator):
    """AND operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"({x0} & {x1})"


class Xor(Operator):
    """XOR operator node."""

    def __init__(self, x0: Expr, x1: Expr):
        super().__init__(x0, x1)

    def __str__(self) -> str:
        x0, x1 = self._xs
        return f"({x0} ^ {x1})"


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
