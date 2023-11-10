"""
Logic Data Type
"""

from enum import Enum
from typing import Self

# Positional Cube (PC) notation
_PC_ZERO = 0b01
_PC_ONE = 0b10


class logic(Enum):
    """
    Logic data type that includes {Null, 0, 1, X}.

    Use a representation known as "positional cube notation", or PCN.
    As far as we know, this term was first used in
    "Logic Minimization Algorithms for VLSI Synthesis", by Brayton et al.

    Null means neither 0 nor 1. This is an illogical or metastable value that
    always dominates other values.

    X means either 0 or 1. This is an unknown value that may either dominate or
    be dominated by other values, depending on the operation.
    In particular:
        1 | X = 1
        0 | X = X
        1 & X = X
        0 & X = 0
        0 ^ X = X
        1 ^ X = X
    """

    N = _PC_ZERO & _PC_ONE
    NULL = N

    F = _PC_ZERO
    ZERO = F

    T = _PC_ONE
    ONE = T

    X = _PC_ZERO | _PC_ONE
    UNKNOWN = X

    def __str__(self) -> str:
        return _logic2char[self]

    def __repr__(self) -> str:
        return self.__str__()

    def __invert__(self) -> Self:
        return self.not_()

    def __or__(self, other: object) -> Self:
        return self.or_(other)

    def __ror__(self, other: object) -> Self:
        return self.or_(other)

    def __and__(self, other: object) -> Self:
        return self.and_(other)

    def __rand__(self, other: object) -> Self:
        return self.and_(other)

    def __xor__(self, other: object) -> Self:
        return self.xor(other)

    def __rxor__(self, other: object) -> Self:
        return self.xor(other)

    def not_(self) -> Self:
        """Return output of NOT function

        f(x) -> y:
            N => N | 00 => 00
            0 => 1 | 01 => 10
            1 => 0 | 10 => 01
            X => X | 11 => 11
        """
        x: int = self.value
        x_0 = x & 1
        x_1 = (x >> 1) & 1

        y_0, y_1 = x_1, x_0
        y = (y_1 << 1) | y_0

        return logic(y)

    def _get_xs(self, other: object) -> tuple[int, int]:
        x0: int = self.value
        match other:
            case logic():
                x1: int = other.value
            case _:
                index = bool(other)
                x1 = (_PC_ZERO, _PC_ONE)[index]
        return x0, x1

    def nor(self, other: object) -> Self:
        """Return output of NOR function

        f(x0, x1) -> y:
            0 0 => 0
            0 1 => 1
            1 0 => 1
            1 1 => 1
            N X => N
            X 0 => X
            1 X => 1

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
              +--+--+--+--+
           01 |00|01|11|10|
              +--+--+--+--+
           11 |00|11|11|10|  y0 = x0[0] & x1[1]
              +--+--+--+--+     | x0[1] & x1[0]
           10 |00|10|10|10|     | x0[1] & x1[1]
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_0 & x1_1 | x0_1 & x1_0 | x0_1 & x1_1
        y_1 = x0_0 & x1_0
        y = (y_1 << 1) | y_0

        return logic(y)

    def or_(self, other: object) -> Self:
        """Return output of OR function

        f(x0, x1) -> y:
            0 0 => 0
            0 1 => 1
            1 0 => 1
            1 1 => 1
            N X => N
            X 0 => X
            1 X => 1

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
              +--+--+--+--+     | x0[1] & x1[0]
           01 |00|01|11|10|     | x0[1] & x1[1]
              +--+--+--+--+
           11 |00|11|11|10|  y0 = x0[0] & x1[0]
              +--+--+--+--+
           10 |00|10|10|10|
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_0 & x1_0
        y_1 = x0_0 & x1_1 | x0_1 & x1_0 | x0_1 & x1_1
        y = (y_1 << 1) | y_0

        return logic(y)

    def nand(self, other: object) -> Self:
        """Return output of NAND function

        f(x0, x1) -> y:
            0 0 => 1
            0 1 => 1
            1 0 => 1
            1 1 => 0
            N X => N
            0 X => 1
            X 1 => X

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
              +--+--+--+--+     | x0[0] & x1[1]
           01 |00|01|01|01|     | x0[1] & x1[0]
              +--+--+--+--+
           11 |00|01|11|11|  y0 = x0[1] & x1[1]
              +--+--+--+--+
           10 |00|01|11|10|
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_1 & x1_1
        y_1 = x0_0 & x1_0 | x0_0 & x1_1 | x0_1 & x1_0
        y = (y_1 << 1) | y_0

        return logic(y)

    def and_(self, other: object) -> Self:
        """Return output of AND function

        f(x0, x1) -> y:
            0 0 => 0
            0 1 => 0
            1 0 => 0
            1 1 => 1
            N X => N
            0 X => 0
            X 1 => X

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[1] & x1[1]
              +--+--+--+--+
           01 |00|01|01|01|
              +--+--+--+--+
           11 |00|01|11|11|  y0 = x0[0] & x1[0]
              +--+--+--+--+     | x0[0] & x1[1]
           10 |00|01|11|10|     | x0[1] & x1[0]
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_0 & x1_0 | x0_0 & x1_1 | x0_1 & x1_0
        y_1 = x0_1 & x1_1
        y = (y_1 << 1) | y_0

        return logic(y)

    def xnor(self, other: object) -> Self:
        """Return output of XNOR function

        f(x0, x1) -> y:
            0 0 => 1
            0 1 => 0
            1 0 => 0
            1 1 => 1
            N X => N
            X 0 => X
            X 1 => X

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[0] & x1[0]
              +--+--+--+--+     | x0[1] & x1[1]
           01 |00|10|11|01|
              +--+--+--+--+
           11 |00|11|11|11|  y0 = x0[0] & x1[1]
              +--+--+--+--+     | x0[1] & x1[0]
           10 |00|01|11|10|
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_0 & x1_1 | x0_1 & x1_0
        y_1 = x0_0 & x1_0 | x0_1 & x1_1
        y = (y_1 << 1) | y_0

        return logic(y)

    def xor(self, other: object) -> Self:
        """Return output of XOR function

        f(x0, x1) -> y:
            0 0 => 0
            0 1 => 1
            1 0 => 1
            1 1 => 0
            N X => N
            X 0 => X
            X 1 => X

               x1
               00 01 11 10
              +--+--+--+--+
        x0 00 |00|00|00|00|  y1 = x0[0] & x1[1]
              +--+--+--+--+     | x0[1] & x1[0]
           01 |00|01|11|10|
              +--+--+--+--+
           11 |00|11|11|11|  y0 = x0[0] & x1[0]
              +--+--+--+--+     | x0[1] & x1[1]
           10 |00|10|11|01|
              +--+--+--+--+
        """
        x0, x1 = self._get_xs(other)
        x0_0 = x0 & 1
        x0_1 = (x0 >> 1) & 1
        x1_0 = x1 & 1
        x1_1 = (x1 >> 1) & 1

        y_0 = x0_0 & x1_0 | x0_1 & x1_1
        y_1 = x0_0 & x1_1 | x0_1 & x1_0
        y = (y_1 << 1) | y_0

        return logic(y)

    def implies(self, other: object) -> Self:
        """Return output of IMPLIES function

        f(p, q) -> y:
            0 0 => 1
            0 1 => 1
            1 0 => 0
            1 1 => 1
            N X => N
            0 X => 1
            1 X => X
            X 0 => X
            X 1 => 1
            X X => X

               q
               00 01 11 10
              +--+--+--+--+
         p 00 |00|00|00|00|  y1 = p[0] & q[0]
              +--+--+--+--+     | p[0] & q[1]
           01 |00|10|10|10|     | p[1] & q[1]
              +--+--+--+--+
           11 |00|11|11|10|  y0 = p[1] & q[0]
              +--+--+--+--+
           10 |00|01|11|10|
              +--+--+--+--+
        """
        p, q = self._get_xs(other)
        p_0 = p & 1
        p_1 = (p >> 1) & 1
        q_0 = q & 1
        q_1 = (q >> 1) & 1

        y_0 = p_1 & q_0
        y_1 = p_0 & q_0 | p_0 & q_1 | p_1 & q_1
        y = (y_1 << 1) | y_0

        return logic(y)


_logic2char = {
    logic.N: "X",
    logic.F: "0",
    logic.T: "1",
    logic.X: "x",
}


_char2logic = {
    "X": logic.N,
    "0": logic.F,
    "1": logic.T,
    "x": logic.X,
}


_int2logic = (logic.F, logic.T)
