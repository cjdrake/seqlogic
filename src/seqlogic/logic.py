"""Logic Data Type."""

from __future__ import annotations

from enum import Enum

from . import pcn
from .pcn import PcItem


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

    N = pcn.NULL
    NULL = pcn.NULL

    F = pcn.ZERO
    ZERO = pcn.ZERO

    T = pcn.ONE
    ONE = pcn.ONE

    X = pcn.DC
    UNKNOWN = pcn.DC

    def __str__(self) -> str:
        return pcn.to_char[self.value]

    def __repr__(self) -> str:
        return self.__str__()

    def __invert__(self) -> logic:
        return self.lnot()

    def __or__(self, other: object) -> logic:
        return self.lor(other)

    def __ror__(self, other: object) -> logic:
        return self.lor(other)

    def __and__(self, other: object) -> logic:
        return self.land(other)

    def __rand__(self, other: object) -> logic:
        return self.land(other)

    def __xor__(self, other: object) -> logic:
        return self.lxor(other)

    def __rxor__(self, other: object) -> logic:
        return self.lxor(other)

    def _get_xs(self, other: object) -> tuple[PcItem, PcItem]:
        x0 = self.value
        match other:
            case logic():
                x1 = other.value
            case _:
                index = bool(other)
                x1 = pcn.from_int[index]
        return x0, x1

    def lnot(self) -> logic:
        """Return output of "lifted" NOT function."""
        x = self.value
        return logic(pcn.lnot(x))

    def lnor(self, other: object) -> logic:
        """Return output of "lifted" NOR function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.lnor(x0, x1))

    def lor(self, other: object) -> logic:
        """Return output of "lifted" OR function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.lor(x0, x1))

    def lnand(self, other: object) -> logic:
        """Return output of "lifted" NAND function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.lnand(x0, x1))

    def land(self, other: object) -> logic:
        """Return output of "lifted" AND function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.land(x0, x1))

    def lxnor(self, other: object) -> logic:
        """Return output of "lifted" XNOR function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.lxnor(x0, x1))

    def lxor(self, other: object) -> logic:
        """Return output of "lifted" XOR function."""
        x0, x1 = self._get_xs(other)
        return logic(pcn.lxor(x0, x1))

    def limplies(self, other: object) -> logic:
        """Return output of "lifted" IMPLIES function."""
        p, q = self._get_xs(other)
        return logic(pcn.limplies(p, q))
