"""Sequential Logic."""

from ._design import (
    AssertError,
    AssumeError,
    DesignError,
    Float,
    Logic,
    Module,
    Packed,
    Unpacked,
)
from ._expr import (
    EQ,
    GE,
    GT,
    ITE,
    LE,
    LT,
    NE,
    Adc,
    Add,
    And,
    BitsConst,
    Cat,
    Const,
    Div,
    Expr,
    GetAttr,
    GetItem,
    Impl,
    IntConst,
    Land,
    Lor,
    Lrot,
    Lsh,
    Lxor,
    Mod,
    Mul,
    Mux,
    Neg,
    Ngc,
    Not,
    Or,
    Rep,
    Rrot,
    Rsh,
    Sbc,
    Srsh,
    Sub,
    Sxt,
    Uand,
    Uor,
    Uxor,
    Variable,
    Xor,
    Xt,
)

__all__ = [
    # design
    "DesignError",
    "AssumeError",
    "AssertError",
    "Module",
    "Logic",
    "Packed",
    "Unpacked",
    "Float",
    # expr
    "Expr",
    "Const",
    "IntConst",
    "BitsConst",
    "Variable",
    # expr.bitwise
    "Not",
    "Or",
    "And",
    "Xor",
    "Impl",
    "ITE",
    "Mux",
    # expr.logical
    "Lor",
    "Land",
    "Lxor",
    # expr.unary
    "Uor",
    "Uand",
    "Uxor",
    # expr.arithmetic
    "Add",
    "Adc",
    "Sub",
    "Sbc",
    "Neg",
    "Ngc",
    "Mul",
    "Div",
    "Mod",
    "Lsh",
    "Rsh",
    "Srsh",
    # expr.word
    "Xt",
    "Sxt",
    "Lrot",
    "Rrot",
    "Cat",
    "Rep",
    # expr.predicate
    "EQ",
    "NE",
    "GT",
    "GE",
    "LT",
    "LE",
    # expr.getters
    "GetItem",
    "GetAttr",
]
