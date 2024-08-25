"""Sequential Logic."""

from .bits import (
    AddResult,
    Array,
    Bits,
    Empty,
    Enum,
    Scalar,
    Struct,
    Union,
    Vector,
    add,
    and_,
    bits,
    cat,
    i2bv,
    nand,
    nor,
    or_,
    rep,
    stack,
    sub,
    u2bv,
    xnor,
    xor,
)
from .design import Module, Packed, Unpacked
from .expr import Op
from .sim import Region, changed, get_loop, resume, sleep
from .util import clog2

# Alias Vector to Vec for brevity
Vec = Vector

__all__ = [
    # bits
    "AddResult",
    "Array",
    "Bits",
    "Empty",
    "Enum",
    "Scalar",
    "Struct",
    "Union",
    "Vec",
    "Vector",
    "add",
    "and_",
    "bits",
    "cat",
    "i2bv",
    "nand",
    "nor",
    "or_",
    "rep",
    "stack",
    "sub",
    "u2bv",
    "xnor",
    "xor",
    # design
    "Module",
    "Packed",
    "Unpacked",
    # expr
    "Op",
    # sim
    "Region",
    "changed",
    "get_loop",
    "resume",
    "sleep",
    # util
    "clog2",
]
