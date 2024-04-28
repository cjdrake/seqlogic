"""Sequential Logic."""

from .design import Array, Bit, Bits, Enum, Module, Struct, simify
from .sim import changed, get_loop, resume, sleep
from .util import clog2

__all__ = [
    # design
    "Array",
    "Bit",
    "Bits",
    "Enum",
    "Module",
    "Struct",
    "simify",
    # sim
    "changed",
    "get_loop",
    "resume",
    "sleep",
    # util
    "clog2",
]
