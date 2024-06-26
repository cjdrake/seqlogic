"""Sequential Logic."""

from .design import Array, Bit, Bits, Module, simify
from .sim import changed, get_loop, resume, sleep
from .util import clog2

__all__ = [
    # design
    "Array",
    "Bit",
    "Bits",
    "Module",
    "simify",
    # sim
    "changed",
    "get_loop",
    "resume",
    "sleep",
    # util
    "clog2",
]
