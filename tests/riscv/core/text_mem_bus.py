"""Text Memory Bus."""

from seqlogic import Module
from seqlogic import Vector as Vec
from seqlogic import clog2, uint2vec

from . import TEXT_BASE, TEXT_SIZE, Addr
from .text_mem import TextMem


def f_rd_data(is_text: Vec[1], text: Vec[32]) -> Vec[32]:
    match is_text:
        case "1b1":
            return text
        case _:
            return Vec[32].xprop(is_text)


class TextMemBus(Module):
    """Text Memory Bus."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)

        # Parameters
        word_addr_bits = clog2(depth)
        text_start = TEXT_BASE
        text_stop = TEXT_BASE + TEXT_SIZE

        # Ports
        rd_addr = self.input(name="rd_addr", dtype=Addr)
        rd_data = self.output(name="rd_data", dtype=Vec[32])

        # State
        is_text = self.logic(name="is_text", dtype=Vec[1])
        text = self.logic(name="text", dtype=Vec[32])

        # Submodules
        m, n = 2, 2 + word_addr_bits
        self.submod(
            name="text_mem",
            mod=TextMem,
            word_addr_bits=word_addr_bits,
        ).connect(
            rd_addr=(lambda a: a[m:n], rd_addr),
            rd_data=text,
        )

        # Combinational Logic
        def f_is_text(addr: Addr) -> Vec[1]:
            start = uint2vec(text_start, 32)
            stop = uint2vec(text_stop, 32)
            return start.le(addr) & addr.lt(stop)

        self.combi(is_text, f_is_text, rd_addr)
        self.combi(rd_data, f_rd_data, is_text, text)
