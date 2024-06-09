"""Text Memory Bus."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module, clog2
from seqlogic.vec import Vec, uint2vec

from . import TEXT_BASE, TEXT_SIZE
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
        rd_addr = self.bits(name="rd_addr", dtype=Vec[32], port=True)
        rd_data = self.bits(name="rd_data", dtype=Vec[32], port=True)

        # State
        is_text = self.bit(name="is_text")
        text = self.bits(name="text", dtype=Vec[32])

        # Submodules
        text_mem = self.submod(name="text_mem", mod=TextMem, word_addr_bits=word_addr_bits)
        self.assign(text, text_mem.rd_data)

        # Combinational Logic
        def f_is_text(addr: Vec[32]) -> Vec[1]:
            start = uint2vec(text_start, 32)
            stop = uint2vec(text_stop, 32)
            return start.lteu(addr) & addr.ltu(stop)

        self.combi(is_text, f_is_text, rd_addr)
        self.combi(rd_data, f_rd_data, is_text, text)
        m, n = 2, 2 + word_addr_bits
        self.combi(text_mem.rd_addr, lambda a: a[m:n], rd_addr)
