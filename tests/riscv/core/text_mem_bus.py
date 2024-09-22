"""Text Memory Bus."""

from seqlogic import GE, LT, GetItem, Module, Mux, Vec, clog2, u2bv

from . import TEXT_BASE, TEXT_SIZE, Addr
from .text_mem import TextMem


class TextMemBus(Module):
    """Text Memory Bus."""

    DEPTH: int = 1024

    def build(self):
        # Parameters
        word_addr_bits = clog2(self.DEPTH)
        text_start = u2bv(TEXT_BASE, 32)
        text_stop = u2bv(TEXT_BASE + TEXT_SIZE, 32)

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
            WORD_ADDR_BITS=word_addr_bits,
        ).connect(
            rd_addr=GetItem(rd_addr, slice(m, n)),
            rd_data=text,
        )

        self.expr(is_text, GE(rd_addr, text_start) & LT(rd_addr, text_stop))
        self.expr(rd_data, Mux(is_text, x1=text))
