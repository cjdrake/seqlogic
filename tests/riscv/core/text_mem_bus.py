"""Text Memory Bus."""

from seqlogic import Module, Op, Vec, clog2, u2bv

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

    depth: int = 1024

    def build(self):
        # Parameters
        word_addr_bits = clog2(self.depth)
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
            word_addr_bits=word_addr_bits,
        ).connect(
            rd_addr=rd_addr[m:n],
            rd_data=text,
        )

        self.expr(is_text, (Op.AND, (Op.GE, rd_addr, text_start), (Op.LT, rd_addr, text_stop)))
        self.combi(rd_data, f_rd_data, is_text, text)
