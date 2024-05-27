"""Text Memory Bus."""

from seqlogic import Bit, Bits, Module, changed, clog2
from seqlogic.lbool import Vec, uint2vec
from seqlogic.sim import reactive

from . import TEXT_BASE, TEXT_SIZE
from .text_mem import TextMem

ADDR_BITS = 32
WORD_BYTES = 4
BYTE_BITS = 8


class TextMemBus(Module):
    """Text Memory Bus."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)

        width = WORD_BYTES * BYTE_BITS
        word_addr_bits = clog2(depth)
        byte_addr_bits = clog2(WORD_BYTES)
        text_start = uint2vec(TEXT_BASE, ADDR_BITS)
        text_stop = uint2vec(TEXT_BASE + TEXT_SIZE, ADDR_BITS)

        # Ports
        rd_addr = Bits(name="rd_addr", parent=self, dtype=Vec[ADDR_BITS])
        rd_data = Bits(name="rd_data", parent=self, dtype=Vec[width])

        # State
        is_text = Bit(name="is_text", parent=self)
        text = Bits(name="text", parent=self, dtype=Vec[width])

        # Submodules
        text_mem = TextMem(
            "text_mem",
            parent=self,
            word_addr_bits=word_addr_bits,
            byte_addr_bits=byte_addr_bits,
        )
        self.connect(text, text_mem.rd_data)

        # TODO(cjdrake): Remove
        self.word_addr_bits = word_addr_bits
        self.byte_addr_bits = byte_addr_bits
        self.text_start = text_start
        self.text_stop = text_stop
        self.rd_addr = rd_addr
        self.rd_data = rd_data
        self.is_text = is_text
        self.text = text
        self.text_mem = text_mem

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.rd_addr)
            start_lte_addr = self.text_start.lteu(self.rd_addr.value)
            addr_lt_stop = self.rd_addr.value.ltu(self.text_stop)
            self.is_text.next = start_lte_addr & addr_lt_stop

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.is_text, self.text)
            sel = self.is_text.value
            match sel:
                case "1b1":
                    self.rd_data.next = self.text.value
                case _:
                    self.rd_data.xprop(sel)

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.rd_addr)
            m = self.byte_addr_bits
            n = self.byte_addr_bits + self.word_addr_bits
            self.text_mem.rd_addr.next = self.rd_addr.value[m:n]
