"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.lbool import xes
from seqlogic.sim import always_comb

from .. import WORD_BITS
from .constants import TEXT_BASE, TEXT_BITS, TEXT_SIZE
from .text_memory import TextMemory


class TextMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, shape=(32,))
        self.rd_data = Bits(name="rd_data", parent=self, shape=(WORD_BITS,))
        # State
        self._text = Bits(name="text", parent=self, shape=(WORD_BITS,))
        # Submodules
        self.text_memory = TextMemory("text_memory", parent=self)

    def connect(self):
        self._text.connect(self.text_memory.rd_data)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.rd_addr, self._text)
            addr = self.rd_addr.next.to_uint()
            is_text = TEXT_BASE <= addr < (TEXT_BASE + TEXT_SIZE)
            if is_text:
                self.rd_data.next = self._text.next
            else:
                self.rd_data.next = xes(WORD_BITS)

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.rd_addr)
            self.text_memory.rd_addr.next = self.rd_addr.next[2:TEXT_BITS]
