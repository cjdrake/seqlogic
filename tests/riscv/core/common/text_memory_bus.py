"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, notify
from seqlogic.bits import xes
from seqlogic.sim import always_comb

from .constants import TEXT_BASE, TEXT_BITS, TEXT_SIZE
from .text_memory import TextMemory


class TextMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, shape=(32,))
        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))
        # State
        self.text = Bits(name="text", parent=self, shape=(32,))
        # Submodules
        self.text_memory = TextMemory("text_memory", parent=self)

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.text.connect(self.text_memory.rd_data)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed, self.text.changed)
            addr = self.rd_addr.next.to_uint()
            is_text = TEXT_BASE <= addr < (TEXT_BASE + TEXT_SIZE)
            if is_text:
                self.rd_data.next = self.text.next
            else:
                self.rd_data.next = xes((32,))

    @always_comb
    async def p_c_1(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed)
            self.text_memory.rd_addr.next = self.rd_addr.next[2:TEXT_BITS]
