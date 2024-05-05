"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed, clog2
from seqlogic.lbool import Vec, uint2vec
from seqlogic.sim import always_comb

from .. import TEXT_BASE, TEXT_SIZE
from .text_memory import TextMemory

ADDR_BITS = 32
WORD_BYTES = 4
BYTE_BITS = 8


class TextMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)
        self._depth = depth
        self._width = WORD_BYTES * BYTE_BITS
        self._word_addr_bits = clog2(depth)
        self._byte_addr_bits = clog2(WORD_BYTES)
        self._text_start = uint2vec(TEXT_BASE, ADDR_BITS)
        self._text_stop = uint2vec(TEXT_BASE + TEXT_SIZE, ADDR_BITS)
        self.build()
        self.connect()

    def build(self):
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, dtype=Vec[ADDR_BITS])
        self.rd_data = Bits(name="rd_data", parent=self, dtype=Vec[self._width])

        # Submodules
        self.text_memory = TextMemory(
            "text_memory",
            parent=self,
            word_addr_bits=self._word_addr_bits,
            byte_addr_bits=self._byte_addr_bits,
        )

        # State
        self._is_text = Bit(name="is_text", parent=self)
        self._text = Bits(name="text", parent=self, dtype=Vec[self._width])

    def connect(self):
        self._text.connect(self.text_memory.rd_data)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.rd_addr)
            start_lte_addr = self._text_start.lteu(self.rd_addr.value)
            addr_lt_stop = self.rd_addr.value.ltu(self._text_stop)
            self._is_text.next = start_lte_addr & addr_lt_stop

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self._is_text, self._text)
            self.rd_data.next = self._is_text.value.ite(self._text.value)

    @always_comb
    async def p_c_2(self):
        while True:
            await changed(self.rd_addr)
            m = self._byte_addr_bits
            n = self._byte_addr_bits + self._word_addr_bits
            self.text_memory.rd_addr.next = self.rd_addr.value[m:n]
