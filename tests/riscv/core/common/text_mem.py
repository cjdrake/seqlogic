"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

BYTE_BITS = 8


class TextMem(Module):
    """Text (i.e. instruction) random access, read-only memory."""

    def __init__(
        self,
        name: str,
        parent: Module | None,
        word_addr_bits: int = 10,
        byte_addr_bits: int = 2,
    ):
        super().__init__(name, parent)
        self._word_addr_bits = word_addr_bits
        self._byte_addr_bits = byte_addr_bits
        self._depth = 2**self._word_addr_bits
        self._width = 2**self._byte_addr_bits * BYTE_BITS
        self.build()

    def build(self):
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, dtype=Vec[self._word_addr_bits])
        self.rd_data = Bits(name="rd_data", parent=self, dtype=Vec[self._width])

        # State
        self._mem = Array(
            name="mem",
            parent=self,
            shape=(self._depth,),
            dtype=Vec[self._width],
        )

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.rd_addr, self._mem)
            addr = self.rd_addr.value
            self.rd_data.next = self._mem[addr].value
