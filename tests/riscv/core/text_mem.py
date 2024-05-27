"""Text Memory."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive


class TextMem(Module):
    """Text (i.e. instruction) random access, read-only memory."""

    def __init__(
        self,
        name: str,
        parent: Module | None,
        word_addr_bits: int = 10,
    ):
        super().__init__(name, parent)

        # Ports
        self.bits(name="rd_addr", dtype=Vec[word_addr_bits], port=True)
        self.bits(name="rd_data", dtype=Vec[32], port=True)

        # State
        self.array(name="mem", dtype=Vec[32])

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self._rd_addr, self._mem)
            addr = self._rd_addr.value
            self._rd_data.next = self._mem[addr].value
