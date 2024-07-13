"""Text Memory."""

import operator

from seqlogic import Module
from seqlogic import Vector as Vec


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
        rd_addr = self.input(name="rd_addr", dtype=Vec[word_addr_bits])
        rd_data = self.output(name="rd_data", dtype=Vec[32])

        # State
        mem = self.logic(name="mem", dtype=Vec[32], shape=(1024,))

        # Read Port
        self.combi(rd_data, operator.getitem, mem, rd_addr)
