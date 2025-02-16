"""Text Memory."""

import operator

from bvwx import Vec

from seqlogic import Module


class TextMem(Module):
    """Text (i.e. instruction) random access, read-only memory."""

    WORD_ADDR_BITS: int = 10

    def build(self):
        # Ports
        rd_addr = self.input(name="rd_addr", dtype=Vec[self.WORD_ADDR_BITS])
        rd_data = self.output(name="rd_data", dtype=Vec[32])

        # State
        mem = self.logic(name="mem", dtype=Vec[32], shape=(1024,))

        # Read Port
        self.combi(rd_data, operator.getitem, mem, rd_addr)
