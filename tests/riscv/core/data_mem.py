"""Data Memory."""

# pyright: reportAttributeAccessIssue=false

import operator

from seqlogic import Module
from seqlogic.lbool import Vec


class DataMem(Module):
    """Data random access, read/write memory."""

    def __init__(self, name: str, parent: Module | None, word_addr_bits: int = 10):
        super().__init__(name, parent)

        # Ports
        addr = self.bits(name="addr", dtype=Vec[word_addr_bits], port=True)
        wr_en = self.bit(name="wr_en", port=True)
        wr_be = self.bits(name="wr_be", dtype=Vec[4], port=True)
        wr_data = self.bits(name="wr_data", dtype=Vec[32], port=True)
        rd_data = self.bits(name="rd_data", dtype=Vec[32], port=True)
        clock = self.bit(name="clock", port=True)

        # State
        mem = self.array(name="mem", dtype=Vec[32])

        # Write Port
        self.mem_wr_be(mem, addr, wr_data, wr_en, wr_be, clock, nbytes=4)

        # Read Port
        self.combi(rd_data, operator.getitem, mem, addr)
