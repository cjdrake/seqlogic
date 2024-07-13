"""Data Memory."""

import operator

from seqlogic import Array, Module
from seqlogic import Vector as Vec


class DataMem(Module):
    """Data random access, read/write memory."""

    def __init__(self, name: str, parent: Module | None, word_addr_bits: int = 10):
        super().__init__(name, parent)

        # Ports
        addr = self.input(name="addr", dtype=Vec[word_addr_bits])
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_be = self.input(name="wr_be", dtype=Vec[4])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rd_data = self.output(name="rd_data", dtype=Vec[32])
        clock = self.input(name="clock", dtype=Vec[1])

        # State
        mem = self.logic(name="mem", dtype=Array[4, 8], shape=(1024,))

        # Write Port
        self.mem_wr_be(mem, addr, wr_data, wr_en, wr_be, clock)

        # Read Port
        self.combi(rd_data, operator.getitem, mem, addr)
