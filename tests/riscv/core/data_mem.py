"""Data Memory."""

import operator

from bvwx import Array, Vec

from seqlogic import Module


class DataMem(Module):
    """Data random access, read/write memory."""

    WORD_ADDR_BITS: int = 10

    def build(self):
        # Ports
        addr = self.input(name="addr", dtype=Vec[self.WORD_ADDR_BITS])
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_be = self.input(name="wr_be", dtype=Vec[4])
        wr_data = self.input(name="wr_data", dtype=Array[4, 8])
        rd_data = self.output(name="rd_data", dtype=Vec[32])
        clock = self.input(name="clock", dtype=Vec[1])

        # State
        mem = self.logic(name="mem", dtype=Array[4, 8], shape=(1024,))

        # Write Port
        self.mem_wr(mem, addr, wr_data, clock, wr_en, be=wr_be)

        # Read Port
        self.combi(rd_data, operator.getitem, mem, addr)
