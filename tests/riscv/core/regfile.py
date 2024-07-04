"""Register File."""

# pyright: reportArgumentType=false

import operator

from seqlogic import Module
from seqlogic.vec import Vec, uint2vec


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_addr = self.input(name="wr_addr", dtype=Vec[5])
        wr_data = self.input(name="wr_data", dtype=Vec[32])
        rs1_addr = self.input(name="rs1_addr", dtype=Vec[5])
        rs1_data = self.output(name="rs1_data", dtype=Vec[32])
        rs2_addr = self.input(name="rs2_addr", dtype=Vec[5])
        rs2_data = self.output(name="rs2_data", dtype=Vec[32])
        clock = self.input(name="clock", dtype=Vec[1])

        # State
        regs = self.array(name="regs", dtype=Vec[32])

        # Assign r0 to zero
        a0 = uint2vec(0, 5)
        self.assign(regs[a0], "32h0000_0000")

        en = self.bits(name="en", dtype=Vec[1])
        self.combi(en, lambda we, a: we & a.ne("5b0_0000"), wr_en, wr_addr)

        # Write Port
        self.mem_wr_en(regs, wr_addr, wr_data, en, clock)

        # Read Ports
        self.combi(rs1_data, operator.getitem, regs, rs1_addr)
        self.combi(rs2_data, operator.getitem, regs, rs2_addr)
