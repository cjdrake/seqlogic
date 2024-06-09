"""Register File."""

import operator

from seqlogic import Module
from seqlogic.vec import Vec, uint2vec


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        wr_en = self.bit(name="wr_en", port=True)
        wr_addr = self.bits(name="wr_addr", dtype=Vec[5], port=True)
        wr_data = self.bits(name="wr_data", dtype=Vec[32], port=True)
        rs1_addr = self.bits(name="rs1_addr", dtype=Vec[5], port=True)
        rs1_data = self.bits(name="rs1_data", dtype=Vec[32], port=True)
        rs2_addr = self.bits(name="rs2_addr", dtype=Vec[5], port=True)
        rs2_data = self.bits(name="rs2_data", dtype=Vec[32], port=True)
        clock = self.bit(name="clock", port=True)

        # State
        regs = self.array(name="regs", dtype=Vec[32])

        # Assign r0 to zero
        a0 = uint2vec(0, 5)
        self.assign(regs[a0], "32h0000_0000")

        en = self.bit(name="en")
        self.combi(en, lambda we, a: we & a.neq("5b0_0000"), wr_en, wr_addr)

        # Write Port
        self.mem_wr_en(regs, wr_addr, wr_data, en, clock)

        # Read Ports
        self.combi(rs1_data, operator.getitem, regs, rs1_addr)
        self.combi(rs2_data, operator.getitem, regs, rs2_addr)
