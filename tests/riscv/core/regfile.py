"""Register File."""

import operator

from seqlogic import NE, Module, Vec, clog2, u2bv

N = 32
Addr = Vec[clog2(N)]
Data = Vec[32]


class RegFile(Module):
    """Register File."""

    def build(self):
        # Ports
        wr_en = self.input(name="wr_en", dtype=Vec[1])
        wr_addr = self.input(name="wr_addr", dtype=Addr)
        wr_data = self.input(name="wr_data", dtype=Data)

        rs1_addr = self.input(name="rs1_addr", dtype=Addr)
        rs1_data = self.output(name="rs1_data", dtype=Data)

        rs2_addr = self.input(name="rs2_addr", dtype=Addr)
        rs2_data = self.output(name="rs2_data", dtype=Data)

        clock = self.input(name="clock", dtype=Vec[1])

        # State
        regs = self.logic(name="regs", dtype=Data, shape=(N,))

        # Assign r0 to zero
        a0 = u2bv(0, Addr.size)
        self.assign(regs[a0], "32h0000_0000")

        en = self.logic(name="en", dtype=Vec[1])
        self.expr(en, wr_en & NE(wr_addr, a0))

        # Write Port
        self.mem_wr(regs, wr_addr, wr_data, clock, en)

        # Read Ports
        self.combi(rs1_data, operator.getitem, regs, rs1_addr)
        self.combi(rs2_data, operator.getitem, regs, rs2_addr)
