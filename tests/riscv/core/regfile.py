"""Register File."""

import operator

from seqlogic import Module, Vec, clog2, uint2vec

N = 32
Addr = Vec[clog2(N)]
Data = Vec[32]


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

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
        a0 = uint2vec(0, Addr.size)
        self.assign(regs[a0], "32h0000_0000")

        en = self.logic(name="en", dtype=Vec[1])
        self.combi(en, lambda we, a: we & a.ne(a0), wr_en, wr_addr)

        # Write Port
        self.mem_wr_en(regs, wr_addr, wr_data, en, clock)

        # Read Ports
        self.combi(rs1_data, operator.getitem, regs, rs1_addr)
        self.combi(rs2_data, operator.getitem, regs, rs2_addr)
