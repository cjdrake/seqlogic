"""Register File."""

from seqlogic import Array, Bit, Bits, Module, changed, resume
from seqlogic.lbool import Vec
from seqlogic.sim import active, reactive


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        wr_en = Bit(name="wr_en", parent=self)
        wr_addr = Bits(name="wr_addr", parent=self, dtype=Vec[5])
        wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])
        rs1_addr = Bits(name="rs1_addr", parent=self, dtype=Vec[5])
        rs1_data = Bits(name="rs1_data", parent=self, dtype=Vec[32])
        rs2_addr = Bits(name="rs2_addr", parent=self, dtype=Vec[5])
        rs2_data = Bits(name="rs2_data", parent=self, dtype=Vec[32])
        clock = Bit(name="clock", parent=self)
        reset = Bit(name="reset", parent=self)

        # State
        regs = Array(name="regs", parent=self, dtype=Vec[32])

        # TODO(cjdrake): Remove
        self.wr_en = wr_en
        self.wr_addr = wr_addr
        self.wr_data = wr_data
        self.rs1_addr = rs1_addr
        self.rs1_data = rs1_data
        self.rs2_addr = rs2_addr
        self.rs2_data = rs2_data
        self.clock = clock
        self.reset = reset
        self.regs = regs

    @active
    async def p_wr_port(self):
        def f():
            return self.reset.is_neg() and self.clock.is_posedge() and self.wr_en.value == "1b1"

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            if state is self.reset:
                for i in range(32):
                    self.regs[i].next = "32h0000_0000"
            elif state is self.clock:
                addr = self.wr_addr.value
                # If wr_en=1, address must be known
                assert not addr.has_unknown()
                # Write to address zero has no effect
                if addr != "5b0_0000":
                    self.regs[addr].next = self.wr_data.value
            else:
                assert False

    @reactive
    async def p_rd_port_1(self):
        while True:
            await changed(self.rs1_addr, self.regs)
            addr = self.rs1_addr.value
            self.rs1_data.next = self.regs[addr].value

    @reactive
    async def p_rd_port_2(self):
        while True:
            await changed(self.rs2_addr, self.regs)
            addr = self.rs2_addr.value
            self.rs2_data.next = self.regs[addr].value
