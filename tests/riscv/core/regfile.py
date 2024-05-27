"""Register File."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Module, changed, resume
from seqlogic.lbool import Vec
from seqlogic.sim import active, reactive


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        # Ports
        self.bit(name="wr_en", port=True)
        self.bits(name="wr_addr", dtype=Vec[5], port=True)
        self.bits(name="wr_data", dtype=Vec[32], port=True)
        self.bits(name="rs1_addr", dtype=Vec[5], port=True)
        self.bits(name="rs1_data", dtype=Vec[32], port=True)
        self.bits(name="rs2_addr", dtype=Vec[5], port=True)
        self.bits(name="rs2_data", dtype=Vec[32], port=True)
        self.bit(name="clock", port=True)
        self.bit(name="reset", port=True)

        # State
        self.array(name="regs", dtype=Vec[32])

    @active
    async def p_wr_port(self):
        def f():
            return self._reset.is_neg() and self._clock.is_posedge() and self._wr_en.value == "1b1"

        while True:
            state = await resume((self._reset, self._reset.is_posedge), (self._clock, f))
            if state is self._reset:
                for i in range(32):
                    self._regs[i].next = "32h0000_0000"
            elif state is self._clock:
                addr = self._wr_addr.value
                # If wr_en=1, address must be known
                assert not addr.has_unknown()
                # Write to address zero has no effect
                if addr != "5b0_0000":
                    self._regs[addr].next = self._wr_data.value
            else:
                assert False

    @reactive
    async def p_rd_port_1(self):
        while True:
            await changed(self._rs1_addr, self._regs)
            addr = self._rs1_addr.value
            self._rs1_data.next = self._regs[addr].value

    @reactive
    async def p_rd_port_2(self):
        while True:
            await changed(self._rs2_addr, self._regs)
            addr = self._rs2_addr.value
            self._rs2_data.next = self._regs[addr].value
