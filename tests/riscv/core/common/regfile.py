"""Register File."""

from seqlogic import Array, Bit, Bits, Module, changed, clog2, resume
from seqlogic.lbool import Vec, zeros
from seqlogic.sim import active, reactive

DEPTH = 32
WORD_BYTES = 4
BYTE_BITS = 8
WORD_BITS = WORD_BYTES * BYTE_BITS


class RegFile(Module):
    """Register File."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._addr_bits = clog2(DEPTH)
        self._build()

    def _build(self):
        # Ports
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_addr = Bits(name="wr_addr", parent=self, dtype=Vec[self._addr_bits])
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[WORD_BITS])
        self.rs1_addr = Bits(name="rs1_addr", parent=self, dtype=Vec[self._addr_bits])
        self.rs1_data = Bits(name="rs1_data", parent=self, dtype=Vec[WORD_BITS])
        self.rs2_addr = Bits(name="rs2_addr", parent=self, dtype=Vec[self._addr_bits])
        self.rs2_data = Bits(name="rs2_data", parent=self, dtype=Vec[WORD_BITS])
        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self._regs = Array(name="regs", parent=self, shape=(self._addr_bits,), dtype=Vec[WORD_BITS])

    @active
    async def p_wr_port(self):
        def f():
            return self.reset.is_neg() and self.clock.is_posedge() and self.wr_en.value == "1b1"

        while True:
            state = await resume((self.reset, self.reset.is_posedge), (self.clock, f))
            match state:
                case self.reset:
                    for i in range(DEPTH):
                        self._regs[i].next = zeros(WORD_BITS)
                case self.clock:
                    addr = self.wr_addr.value
                    # If wr_en=1, address must be known
                    assert not addr.has_unknown()
                    # Write to address zero has no effect
                    if addr != zeros(self._addr_bits):
                        self._regs[addr].next = self.wr_data.value
                case _:
                    assert False

    @reactive
    async def p_rd_port_1(self):
        while True:
            await changed(self.rs1_addr, self._regs)
            addr = self.rs1_addr.value
            self.rs1_data.next = self._regs[addr].value

    @reactive
    async def p_rd_port_2(self):
        while True:
            await changed(self.rs2_addr, self._regs)
            addr = self.rs2_addr.value
            self.rs2_data.next = self._regs[addr].value
