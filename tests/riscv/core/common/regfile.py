"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, changed, clog2, resume
from seqlogic.lbool import ones, zeros
from seqlogic.sim import always_comb, always_ff, initial

DEPTH = 32
WORD_BYTES = 4
BYTE_BITS = 8
WORD_BITS = WORD_BYTES * BYTE_BITS


class RegFile(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._addr_bits = clog2(DEPTH)
        self.build()

    def build(self):
        # Ports
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_addr = Bits(name="wr_addr", parent=self, shape=(self._addr_bits,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(WORD_BITS,))
        self.rs1_addr = Bits(name="rs1_addr", parent=self, shape=(self._addr_bits,))
        self.rs1_data = Bits(name="rs1_data", parent=self, shape=(WORD_BITS,))
        self.rs2_addr = Bits(name="rs2_addr", parent=self, shape=(self._addr_bits,))
        self.rs2_data = Bits(name="rs2_data", parent=self, shape=(WORD_BITS,))
        self.clock = Bit(name="clock", parent=self)

        # State
        self._regs = Array(
            name="regs", parent=self, unpacked_shape=(self._addr_bits,), packed_shape=(WORD_BITS,)
        )

    @initial
    async def p_i_0(self):
        # Register zero is hard-coded to zero
        self._regs[0].next = zeros(WORD_BITS)

        # TODO(cjdrake): This should be reset
        for i in range(1, DEPTH):
            self._regs[i].next = zeros(WORD_BITS)

    @always_ff
    async def p_f_0(self):
        def f():
            return self.clock.is_posedge() and self.wr_en.value == ones(1)

        while True:
            await resume((self.clock, f))
            # TODO(cjdrake): If wr_en=1, address must be known
            addr = self.wr_addr.value
            if addr.has_unknown():
                pass
            elif addr.neq(zeros(self._addr_bits)):
                self._regs[addr].next = self.wr_data.value

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.rs1_addr, self._regs)
            addr = self.rs1_addr.value
            self.rs1_data.next = self._regs[addr].value

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.rs2_addr, self._regs)
            addr = self.rs2_addr.value
            self.rs2_data.next = self._regs[addr].value
