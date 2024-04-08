"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, changed, clog2, resume
from seqlogic.lbool import ones, xes, zeros
from seqlogic.sim import always_comb, always_ff, initial

from .. import WORD_BITS

DEPTH = 32


class RegFile(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_addr = Bits(name="wr_addr", parent=self, shape=(clog2(DEPTH),))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(WORD_BITS,))
        self.rs1_addr = Bits(name="rs1_addr", parent=self, shape=(clog2(DEPTH),))
        self.rs1_data = Bits(name="rs1_data", parent=self, shape=(WORD_BITS,))
        self.rs2_addr = Bits(name="rs2_addr", parent=self, shape=(clog2(DEPTH),))
        self.rs2_data = Bits(name="rs2_data", parent=self, shape=(WORD_BITS,))
        self.clock = Bit(name="clock", parent=self)

        # State
        self._regs = Array(
            name="regs", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WORD_BITS,)
        )

    @initial
    async def p_i_0(self):
        """TODO(cjdrake): Write docstring."""
        self._regs.set_next(0, zeros(WORD_BITS))
        for i in range(1, DEPTH):
            self._regs.set_next(i, zeros(WORD_BITS))

    @always_ff
    async def p_f_0(self):
        """TODO(cjdrake): Write docstring."""

        def f():
            return self.clock.posedge() and self.wr_en.value == ones(1)

        while True:
            await resume((self.clock, f))
            i = self.wr_addr.value.to_uint()
            if i != 0:
                self._regs.set_next(i, self.wr_data.value)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.rs1_addr, self._regs)
            try:
                i = self.rs1_addr.next.to_uint()
            except ValueError:
                self.rs1_data.next = xes(WORD_BITS)
            else:
                self.rs1_data.next = self._regs.get_next(i)

    @always_comb
    async def p_c_1(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.rs2_addr, self._regs)
            try:
                i = self.rs2_addr.next.to_uint()
            except ValueError:
                self.rs2_data.next = xes(WORD_BITS)
            else:
                self.rs2_data.next = self._regs.get_next(i)
