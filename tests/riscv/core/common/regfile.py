"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, notify
from seqlogic.bits import T, xes, zeros
from seqlogic.sim import always_comb, always_ff, initial

WIDTH = 32
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
        self.wr_addr = Bits(name="wr_addr", parent=self, shape=(5,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(WIDTH,))
        self.rs1_addr = Bits(name="rs1_addr", parent=self, shape=(5,))
        self.rs1_data = Bits(name="rs1_data", parent=self, shape=(WIDTH,))
        self.rs2_addr = Bits(name="rs2_addr", parent=self, shape=(5,))
        self.rs2_data = Bits(name="rs2_data", parent=self, shape=(WIDTH,))
        self.clock = Bit(name="clock", parent=self)
        # State
        self.regs = Array(name="regs", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WIDTH,))

    @initial
    async def proc_init(self):
        """TODO(cjdrake): Write docstring."""
        self.regs.set_next(0, zeros((WIDTH,)))
        for i in range(1, DEPTH):
            self.regs.set_next(i, zeros((WIDTH,)))

    @always_ff
    async def proc_wr_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.clock.posedge)
            if self.wr_en.value == T:
                i = self.wr_addr.value.to_uint()
                if i != 0:
                    self.regs.set_next(i, self.wr_data.value)

    @always_comb
    async def proc_rd1_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rs1_addr.changed, self.regs.changed)
            try:
                i = self.rs1_addr.next.to_uint()
            except ValueError:
                self.rs1_data.next = xes((WIDTH,))
            else:
                self.rs1_data.next = self.regs.get_next(i)

    @always_comb
    async def proc_rd2_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rs2_addr.changed, self.regs.changed)
            try:
                i = self.rs2_addr.next.to_uint()
            except ValueError:
                self.rs2_data.next = xes((WIDTH,))
            else:
                self.rs2_data.next = self.regs.get_next(i)
