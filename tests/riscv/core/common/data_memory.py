"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bit, Bits, Module, notify
from seqlogic.bits import T, cat, xes
from seqlogic.sim import always_comb, always_ff

WIDTH = 32
DEPTH = 32 * 1024


class DataMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """Write docstring."""
        # Ports
        self.addr = Bits(name="addr", parent=self, shape=(15,))

        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, shape=(WIDTH // 8,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(WIDTH,))

        self.rd_data = Bits(name="rd_data", parent=self, shape=(WIDTH,))

        self.clock = Bit(name="clock", parent=self)

        # State
        self.mem = Array(name="mem", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WIDTH,))

    @always_ff
    async def proc_wr_port(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.clock.posedge)
            if self.wr_en.value == T:
                i = self.addr.value.to_uint()
                parts = []
                for j in range(WIDTH // 8):
                    if self.wr_be.value[j] == T:
                        v = self.wr_data.value
                    else:
                        v = self.mem.get_value(i)
                    a, b = 8 * j, 8 * (j + 1)
                    parts.append(v[a:b])
                self.mem.set_next(i, cat(parts, flatten=True))

    @always_comb
    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.addr.changed, self.mem.changed)
            try:
                i = self.addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes((WIDTH,))
            else:
                self.rd_data.next = self.mem.get_next(i)
