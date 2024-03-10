"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bits, Module, notify
from seqlogic.bits import xes
from seqlogic.sim import always_comb

WIDTH = 32
DEPTH = 1024


class TextMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, shape=(14,))
        self.rd_data = Bits(name="rd_data", parent=self, shape=(WIDTH,))

        # State
        self.mem = Array(name="mem", parent=self, unpacked_shape=(DEPTH,), packed_shape=(WIDTH,))

    # output logic [31:0] rd_data
    @always_comb
    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed, self.mem.changed)
            try:
                i = self.rd_addr.next.to_uint()
            except ValueError:
                self.rd_data.next = xes((WIDTH,))
            else:
                self.rd_data.next = self.mem.get_next(i)
