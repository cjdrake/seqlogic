"""TODO(cjdrake): Write docstring."""

from seqlogic import Array, Bits, Module
from seqlogic.logicvec import xes
from seqlogic.sim import notify

from ..misc import COMBI

WIDTH = 32
DEPTH = 1024


class TextMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.rd_addr = Bits(name="rd_addr", parent=self, shape=(14,))
        self.rd_data = Bits(name="rd_data", parent=self, shape=(WIDTH,))

        # State
        self.mem = Array(name="mem", parent=self, packed_shape=(WIDTH,), unpacked_shape=(DEPTH,))

        # Processes
        self._procs.add((self.proc_rd_data, COMBI))

    # output logic [31:0] rd_data
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
