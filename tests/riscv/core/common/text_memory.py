"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.hier import Dict
from seqlogic.sim import notify
from seqlogic.var import LogicVec

from ..misc import COMBI

NUM = 1024


class TextMemory(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.rd_addr = LogicVec(name="rd_addr", parent=self, shape=(14,))
        self.rd_data = LogicVec(name="rd_data", parent=self, shape=(32,))

        # State
        self.mem = Dict(name="mem", parent=self)
        for i in range(NUM):
            self.mem[i] = LogicVec(name=str(i), parent=self.mem, shape=(32,))

        # Processes
        self._procs.add((self.proc_rd_data, COMBI))

    # output logic [31:0] rd_data
    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_addr.changed)
            i = self.rd_addr.next.to_uint()
            self.rd_data.next = self.mem[i].value
