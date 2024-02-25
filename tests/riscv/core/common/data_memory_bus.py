"""TODO(cjdrake): Write docstring."""

from seqlogic import Module
from seqlogic.logicvec import F, T, X, xes
from seqlogic.sim import notify
from seqlogic.var import Logic, LogicVec

from ..misc import COMBI
from .constants import DATA_BASE, DATA_SIZE
from .data_memory import DataMemory


class DataMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        # Ports
        self.addr = LogicVec(name="addr", parent=self, shape=(32,))

        self.wr_en = Logic(name="wr_en", parent=self)
        self.wr_be = LogicVec(name="wr_be", parent=self, shape=(4,))
        self.wr_data = LogicVec(name="wr_data", parent=self, shape=(32,))

        self.rd_en = Logic(name="rd_en", parent=self)
        self.rd_data = LogicVec(name="rd_data", parent=self, shape=(32,))

        self.clock = Logic(name="clock", parent=self)

        # State
        self.data = LogicVec(name="data", parent=self, shape=(32,))
        self.is_data = Logic(name="is_data", parent=self)

        # Submodules
        self.data_memory = DataMemory("data_memory", parent=self)

        # Processes
        self.connect(self.data_memory.wr_be, self.wr_be)
        self.connect(self.data_memory.wr_data, self.wr_data)
        self.connect(self.data, self.data_memory.rd_data)
        self.connect(self.data_memory.clock, self.clock)

        self._procs.add((self.proc_is_data, COMBI))
        self._procs.add((self.proc_wr_en, COMBI))
        self._procs.add((self.proc_data_memory_addr, COMBI))
        self._procs.add((self.proc_rd_data, COMBI))

    async def proc_is_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.addr.changed)
            try:
                addr = self.addr.next.to_uint()
            except ValueError:
                self.is_data.next = X
            else:
                if DATA_BASE <= addr < (DATA_BASE + DATA_SIZE):
                    self.is_data.next = T
                else:
                    self.is_data.next = F

    async def proc_wr_en(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.wr_en.changed, self.is_data.changed)
            self.data_memory.wr_en.next = self.wr_en.next & self.is_data.next

    async def proc_data_memory_addr(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.addr.changed)
            self.data_memory.addr.next = self.addr.next[2:17]

    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.rd_en.changed, self.is_data.changed, self.data.changed)
            if self.rd_en.next == T and self.is_data.next == T:
                self.rd_data.next = self.data.next
            else:
                self.rd_data.next = xes((32,))
