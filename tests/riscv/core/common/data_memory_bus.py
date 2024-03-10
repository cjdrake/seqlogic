"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.bits import F, T, X, xes
from seqlogic.sim import always_comb

from .constants import DATA_BASE, DATA_SIZE
from .data_memory import DataMemory


class DataMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.addr = Bits(name="addr", parent=self, shape=(32,))

        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, shape=(4,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))

        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)

        # State
        self.data = Bits(name="data", parent=self, shape=(32,))
        self.is_data = Bit(name="is_data", parent=self)

        # Submodules
        self.data_memory = DataMemory("data_memory", parent=self)

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.data_memory.wr_be.connect(self.wr_be)
        self.data_memory.wr_data.connect(self.wr_data)
        self.data.connect(self.data_memory.rd_data)
        self.data_memory.clock.connect(self.clock)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.addr)
            try:
                addr = self.addr.next.to_uint()
            except ValueError:
                self.is_data.next = X
            else:
                if DATA_BASE <= addr < (DATA_BASE + DATA_SIZE):
                    self.is_data.next = T
                else:
                    self.is_data.next = F

    @always_comb
    async def p_c_1(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.wr_en, self.is_data)
            self.data_memory.wr_en.next = self.wr_en.next & self.is_data.next

    @always_comb
    async def p_c_2(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.addr)
            self.data_memory.addr.next = self.addr.next[2:17]

    @always_comb
    async def p_c_3(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.rd_en, self.is_data, self.data)
            if self.rd_en.next == T and self.is_data.next == T:
                self.rd_data.next = self.data.next
            else:
                self.rd_data.next = xes((32,))
