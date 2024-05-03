"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import ones, xes, zeros
from seqlogic.sim import always_comb

from .. import DATA_BASE, DATA_SIZE
from .data_memory import DataMemory


class DataMemoryBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        # Ports
        self.addr = Bits(name="addr", parent=self, shape=(32,))

        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, shape=(4,))
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))

        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)

        # State
        self._data = Bits(name="data", parent=self, shape=(32,))
        self._is_data = Bit(name="is_data", parent=self)

        # Submodules
        self.data_memory = DataMemory("data_memory", parent=self)

    def connect(self):
        self.data_memory.wr_be.connect(self.wr_be)
        self.data_memory.wr_data.connect(self.wr_data)
        self._data.connect(self.data_memory.rd_data)
        self.data_memory.clock.connect(self.clock)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.addr)
            try:
                addr = self.addr.value.to_uint()
            except ValueError:
                self._is_data.next = xes(1)
            else:
                if DATA_BASE <= addr < (DATA_BASE + DATA_SIZE):
                    self._is_data.next = ones(1)
                else:
                    self._is_data.next = zeros(1)

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.wr_en, self._is_data)
            self.data_memory.wr_en.next = self.wr_en.value & self._is_data.value

    @always_comb
    async def p_c_2(self):
        while True:
            await changed(self.addr)
            self.data_memory.addr.next = self.addr.value[2:17]

    @always_comb
    async def p_c_3(self):
        while True:
            await changed(self.rd_en, self._is_data, self._data)
            if self.rd_en.value == ones(1) and self._is_data.value == ones(1):
                self.rd_data.next = self._data.value
            else:
                self.rd_data.next = xes(32)
