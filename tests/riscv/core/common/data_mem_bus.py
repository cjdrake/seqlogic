"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed, clog2
from seqlogic.lbool import Vec, uint2vec
from seqlogic.sim import reactive

from .. import DATA_BASE, DATA_SIZE
from .data_mem import DataMem

ADDR_BITS = 32
WORD_BYTES = 4
BYTE_BITS = 8


class DataMemBus(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)
        self._depth = depth
        self._width = WORD_BYTES * BYTE_BITS
        self._word_addr_bits = clog2(WORD_BYTES)
        self._byte_addr_bits = clog2(WORD_BYTES)
        self._data_start = uint2vec(DATA_BASE, ADDR_BITS)
        self._data_stop = uint2vec(DATA_BASE + DATA_SIZE, ADDR_BITS)
        self.build()
        self.connect()

    def build(self):
        # Ports
        self.addr = Bits(name="addr", parent=self, dtype=Vec[ADDR_BITS])
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_be = Bits(name="wr_be", parent=self, dtype=Vec[WORD_BYTES])
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[self._width])
        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, dtype=Vec[self._width])
        self.clock = Bit(name="clock", parent=self)

        # Submodules
        self.data_memory = DataMem(
            "data_memory",
            parent=self,
            word_addr_bits=self._word_addr_bits,
            byte_addr_bits=self._byte_addr_bits,
        )

        # State
        self._is_data = Bit(name="is_data", parent=self)
        self._data = Bits(name="data", parent=self, dtype=Vec[self._width])

    def connect(self):
        self.data_memory.wr_be.connect(self.wr_be)
        self.data_memory.wr_data.connect(self.wr_data)
        self._data.connect(self.data_memory.rd_data)
        self.data_memory.clock.connect(self.clock)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.addr)
            start_lte_addr = self._data_start.lteu(self.addr.value)
            addr_lt_stop = self.addr.value.ltu(self._data_stop)
            self._is_data.next = start_lte_addr & addr_lt_stop

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.wr_en, self._is_data)
            self.data_memory.wr_en.next = self.wr_en.value & self._is_data.value

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.addr)
            m = self._byte_addr_bits
            n = self._byte_addr_bits + self._word_addr_bits
            self.data_memory.addr.next = self.addr.value[m:n]

    @reactive
    async def p_c_3(self):
        while True:
            await changed(self.rd_en, self._is_data, self._data)
            self.rd_data.next = self._is_data.value.ite(self._data.value)
