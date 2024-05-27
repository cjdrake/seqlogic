"""Data Memory Bus."""

from seqlogic import Bit, Bits, Module, changed, clog2
from seqlogic.lbool import Vec, uint2vec
from seqlogic.sim import reactive

from . import DATA_BASE, DATA_SIZE
from .data_mem import DataMem

ADDR_BITS = 32
WORD_BYTES = 4
BYTE_BITS = 8


class DataMemBus(Module):
    """Data Memory Bus."""

    def __init__(self, name: str, parent: Module | None, depth: int = 1024):
        super().__init__(name, parent)

        self._depth = depth

        width = WORD_BYTES * BYTE_BITS
        word_addr_bits = clog2(WORD_BYTES)
        byte_addr_bits = clog2(WORD_BYTES)
        data_start = uint2vec(DATA_BASE, ADDR_BITS)
        data_stop = uint2vec(DATA_BASE + DATA_SIZE, ADDR_BITS)

        # Ports
        addr = Bits(name="addr", parent=self, dtype=Vec[ADDR_BITS])
        wr_en = Bit(name="wr_en", parent=self)
        wr_be = Bits(name="wr_be", parent=self, dtype=Vec[WORD_BYTES])
        wr_data = Bits(name="wr_data", parent=self, dtype=Vec[width])
        rd_en = Bit(name="rd_en", parent=self)
        rd_data = Bits(name="rd_data", parent=self, dtype=Vec[width])
        clock = Bit(name="clock", parent=self)

        # State
        is_data = Bit(name="is_data", parent=self)
        data = Bits(name="data", parent=self, dtype=Vec[width])

        # Submodules
        data_mem = DataMem(
            "data_mem",
            parent=self,
            word_addr_bits=word_addr_bits,
            byte_addr_bits=byte_addr_bits,
        )
        self.connect(data_mem.wr_be, wr_be)
        self.connect(data_mem.wr_data, wr_data)
        self.connect(data, data_mem.rd_data)
        self.connect(data_mem.clock, clock)

        # TODO(cjdrake): Remove
        self.word_addr_bits = word_addr_bits
        self.byte_addr_bits = byte_addr_bits
        self.data_start = data_start
        self.data_stop = data_stop

        self.addr = addr
        self.wr_en = wr_en
        self.wr_be = wr_be
        self.wr_data = wr_data
        self.rd_en = rd_en
        self.rd_data = rd_data
        self.clock = clock

        self.is_data = is_data
        self.data = data

        self.data_mem = data_mem

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.addr)
            start_lte_addr = self.data_start.lteu(self.addr.value)
            addr_lt_stop = self.addr.value.ltu(self.data_stop)
            self.is_data.next = start_lte_addr & addr_lt_stop

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.wr_en, self.is_data)
            self.data_mem.wr_en.next = self.wr_en.value & self.is_data.value

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.addr)
            m = self.byte_addr_bits
            n = self.byte_addr_bits + self.word_addr_bits
            self.data_mem.addr.next = self.addr.value[m:n]

    @reactive
    async def p_c_3(self):
        while True:
            await changed(self.rd_en, self.is_data, self.data)
            sel = self.is_data.value
            match sel:
                case "1b1":
                    self.rd_data.next = self.data.value
                case _:
                    self.rd_data.xprop(sel)
