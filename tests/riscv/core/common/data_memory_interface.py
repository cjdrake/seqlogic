"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, cat, rep, vec
from seqlogic.sim import always_comb


class DataMemoryInterface(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        # Ports
        self.data_format = Bits(name="data_format", parent=self, dtype=Vec[3])

        self.addr = Bits(name="addr", parent=self, dtype=Vec[32])
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])
        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, dtype=Vec[32])

        self.bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[4])
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[32])
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[32])

    def connect(self):
        self.bus_addr.connect(self.addr)
        self.bus_wr_en.connect(self.wr_en)
        self.bus_rd_en.connect(self.rd_en)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.data_format, self.addr)
            if self.data_format.value[:2] == vec("2b00"):
                self.bus_wr_be.next = vec("4b0001") << self.addr.value[:2]
            elif self.data_format.value[:2] == vec("2b01"):
                self.bus_wr_be.next = vec("4b0011") << self.addr.value[:2]
            elif self.data_format.value[:2] == vec("2b10"):
                self.bus_wr_be.next = vec("4b1111") << self.addr.value[:2]
            elif self.data_format.value[:2] == vec("2b11"):
                self.bus_wr_be.next = vec("4b0000")
            else:
                self.bus_wr_be.next = Vec[4].dcs()

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.addr, self.wr_data)
            n = cat(vec("3b000"), self.addr.value[:2])
            self.bus_wr_data.next = self.wr_data.value << n

    @always_comb
    async def p_c_2(self):
        while True:
            await changed(self.data_format, self.addr, self.bus_rd_data)
            n = cat(vec("3b000"), self.addr.value[:2])
            temp = self.bus_rd_data.value >> n
            if self.data_format.value[:2] == vec("2b00"):
                x = ~(self.data_format.value[2]) & temp[7]
                self.rd_data.next = cat(temp[:8], rep(x, 24))
            elif self.data_format.value[:2] == vec("2b01"):
                x = ~(self.data_format.value[2]) & temp[15]
                self.rd_data.next = cat(temp[:16], rep(x, 16))
            elif self.data_format.value[:2] == vec("2b10"):
                self.rd_data.next = temp
            else:
                self.rd_data.next = Vec[32].dcs()
