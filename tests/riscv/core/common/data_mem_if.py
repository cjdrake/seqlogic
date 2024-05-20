"""Data Memory Interface."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, cat, rep, vec
from seqlogic.sim import reactive


class DataMemIf(Module):
    """Data Memory Interface."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._build()
        self._connect()

    def _build(self):
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

    def _connect(self):
        self.bus_addr.connect(self.addr)
        self.bus_wr_en.connect(self.wr_en)
        self.bus_rd_en.connect(self.rd_en)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.data_format, self.addr)
            match self.data_format.value[:2]:
                case "2b00":
                    self.bus_wr_be.next = vec("4b0001") << self.addr.value[:2]
                case "2b01":
                    self.bus_wr_be.next = vec("4b0011") << self.addr.value[:2]
                case "2b10":
                    self.bus_wr_be.next = vec("4b1111") << self.addr.value[:2]
                case "2b11":
                    self.bus_wr_be.next = vec("4b0000")
                case _:
                    self.bus_wr_be.next = Vec[4].dcs()

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.addr, self.wr_data)
            n = cat("3b000", self.addr.value[:2])
            self.bus_wr_data.next = self.wr_data.value << n

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.data_format, self.addr, self.bus_rd_data)

            n = cat("3b000", self.addr.value[:2])
            temp = self.bus_rd_data.value >> n

            match self.data_format.value[:2]:
                case "2b00":
                    x = ~(self.data_format.value[2]) & temp[8 - 1]
                    self.rd_data.next = cat(temp[:8], rep(x, 24))
                case "2b01":
                    x = ~(self.data_format.value[2]) & temp[16 - 1]
                    self.rd_data.next = cat(temp[:16], rep(x, 16))
                case "2b10":
                    self.rd_data.next = temp
                case _:
                    self.rd_data.next = Vec[32].dcs()
