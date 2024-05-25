"""Data Memory Interface."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, cat, rep
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

        self._byte_addr = Bits(name="byte_addr", parent=self, dtype=Vec[2])

    def _connect(self):
        self.bus_addr.connect(self.addr)
        self.bus_wr_en.connect(self.wr_en)
        self.bus_rd_en.connect(self.rd_en)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.data_format, self._byte_addr)
            sel = self.data_format.value[:2]
            match sel:
                case "2b00":
                    self.bus_wr_be.next = "4b0001" << self._byte_addr.value
                case "2b01":
                    self.bus_wr_be.next = "4b0011" << self._byte_addr.value
                case "2b10":
                    self.bus_wr_be.next = "4b1111" << self._byte_addr.value
                case "2b11":
                    self.bus_wr_be.next = "4b0000"
                case _:
                    self.bus_wr_be.xprop(sel)

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.addr, self.wr_data, self._byte_addr)
            n = cat("3b000", self._byte_addr.value)
            self.bus_wr_data.next = self.wr_data.value << n

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.data_format, self.bus_rd_data, self._byte_addr)
            n = cat("3b000", self._byte_addr.value)
            data = self.bus_rd_data.value >> n
            sel = self.data_format.value[:2]
            match sel:
                case "2b00":
                    pad = ~(self.data_format.value[2]) & data[8 - 1]
                    self.rd_data.next = cat(data[:8], rep(pad, 24))
                case "2b01":
                    pad = ~(self.data_format.value[2]) & data[16 - 1]
                    self.rd_data.next = cat(data[:16], rep(pad, 16))
                case "2b10":
                    self.rd_data.next = data
                case _:
                    self.rd_data.xprop(sel)

    @reactive
    async def p_c_3(self):
        while True:
            await changed(self.addr)
            self._byte_addr.next = self.addr.value[:2]
