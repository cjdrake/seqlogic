"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.bits import bits, cat, rep, xes
from seqlogic.sim import always_comb


class DataMemoryInterface(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.data_format = Bits(name="data_format", parent=self, shape=(3,))

        self.addr = Bits(name="addr", parent=self, shape=(32,))
        self.wr_en = Bit(name="wr_en", parent=self)
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))
        self.rd_en = Bit(name="rd_en", parent=self)
        self.rd_data = Bits(name="rd_data", parent=self, shape=(32,))

        self.bus_addr = Bits(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, shape=(32,))

        # State
        self.position_fix = Bits(name="position_fix", parent=self, shape=(32,))
        self.sign_fix = Bits(name="sign_fix", parent=self, shape=(32,))

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.bus_addr.connect(self.addr)
        self.bus_wr_en.connect(self.wr_en)
        self.bus_rd_en.connect(self.rd_en)

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.data_format, self.addr)
            if self.data_format.next[:2] == bits("2b00"):
                self.bus_wr_be.next = bits("4b0001") << self.addr.next[:2]
            elif self.data_format.next[:2] == bits("2b01"):
                self.bus_wr_be.next = bits("4b0011") << self.addr.next[:2]
            elif self.data_format.next[:2] == bits("2b10"):
                self.bus_wr_be.next = bits("4b1111") << self.addr.next[:2]
            elif self.data_format.next[:2] == bits("2b11"):
                self.bus_wr_be.next = bits("4b0000")
            else:
                self.bus_wr_be.next = xes((4,))

    @always_comb
    async def p_c_1(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.addr, self.wr_data)
            n = cat([bits("3b000"), self.addr.next[:2]])
            self.bus_wr_data.next = self.wr_data.next << n

    @always_comb
    async def p_c_2(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.data_format, self.addr, self.bus_rd_data)
            n = cat([bits("3b000"), self.addr.next[:2]])
            temp = self.bus_rd_data.next >> n
            if self.data_format.next[:2] == bits("2b00"):
                x = ~(self.data_format.next[2]) & temp[7]
                self.rd_data.next = cat([temp[:8], rep(x, 24)], flatten=True)
            elif self.data_format.next[:2] == bits("2b01"):
                x = ~(self.data_format.next[2]) & temp[15]
                self.rd_data.next = cat([temp[:16], rep(x, 16)], flatten=True)
            elif self.data_format.next[:2] == bits("2b10"):
                self.rd_data.next = temp
            else:
                self.rd_data.next = xes((32,))
