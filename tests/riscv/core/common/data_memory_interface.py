"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, notify
from seqlogic.bits import cat, foo, rep, xes

from ..misc import COMBI


class DataMemoryInterface(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

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

        self.connect(self.bus_addr, self.addr)
        self.connect(self.bus_wr_en, self.wr_en)
        self.connect(self.bus_rd_en, self.rd_en)

        self._procs.add((self.proc_wr_be, COMBI))
        self._procs.add((self.proc_wr_data, COMBI))
        self._procs.add((self.proc_rd_data, COMBI))

    async def proc_wr_be(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.data_format.changed, self.addr.changed)
            if self.data_format.next[:2] == foo("2b00"):
                self.bus_wr_be.next = foo("4b0001") << self.addr.next[:2]
            elif self.data_format.next[:2] == foo("2b01"):
                self.bus_wr_be.next = foo("4b0011") << self.addr.next[:2]
            elif self.data_format.next[:2] == foo("2b10"):
                self.bus_wr_be.next = foo("4b1111") << self.addr.next[:2]
            elif self.data_format.next[:2] == foo("2b11"):
                self.bus_wr_be.next = foo("4b0000")
            else:
                self.bus_wr_be.next = xes((4,))

    async def proc_wr_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.addr.changed, self.wr_data.changed)
            n = cat([foo("3b000"), self.addr.next[:2]])
            self.bus_wr_data.next = self.wr_data.next << n

    async def proc_rd_data(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.data_format.changed, self.addr.changed, self.bus_rd_data.changed)
            n = cat([foo("3b000"), self.addr.next[:2]])
            temp = self.bus_rd_data.next >> n
            if self.data_format.next[:2] == foo("2b00"):
                x = ~(self.data_format.next[2]) & temp[7]
                self.rd_data.next = cat([temp[:8], rep(x, 24)], flatten=True)
            elif self.data_format.next[:2] == foo("2b01"):
                x = ~(self.data_format.next[2]) & temp[15]
                self.rd_data.next = cat([temp[:16], rep(x, 16)], flatten=True)
            elif self.data_format.next[:2] == foo("2b10"):
                self.rd_data.next = temp
            else:
                self.rd_data.next = xes((32,))
