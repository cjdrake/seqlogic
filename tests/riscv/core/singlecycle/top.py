"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, sleep
from seqlogic.bits import bits
from seqlogic.sim import initial

from ..common.data_memory_bus import DataMemoryBus
from ..common.text_memory_bus import TextMemoryBus
from .core import Core


class Top(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent=None)

        self.build()
        self.connect()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, shape=(4,))
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, shape=(32,))
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, shape=(32,))

        self.pc = Bits(name="pc", parent=self, shape=(32,))
        self.inst = Bits(name="inst", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # Submodules
        self.text_memory_bus = TextMemoryBus(name="text_memory_bus", parent=self)
        self.data_memory_bus = DataMemoryBus(name="data_memory_bus", parent=self)
        self.core = Core(name="core", parent=self)

    def connect(self):
        """TODO(cjdrake): Write docstring."""
        self.text_memory_bus.rd_addr.connect(self.pc)
        self.inst.connect(self.text_memory_bus.rd_data)

        self.data_memory_bus.addr.connect(self.bus_addr)
        self.data_memory_bus.wr_en.connect(self.bus_wr_en)
        self.data_memory_bus.wr_be.connect(self.bus_wr_be)
        self.data_memory_bus.wr_data.connect(self.bus_wr_data)
        self.data_memory_bus.rd_en.connect(self.bus_rd_en)
        self.bus_rd_data.connect(self.data_memory_bus.rd_data)
        self.data_memory_bus.clock.connect(self.clock)

        self.bus_addr.connect(self.core.bus_addr)
        self.bus_wr_en.connect(self.core.bus_wr_en)
        self.bus_wr_be.connect(self.core.bus_wr_be)
        self.bus_wr_data.connect(self.core.bus_wr_data)
        self.bus_rd_en.connect(self.core.bus_rd_en)
        self.core.bus_rd_data.connect(self.bus_rd_data)
        self.pc.connect(self.core.pc)
        self.core.inst.connect(self.inst)
        self.core.clock.connect(self.clock)
        self.core.reset.connect(self.reset)

    @initial
    async def p_i_0(self):
        """TODO(cjdrake): Write docstring."""
        self.clock.next = bits("1b0")
        await sleep(1)
        while True:
            self.clock.next = ~(self.clock.value)
            await sleep(1)
            self.clock.next = ~(self.clock.value)
            await sleep(1)

    @initial
    async def p_i_1(self):
        """TODO(cjdrake): Write docstring."""
        self.reset.next = bits("1b0")
        await sleep(5)
        self.reset.next = ~(self.reset.value)
        await sleep(5)
        self.reset.next = ~(self.reset.value)
