"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, sleep
from seqlogic.logicvec import vec

from ..common.data_memory_bus import DataMemoryBus
from ..common.text_memory_bus import TextMemoryBus
from ..misc import TASK
from .core import Core


class Top(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent=None)

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
        self.connect(self.text_memory_bus.rd_addr, self.pc)
        self.connect(self.inst, self.text_memory_bus.rd_data)

        self.data_memory_bus = DataMemoryBus(name="data_memory_bus", parent=self)
        self.connect(self.data_memory_bus.addr, self.bus_addr)
        self.connect(self.data_memory_bus.wr_en, self.bus_wr_en)
        self.connect(self.data_memory_bus.wr_be, self.bus_wr_be)
        self.connect(self.data_memory_bus.wr_data, self.bus_wr_data)
        self.connect(self.data_memory_bus.rd_en, self.bus_rd_en)
        self.connect(self.bus_rd_data, self.data_memory_bus.rd_data)
        self.connect(self.data_memory_bus.clock, self.clock)

        self.core = Core(name="core", parent=self)
        self.connect(self.bus_addr, self.core.bus_addr)
        self.connect(self.bus_wr_en, self.core.bus_wr_en)
        self.connect(self.bus_wr_be, self.core.bus_wr_be)
        self.connect(self.bus_wr_data, self.core.bus_wr_data)
        self.connect(self.bus_rd_en, self.core.bus_rd_en)
        self.connect(self.core.bus_rd_data, self.bus_rd_data)
        self.connect(self.pc, self.core.pc)
        self.connect(self.core.inst, self.inst)
        self.connect(self.core.clock, self.clock)
        self.connect(self.core.reset, self.reset)

        # Processes
        self._procs.add((self.proc_clock, TASK))
        self._procs.add((self.proc_reset, TASK))

    # input logic clock
    async def proc_clock(self):
        """TODO(cjdrake): Write docstring."""
        self.clock.next = vec("1b0")
        await sleep(1)
        while True:
            self.clock.next = ~(self.clock.next)
            await sleep(1)
            self.clock.next = ~(self.clock.next)
            await sleep(1)

    # input logic reset
    async def proc_reset(self):
        """TODO(cjdrake): Write docstring."""
        self.reset.next = vec("1b0")
        await sleep(5)
        self.reset.next = ~(self.reset.next)
        await sleep(5)
        self.reset.next = ~(self.reset.next)
