"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, sleep
from seqlogic.lbool import vec
from seqlogic.sim import initial

from .. import WORD_BITS, WORD_BYTES
from ..common.data_memory_bus import DataMemoryBus
from ..common.text_memory_bus import TextMemoryBus
from .core import Core

_CLOCK_PHASE_SHIFT = 1
_CLOCK_PHASE1 = 1
_CLOCK_PHASE2 = 1
_RESET_PHASE1 = 5
_RESET_PHASE2 = 5


class Top(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)

        self.build()
        self.connect()

    def build(self):
        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, shape=(32,))
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, shape=(WORD_BYTES,))
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, shape=(WORD_BITS,))
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, shape=(WORD_BITS,))

        self._pc = Bits(name="pc", parent=self, shape=(32,))
        self._inst = Bits(name="inst", parent=self, shape=(WORD_BITS,))

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # Submodules
        self.text_memory_bus = TextMemoryBus(name="text_memory_bus", parent=self)
        self.data_memory_bus = DataMemoryBus(name="data_memory_bus", parent=self)
        self.core = Core(name="core", parent=self)

    def connect(self):
        self.text_memory_bus.rd_addr.connect(self._pc)
        self._inst.connect(self.text_memory_bus.rd_data)

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
        self._pc.connect(self.core.pc)
        self.core.inst.connect(self._inst)
        self.core.clock.connect(self.clock)
        self.core.reset.connect(self.reset)

    @initial
    async def p_i_0(self):
        self.clock.next = vec("1b0")
        await sleep(_CLOCK_PHASE_SHIFT)
        while True:
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE1)
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE2)

    @initial
    async def p_i_1(self):
        self.reset.next = vec("1b0")
        await sleep(_RESET_PHASE1)
        self.reset.next = ~self.reset.value
        await sleep(_RESET_PHASE2)
        self.reset.next = ~self.reset.value
