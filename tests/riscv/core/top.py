"""Top Level Module."""

# pyright: reportCallIssue=false

from seqlogic import Bit, Bits, Module, changed, sleep
from seqlogic.lbool import Vec, vec
from seqlogic.sim import active, reactive

from . import WORD_BITS, WORD_BYTES, Inst, Opcode
from .core import Core
from .data_mem_bus import DataMemBus
from .text_mem_bus import TextMemBus

_CLOCK_PHASE_SHIFT = 1
_CLOCK_PHASE1 = 1
_CLOCK_PHASE2 = 1
_RESET_PHASE1 = 5
_RESET_PHASE2 = 5


class Top(Module):
    """Top Level Module."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)
        self._build()
        self._connect()

    def _build(self):
        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[WORD_BYTES])
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[WORD_BITS])
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[WORD_BITS])

        self._pc = Bits(name="pc", parent=self, dtype=Vec[32])
        self._inst = Bits(name="inst", parent=self, dtype=Inst)

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # Submodules:
        # 16K Instruction Memory
        self.text_mem_bus = TextMemBus(name="text_mem_bus", parent=self, depth=4096)
        # 32K Data Memory
        self.data_mem_bus = DataMemBus(name="data_mem_bus", parent=self, depth=8096)
        # RISC-V Core
        self.core = Core(name="core", parent=self)

    def _connect(self):
        self.text_mem_bus.rd_addr.connect(self._pc)

        self.data_mem_bus.addr.connect(self.bus_addr)
        self.data_mem_bus.wr_en.connect(self.bus_wr_en)
        self.data_mem_bus.wr_be.connect(self.bus_wr_be)
        self.data_mem_bus.wr_data.connect(self.bus_wr_data)
        self.data_mem_bus.rd_en.connect(self.bus_rd_en)
        self.bus_rd_data.connect(self.data_mem_bus.rd_data)
        self.data_mem_bus.clock.connect(self.clock)

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

    @active
    async def p_i_0(self):
        self.clock.next = vec("1b0")
        await sleep(_CLOCK_PHASE_SHIFT)
        while True:
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE1)
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE2)

    @active
    async def p_i_1(self):
        self.reset.next = vec("1b0")
        await sleep(_RESET_PHASE1)
        self.reset.next = ~self.reset.value
        await sleep(_RESET_PHASE2)
        self.reset.next = ~self.reset.value

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.text_mem_bus.rd_data)
            self._inst.next = Inst(
                opcode=Opcode(self.text_mem_bus.rd_data.value[0:7]),
                rd=self.text_mem_bus.rd_data.value[7:12],
                funct3=self.text_mem_bus.rd_data.value[12:15],
                rs1=self.text_mem_bus.rd_data.value[15:20],
                rs2=self.text_mem_bus.rd_data.value[20:25],
                funct7=self.text_mem_bus.rd_data.value[25:32],
            )
