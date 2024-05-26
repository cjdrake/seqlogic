"""Top Level Module."""

# pyright: reportCallIssue=false

from seqlogic import Bit, Bits, Module, changed, sleep
from seqlogic.lbool import Vec
from seqlogic.sim import active, reactive

from . import Inst, Opcode
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

        # Ports
        self.bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        self.bus_wr_en = Bit(name="bus_wr_en", parent=self)
        self.bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[4])
        self.bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[32])
        self.bus_rd_en = Bit(name="bus_rd_en", parent=self)
        self.bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[32])

        self._pc = Bits(name="pc", parent=self, dtype=Vec[32])
        self._inst = Bits(name="inst", parent=self, dtype=Inst)

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # Submodules:
        # 16K Instruction Memory
        self.text_mem_bus = TextMemBus(name="text_mem_bus", parent=self, depth=4096)
        self.connect(self.text_mem_bus.rd_addr, self._pc)

        # 32K Data Memory
        self.data_mem_bus = DataMemBus(name="data_mem_bus", parent=self, depth=8096)
        self.connect(self.data_mem_bus.addr, self.bus_addr)
        self.connect(self.data_mem_bus.wr_en, self.bus_wr_en)
        self.connect(self.data_mem_bus.wr_be, self.bus_wr_be)
        self.connect(self.data_mem_bus.wr_data, self.bus_wr_data)
        self.connect(self.data_mem_bus.rd_en, self.bus_rd_en)
        self.connect(self.bus_rd_data, self.data_mem_bus.rd_data)
        self.connect(self.data_mem_bus.clock, self.clock)

        # RISC-V Core
        self.core = Core(name="core", parent=self)
        self.connect(self.bus_addr, self.core.bus_addr)
        self.connect(self.bus_wr_en, self.core.bus_wr_en)
        self.connect(self.bus_wr_be, self.core.bus_wr_be)
        self.connect(self.bus_wr_data, self.core.bus_wr_data)
        self.connect(self.bus_rd_en, self.core.bus_rd_en)
        self.connect(self.core.bus_rd_data, self.bus_rd_data)
        self.connect(self._pc, self.core.pc)
        self.connect(self.core.inst, self._inst)
        self.connect(self.core.clock, self.clock)
        self.connect(self.core.reset, self.reset)

    @active
    async def p_i_0(self):
        self.clock.next = "1b0"
        await sleep(_CLOCK_PHASE_SHIFT)
        while True:
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE1)
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE2)

    @active
    async def p_i_1(self):
        self.reset.next = "1b0"
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
