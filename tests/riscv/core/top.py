"""Top Level Module."""

# pyright: reportCallIssue=false

from seqlogic import Bit, Bits, Module, sleep
from seqlogic.lbool import Vec
from seqlogic.sim import active

from . import Inst, Opcode
from .core import Core
from .data_mem_bus import DataMemBus
from .text_mem_bus import TextMemBus

_CLOCK_PHASE_SHIFT = 1
_CLOCK_PHASE1 = 1
_CLOCK_PHASE2 = 1
_RESET_PHASE1 = 5
_RESET_PHASE2 = 5


def set_inst(data: Vec[32]):
    return Inst(
        opcode=Opcode(data[0:7]),
        rd=data[7:12],
        funct3=data[12:15],
        rs1=data[15:20],
        rs2=data[20:25],
        funct7=data[25:32],
    )


class Top(Module):
    """Top Level Module."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)

        # Ports
        bus_addr = Bits(name="bus_addr", parent=self, dtype=Vec[32])
        bus_wr_en = Bit(name="bus_wr_en", parent=self)
        bus_wr_be = Bits(name="bus_wr_be", parent=self, dtype=Vec[4])
        bus_wr_data = Bits(name="bus_wr_data", parent=self, dtype=Vec[32])
        bus_rd_en = Bit(name="bus_rd_en", parent=self)
        bus_rd_data = Bits(name="bus_rd_data", parent=self, dtype=Vec[32])

        pc = Bits(name="pc", parent=self, dtype=Vec[32])
        inst = Bits(name="inst", parent=self, dtype=Inst)

        clock = Bit(name="clock", parent=self)
        reset = Bit(name="reset", parent=self)

        # Submodules:
        # 16K Instruction Memory
        text_mem_bus = TextMemBus(name="text_mem_bus", parent=self, depth=4096)
        self.connect(text_mem_bus.rd_addr, pc)

        # 32K Data Memory
        self.data_mem_bus = DataMemBus(name="data_mem_bus", parent=self, depth=8096)
        self.connect(self.data_mem_bus.addr, bus_addr)
        self.connect(self.data_mem_bus.wr_en, bus_wr_en)
        self.connect(self.data_mem_bus.wr_be, bus_wr_be)
        self.connect(self.data_mem_bus.wr_data, bus_wr_data)
        self.connect(self.data_mem_bus.rd_en, bus_rd_en)
        self.connect(bus_rd_data, self.data_mem_bus.rd_data)
        self.connect(self.data_mem_bus.clock, clock)

        # RISC-V Core
        self.core = Core(name="core", parent=self)
        self.connect(bus_addr, self.core.bus_addr)
        self.connect(bus_wr_en, self.core.bus_wr_en)
        self.connect(bus_wr_be, self.core.bus_wr_be)
        self.connect(bus_wr_data, self.core.bus_wr_data)
        self.connect(bus_rd_en, self.core.bus_rd_en)
        self.connect(self.core.bus_rd_data, bus_rd_data)
        self.connect(pc, self.core.pc)
        self.connect(self.core.inst, inst)
        self.connect(self.core.clock, clock)
        self.connect(self.core.reset, reset)

        self.combi(inst, set_inst, text_mem_bus.rd_data)

        # TODO(cjdrake): Remove
        self.bus_addr = bus_addr
        self.bus_wr_en = bus_wr_en
        self.bus_wr_data = bus_wr_data
        self.bus_rd_en = bus_rd_en
        self.pc = pc
        self.inst = inst
        self.clock = clock
        self.reset = reset
        self.text_mem_bus = text_mem_bus

    @active
    async def drive_clock(self):
        self.clock.next = "1b0"
        await sleep(_CLOCK_PHASE_SHIFT)
        while True:
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE1)
            self.clock.next = ~self.clock.value
            await sleep(_CLOCK_PHASE2)

    @active
    async def drive_reset(self):
        self.reset.next = "1b0"
        await sleep(_RESET_PHASE1)
        self.reset.next = ~self.reset.value
        await sleep(_RESET_PHASE2)
        self.reset.next = ~self.reset.value
