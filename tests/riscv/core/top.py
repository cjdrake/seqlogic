"""Top Level Module."""

# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

from seqlogic import Module, sleep
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


class Top(Module):
    """Top Level Module."""

    def __init__(self, name: str):
        super().__init__(name, parent=None)

        # Ports
        bus_addr = self.bits(name="bus_addr", dtype=Vec[32], port=True)
        bus_wr_en = self.bit(name="bus_wr_en", port=True)
        bus_wr_be = self.bits(name="bus_wr_be", dtype=Vec[4], port=True)
        bus_wr_data = self.bits(name="bus_wr_data", dtype=Vec[32], port=True)
        bus_rd_en = self.bit(name="bus_rd_en", port=True)
        bus_rd_data = self.bits(name="bus_rd_data", dtype=Vec[32], port=True)

        pc = self.bits(name="pc", dtype=Vec[32])
        inst = self.bits(name="inst", dtype=Inst)

        clock = self.bit(name="clock")
        reset = self.bit(name="reset")

        # Submodules:
        # 16K Instruction Memory
        text_mem_bus = self.submod(name="text_mem_bus", mod=TextMemBus, depth=1024)
        self.assign(text_mem_bus.rd_addr, pc)

        # 32K Data Memory
        data_mem_bus = self.submod(name="data_mem_bus", mod=DataMemBus, depth=1024)
        self.assign(data_mem_bus.addr, bus_addr)
        self.assign(data_mem_bus.wr_en, bus_wr_en)
        self.assign(data_mem_bus.wr_be, bus_wr_be)
        self.assign(data_mem_bus.wr_data, bus_wr_data)
        self.assign(data_mem_bus.rd_en, bus_rd_en)
        self.assign(bus_rd_data, data_mem_bus.rd_data)
        self.assign(data_mem_bus.clock, clock)

        # RISC-V Core
        core = self.submod(name="core", mod=Core)
        self.assign(bus_addr, core.bus_addr)
        self.assign(bus_wr_en, core.bus_wr_en)
        self.assign(bus_wr_be, core.bus_wr_be)
        self.assign(bus_wr_data, core.bus_wr_data)
        self.assign(bus_rd_en, core.bus_rd_en)
        self.assign(core.bus_rd_data, bus_rd_data)
        self.assign(pc, core.pc)
        self.assign(core.inst, inst)
        self.assign(core.clock, clock)
        self.assign(core.reset, reset)

        # Combinational Logic
        self.combi(
            inst,
            lambda data: Inst(
                opcode=Opcode(data[0:7]),
                rd=data[7:12],
                funct3=data[12:15],
                rs1=data[15:20],
                rs2=data[20:25],
                funct7=data[25:32],
            ),
            text_mem_bus.rd_data,
        )

    @active
    async def drive_clock(self):
        self._clock.next = "1b0"
        await sleep(_CLOCK_PHASE_SHIFT)
        while True:
            self._clock.next = ~self._clock.value
            await sleep(_CLOCK_PHASE1)
            self._clock.next = ~self._clock.value
            await sleep(_CLOCK_PHASE2)

    @active
    async def drive_reset(self):
        self._reset.next = "1b0"
        await sleep(_RESET_PHASE1)
        self._reset.next = ~self._reset.value
        await sleep(_RESET_PHASE2)
        self._reset.next = ~self._reset.value
