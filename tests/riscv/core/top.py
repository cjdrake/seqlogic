"""Top Level Module."""

from seqlogic import Module
from seqlogic import Vector as Vec
from seqlogic import active, sleep

from . import Addr, Inst, Opcode
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
        bus_addr = self.output(name="bus_addr", dtype=Addr)
        bus_wr_en = self.output(name="bus_wr_en", dtype=Vec[1])
        bus_wr_be = self.output(name="bus_wr_be", dtype=Vec[4])
        bus_wr_data = self.output(name="bus_wr_data", dtype=Vec[32])
        bus_rd_en = self.output(name="bus_rd_en", dtype=Vec[1])
        bus_rd_data = self.output(name="bus_rd_data", dtype=Vec[32])

        pc = self.logic(name="pc", dtype=Vec[32])
        inst = self.logic(name="inst", dtype=Inst)

        clock = self.logic(name="clock", dtype=Vec[1])
        reset = self.logic(name="reset", dtype=Vec[1])

        # Submodules:
        # 16K Instruction Memory
        self.submod(
            name="text_mem_bus",
            mod=TextMemBus,
            depth=1024,
        ).connect(
            rd_addr=pc,
            rd_data=(
                lambda d: Inst(
                    opcode=Opcode(d[0:7]),
                    rd=d[7:12],
                    funct3=d[12:15],
                    rs1=d[15:20],
                    rs2=d[20:25],
                    funct7=d[25:32],
                ),
                inst,
            ),
        )

        # 32K Data Memory
        self.submod(
            name="data_mem_bus",
            mod=DataMemBus,
            depth=1024,
        ).connect(
            addr=bus_addr,
            wr_en=bus_wr_en,
            wr_be=bus_wr_be,
            wr_data=bus_wr_data,
            rd_en=bus_rd_en,
            rd_data=bus_rd_data,
            clock=clock,
        )

        # RISC-V Core
        self.submod(
            name="core",
            mod=Core,
        ).connect(
            bus_addr=bus_addr,
            bus_wr_en=bus_wr_en,
            bus_wr_be=bus_wr_be,
            bus_wr_data=bus_wr_data,
            bus_rd_en=bus_rd_en,
            bus_rd_data=bus_rd_data,
            pc=pc,
            inst=inst,
            clock=clock,
            reset=reset,
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
