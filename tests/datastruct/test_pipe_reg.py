"""Test Pipe Register."""

import os
from collections import deque
from random import randint

from vcd import VCDWriter

from seqlogic import Module, Struct, Vec, finish, resume, run, sleep
from seqlogic.control.globals import drv_clock, drv_reset
from seqlogic.datastruct.pipe_reg import PipeReg

DIR = os.path.dirname(__file__)


class MyStruct(Struct):
    a: Vec[4]
    b: Vec[4]
    c: Vec[4]
    d: Vec[4]


class Top(Module):
    """Testbench Top."""

    N: int = 100
    T: type = MyStruct

    def build(self):
        rd_valid = self.logic(name="rd_valid", dtype=Vec[1])
        rd_data = self.logic(name="rd_data", dtype=self.T)
        wr_valid = self.logic(name="wr_valid", dtype=Vec[1])
        wr_data = self.logic(name="wr_data", dtype=self.T)

        clock = self.logic(name="clock", dtype=Vec[1])
        reset = self.logic(name="reset", dtype=Vec[1])

        self.submod(
            name="dut",
            mod=PipeReg.parameterize(T=self.T),
        ).connect(
            rd_valid=rd_valid,
            rd_data=rd_data,
            wr_valid=wr_valid,
            wr_data=wr_data,
            clock=clock,
            reset=reset,
        )

        self.drv(drv_clock(clock, shiftticks=1))
        self.drv(drv_reset(reset, shiftticks=2, onticks=2))
        self.drv(self.drv_inputs())

        self.mon(self.mon_wr())
        self.mon(self.mon_rd())

        self.wdata = deque()
        self.rdata = deque()

    async def drv_inputs(self):
        # reset: __/‾‾
        await self._reset.posedge()
        self._wr_valid.next = "1b0"

        # reset: ‾‾\__
        await self._reset.negedge()

        # Delay a couple cycles
        for _ in range(2):
            await self._clock.posedge()

        for _ in range(self.N):
            self._wr_valid.next = "1b1"
            self._wr_data.next = self.T.rand()
            await self._clock.posedge()

            for _ in range(randint(0, 2)):
                self._wr_valid.next = "1b0"
                self._wr_data.next = self.T.xes()
                await self._clock.posedge()

        await sleep(8)
        finish()

    async def mon_wr(self):
        clk, rst = self._clock, self._reset
        en = self._wr_valid
        data = self._wr_data

        def pred():
            return clk.is_posedge() and rst.is_neg() and en.value == "1b1"

        while True:
            await resume((clk, pred))
            self.wdata.append(data.value)

    async def mon_rd(self):
        clk, rst = self._clock, self._reset
        en = self._rd_valid
        data = self._rd_data

        def pred():
            return clk.is_posedge() and rst.is_neg() and en.value == "1b1"

        while True:
            await resume((clk, pred))
            self.rdata.append(data.value)

            exp = self.wdata.popleft()
            got = self.rdata.popleft()
            assert got == exp


def test_pipe_reg():
    vcd = os.path.join(DIR, "pipe_reg.vcd")
    with (
        open(vcd, "w", encoding="utf-8") as f,
        VCDWriter(f, timescale="1ns") as vcdw,
    ):
        # Instantiate top
        top = Top(name="top")

        # Dump all signals to VCD
        top.dump_vcd(vcdw, ".*")

        # Register design w/ event loop
        main = top.elab()

        # Do the damn thing
        run(main)
