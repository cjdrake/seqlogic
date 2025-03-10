"""Test Pipe Register."""

import os
from collections import deque
from random import randint

from bvwx import Struct, Vec
from deltacycle import finish, resume, run, sleep
from vcd import VCDWriter

from seqlogic import Module
from seqlogic.control.sync import drv_clock, drv_reset
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
            mod=PipeReg.paramz(T=self.T),
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
        await self.reset.posedge()
        self.wr_valid.next = "1b0"

        # reset: ‾‾\__
        await self.reset.negedge()

        # Delay a couple cycles
        for _ in range(2):
            await self.clock.posedge()

        for _ in range(self.N):
            self.wr_valid.next = "1b1"
            self.wr_data.next = self.T.rand()
            await self.clock.posedge()

            for _ in range(randint(0, 2)):
                self.wr_valid.next = "1b0"
                self.wr_data.next = self.T.xes()
                await self.clock.posedge()

        await sleep(8)
        finish()

    async def mon_wr(self):
        def pred():
            return self.clock.is_posedge() and self.reset.is_neg() and self.wr_valid.prev == "1b1"

        while True:
            await resume((self.clock, pred))
            self.wdata.append(self.wr_data.prev)

    async def mon_rd(self):
        def pred():
            return self.clock.is_posedge() and self.reset.is_neg() and self.rd_valid.prev == "1b1"

        while True:
            await resume((self.clock, pred))
            self.rdata.append(self.rd_data.prev)

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

        # Do the damn thing
        run(top.main())
