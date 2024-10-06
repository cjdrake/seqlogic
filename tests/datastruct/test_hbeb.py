"""Test Pipe Register."""

import os
from collections import deque
from random import randint

from vcd import VCDWriter

from seqlogic import Module, Struct, Vec, finish, resume, run, sleep
from seqlogic.control.globals import drv_clock, drv_reset
from seqlogic.datastruct.hbeb import Hbeb

DIR = os.path.dirname(__file__)


class MyStruct(Struct):
    a: Vec[4]
    b: Vec[4]
    c: Vec[4]
    d: Vec[4]


class Top(Module):
    """Testbench Top."""

    N: int = 1000
    T: type = MyStruct

    def build(self):
        rd_ready = self.logic(name="rd_ready", dtype=Vec[1])
        rd_valid = self.logic(name="rd_valid", dtype=Vec[1])
        rd_data = self.logic(name="rd_data", dtype=self.T)

        wr_ready = self.logic(name="wr_ready", dtype=Vec[1])
        wr_valid = self.logic(name="wr_valid", dtype=Vec[1])
        wr_data = self.logic(name="wr_data", dtype=self.T)

        clock = self.logic(name="clock", dtype=Vec[1])
        reset = self.logic(name="reset", dtype=Vec[1])

        self.submod(
            name="dut",
            mod=Hbeb,
            T=self.T,
        ).connect(
            rd_ready=rd_ready,
            rd_valid=rd_valid,
            rd_data=rd_data,
            wr_ready=wr_ready,
            wr_valid=wr_valid,
            wr_data=wr_data,
            clock=clock,
            reset=reset,
        )

        self.drv(drv_clock(clock, shiftticks=1))
        self.drv(drv_reset(reset, shiftticks=2, onticks=2))
        self.drv(self.drv_wr())
        self.drv(self.drv_rd())

        self.mon(self.mon_wr())
        self.mon(self.mon_rd())

        self.wdata = deque()
        self.rdata = deque()

    async def drv_wr(self):
        clk, rst = self._clock, self._reset
        wr_ready, wr_valid = self._wr_ready, self._wr_valid
        wr_data = self._wr_data

        # reset: __/‾‾
        await rst.posedge()
        wr_valid.next = "1b0"

        # reset: ‾‾\__
        await rst.negedge()

        # Delay a few cycles
        for _ in range(5):
            await clk.posedge()

        def pred():
            return clk.is_posedge() and wr_ready.value == "1b1"

        for _ in range(self.N):
            wr_valid.next = "1b1"
            wr_data.next = self.T.rand()
            await resume((clk, pred))

            for _ in range(randint(0, 5)):
                wr_valid.next = "1b0"
                wr_data.next = self.T.xes()
                await clk.posedge()

        await sleep(10)
        finish()

    async def drv_rd(self):
        clk, rst = self._clock, self._reset
        rd_ready, rd_valid = self._rd_ready, self._rd_valid

        # reset: __/‾‾
        await rst.posedge()
        rd_ready.next = "1b0"

        # reset: ‾‾\__
        await rst.negedge()

        # Delay a few cycles
        for _ in range(5):
            await clk.posedge()

        def pred():
            return clk.is_posedge() and rd_valid.value == "1b1"

        while True:
            rd_ready.next = "1b1"
            await resume((clk, pred))

            for _ in range(randint(0, 5)):
                rd_ready.next = "1b0"
                await clk.posedge()

    async def mon_wr(self):
        clk, rst = self._clock, self._reset
        wr_ready, wr_valid = self._wr_ready, self._wr_valid
        data = self._wr_data

        def pred():
            return (
                clk.is_posedge()
                and rst.is_neg()  # noqa: W503
                and wr_ready.value == "1b1"  # noqa: W503
                and wr_valid.value == "1b1"  # noqa: W503
            )

        while True:
            await resume((clk, pred))
            self.wdata.append(data.value)

    async def mon_rd(self):
        clk, rst = self._clock, self._reset
        rd_ready, rd_valid = self._rd_ready, self._rd_valid
        data = self._rd_data

        def pred():
            return (
                clk.is_posedge()
                and rst.is_neg()  # noqa: W503
                and rd_ready.value == "1b1"  # noqa: W503
                and rd_valid.value == "1b1"  # noqa: W503
            )

        while True:
            await resume((clk, pred))
            self.rdata.append(data.value)

            exp = self.wdata.popleft()
            got = self.rdata.popleft()
            assert got == exp


def test_hbeb():
    vcd = os.path.join(DIR, "hbeb.vcd")
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
