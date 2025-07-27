"""Test HalfAdd module."""

import os

from bvwx import Vec
from deltacycle import run, sleep
from vcd import VCDWriter

from seqlogic import Module
from seqlogic.algorithm.addition.ha import HalfAdd

DIR = os.path.dirname(__file__)


VALUES = [
    ("1b0", "1b0", "1b0", "1b0"),
    ("1b0", "1b1", "1b1", "1b0"),
    ("1b1", "1b0", "1b1", "1b0"),
    ("1b1", "1b1", "1b0", "1b1"),
]


class Top(Module):
    """Top Level Module."""

    def build(self):
        s = self.output(name="s", dtype=Vec[1])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[1])
        b = self.input(name="b", dtype=Vec[1])

        # Design Under Test
        self.submod(
            name="dut",
            mod=HalfAdd,
        ).connect(
            s=s,
            a=a,
            b=b,
            co=co,
        )

        self.drv(self.drv_inputs)

    async def drv_inputs(self):
        await sleep(1)

        for a, b, s, co in VALUES:
            self.a.next = a
            self.b.next = b

            await sleep(1)

            assert self.s.prev == s
            assert self.co.prev == co

        # This helps w/ VCD visualization
        self.a.next = "1b0"
        self.b.next = "1b0"
        await sleep(1)


def test_ha():
    vcd = os.path.join(DIR, "ha.vcd")
    with (
        open(vcd, "w", encoding="utf-8") as f,
        VCDWriter(f, timescale="1ns") as vcdw,
    ):
        top = Top(name="top")
        top.dump_vcd(vcdw, ".*")
        run(top.main())
