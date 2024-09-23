"""Test FullAdd module."""

import os

from vcd import VCDWriter

from seqlogic import Module, Vec, get_loop, sleep
from seqlogic.algorithm.addition.fa import FullAdd

loop = get_loop()

DIR = os.path.dirname(__file__)


VALUES = [
    ("1b0", "1b0", "1b0", "1b0", "1b0"),
    ("1b0", "1b0", "1b1", "1b1", "1b0"),
    ("1b0", "1b1", "1b0", "1b1", "1b0"),
    ("1b0", "1b1", "1b1", "1b0", "1b1"),
    ("1b1", "1b0", "1b0", "1b1", "1b0"),
    ("1b1", "1b0", "1b1", "1b0", "1b1"),
    ("1b1", "1b1", "1b0", "1b0", "1b1"),
    ("1b1", "1b1", "1b1", "1b1", "1b1"),
]


class Top(Module):
    """Top Level Module."""

    def build(self):
        s = self.output(name="s", dtype=Vec[1])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[1])
        b = self.input(name="b", dtype=Vec[1])
        ci = self.input(name="ci", dtype=Vec[1])

        # Design Under Test
        self.submod(
            name="dut",
            mod=FullAdd,
        ).connect(
            s=s,
            ci=ci,
            a=a,
            b=b,
            co=co,
        )

        self.drv(self.drv_inputs())

    async def drv_inputs(self):
        await sleep(1)

        for a, b, ci, s, co in VALUES:
            self.a.next = a
            self.b.next = b
            self.ci.next = ci

            await sleep(1)

            assert self.s.value == s
            assert self.co.value == co

        # This helps w/ VCD visualization
        self.a.next = "1b0"
        self.b.next = "1b0"
        self.ci.next = "1b0"
        await sleep(1)


def test_fa():
    vcd = os.path.join(DIR, "fa.vcd")
    with (
        open(vcd, "w", encoding="utf-8") as f,
        VCDWriter(f, timescale="1ns") as vcdw,
    ):
        loop.reset()
        top = Top(name="top")
        top.elab()
        top.dump_vcd(vcdw, ".*")
        loop.run()
