"""Test Ripple Carry Addition (RCA) algorithm."""

import os

from vcd import VCDWriter

from seqlogic import Module, Vec, run, sleep, u2bv
from seqlogic.algorithm.addition.rca import RCA, adc

DIR = os.path.dirname(__file__)

N_START = 1
N_STOP = 6


def test_functional():
    for n in range(N_START, N_STOP):
        for i in range(2**n):
            for j in range(2**n):
                for k in range(2**1):
                    # Inputs
                    a = u2bv(i, n)
                    b = u2bv(j, n)
                    ci = u2bv(k, 1)

                    # Outputs
                    s = adc(a, b, ci)

                    # Check outputs
                    assert s.to_uint() == (i + j + k)


class Top(Module):
    """Top Level Module."""

    N: int = 8

    def build(self):
        s = self.output(name="s", dtype=Vec[self.N])
        co = self.output(name="co", dtype=Vec[1])

        a = self.input(name="a", dtype=Vec[self.N])
        b = self.input(name="b", dtype=Vec[self.N])
        ci = self.input(name="ci", dtype=Vec[1])

        # Design Under Test
        self.submod(
            name="dut",
            mod=RCA.parameterize(N=self.N),
        ).connect(
            s=s,
            ci=ci,
            a=a,
            b=b,
            co=co,
        )

        self.drv(self.drv_inputs())

    async def drv_inputs(self):
        await sleep(10)

        for i in range(2**self.N):
            for j in range(2**self.N):
                for k in range(2**1):
                    # Inputs
                    self.a.next = u2bv(i, self.N)
                    self.b.next = u2bv(j, self.N)
                    self.ci.next = u2bv(k, 1)

                    await sleep(1)

                    # Check outputs
                    q, r = divmod(i + j + k, 2**self.N)
                    assert self.s.value.to_uint() == r
                    assert self.co.value.to_uint() == q

        await sleep(10)


def test_structural():
    for n in range(N_START, N_STOP):
        vcd = os.path.join(DIR, f"rca_{n}.vcd")
        with (
            open(vcd, "w", encoding="utf-8") as f,
            VCDWriter(f, timescale="1ns") as vcdw,
        ):
            # Instantiate top
            top_n = Top.parameterize(N=n)
            top = top_n(name="top", parent=None)

            # Dump all signals to VCD
            top.dump_vcd(vcdw, ".*")

            # Do the damn thing
            run(top.main())
