# Sequential Logic

SeqiLog (pronounced seh-kwi-log) is a Python library for logic design and verification.

[Read the docs!](https://seqlogic.rtfd.org) (WIP)

[![Documentation Status](https://readthedocs.org/projects/seqlogic/badge/?version=latest)](https://seqlogic.readthedocs.io/en/latest/?badge=latest)

## Features

SeqiLog provides building blocks to simulate hardware at the register transfer
level (RTL) of abstraction:

* Hierarchical, parameterized `Module` design element
* Four-state `bits` multidimensional array data type
* Discrete event simulation using `async` / `await` syntax

SeqiLog is *declarative*.
To the extent possible,
the designer should only need to know *what* components to declare,
not *how* they interact with the task scheduling algorithm.

SeqiLog is *strict*.
Functions should raise exceptions when arguments have inconsistent types.
Uninitialized or metastable state should always propagate pessimistically.

The API is an experiment in how to create a *Pythonic* meta-HDL.
It is currently a work in progress.
Expect breaking changes from time to time.

## Example

The following code implements a D flip flop (DFF) with the D input connected
to the inverted Q output.

```python
from bvwx import Vec
from vcd import VCDWriter
from seqlogic import Module, run, sleep


async def drv_clock(y):
    """Positive clock w/ no phase shift, period T=2, 50% duty cycle."""
    while True:
        y.next = "1b1"
        await sleep(1)
        y.next = "1b0"
        await sleep(1)


async def drv_reset(y):
    """Positive reset asserting from T=[1..2]"""
    y.next = "1b0"
    await sleep(1)
    y.next = "1b1"
    await sleep(1)
    y.next = "1b0"


class Top(Module):
    """Data flip flop (DFF) Example"""

    def build(self):
        clk = self.logic(name="clk", dtype=Vec[1])
        rst = self.logic(name="rst", dtype=Vec[1])
        q = self.logic(name="q", dtype=Vec[1])
        d = self.logic(name="d", dtype=Vec[1])

        # d = NOT(q)
        self.expr(d, ~q)

        # DFF w/ async positive (active high) reset, reset to 0
        self.dff(q, d, clk, rst=rst, rval="1b0")

        # Clock/Reset
        self.drv(drv_clock, clk)
        self.drv(drv_reset, rst)


# Run simulation w/ VCD dump enabled
with (
    open("dff.vcd", "w") as f,
    VCDWriter(f, timescale="1ns") as vcdw,
):
    top = Top(name="top")
    top.dump_vcd(vcdw, ".*")
    run(top.main(), ticks=20)
```

Use [GTKWave](https://gtkwave.sourceforge.net)
or [Surfer](https://surfer-project.org) to view the VCD wave dump.
It should look this this:

```
T (ns)   0  1  2  3  4  5  6  7  8  9 ...
---------+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

clk      /‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\__/‾‾\

rst      ___/‾‾\___________________________________________________

q        XXXX________/‾‾‾‾‾\_____/‾‾‾‾‾\_____/‾‾‾‾‾\_____/‾‾‾‾‾\___

d        XXXX‾‾‾‾‾‾‾‾\_____/‾‾‾‾‾\_____/‾‾‾‾‾\_____/‾‾‾‾‾\_____/‾‾‾

---------+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

See the `ipynb` and `tests` directories for more examples.

## Installing

SeqiLog is available on [PyPI](https://pypi.org):

    $ pip install seqlogic

It supports Python 3.12+.

## Developing

SeqiLog's repository is on [GitHub](https://github.com):

    $ git clone https://github.com/cjdrake/seqlogic.git

Runtime dependencies are listed in `requirements.txt`.
Development dependencies are listed in `requirements-dev.txt`.
