"""Pipe Register."""

from bvwx import Vec

from seqlogic import Module
from seqlogic.check.ready_valid import CheckValid


class PipeReg(Module):
    """Pipe Register."""

    T: type = Vec[8]

    def build(self):
        # Ports
        rd_valid = self.output(name="rd_valid", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=self.T)

        wr_valid = self.input(name="wr_valid", dtype=Vec[1])
        wr_data = self.input(name="wr_data", dtype=self.T)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        # Valid
        self.dff(rd_valid, wr_valid, clock, rst=reset, rval="1b0")

        # Data
        self.dff(rd_data, wr_data, clock, en=wr_valid)

        self.submod(
            name="rv_check_tx",
            mod=CheckValid(
                T=self.T,
                TX=True,
            ),
        ).connect(
            valid=rd_valid,
            data=rd_data,
            clock=clock,
            reset=reset,
        )

        self.submod(
            name="rv_check_rx",
            mod=CheckValid(
                T=self.T,
                RX=True,
            ),
        ).connect(
            valid=wr_valid,
            data=wr_data,
            clock=clock,
            reset=reset,
        )
