"""Pipe Register."""

from seqlogic import Module, Vec


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
        self.dff_r(rd_valid, wr_valid, clock, reset, rval="1b0")

        # Data
        self.dff_en(rd_data, wr_data, wr_valid, clock)
