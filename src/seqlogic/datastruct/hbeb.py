"""Half bandwidth elastic buffer."""

# PyLint thinks some Packed types are Unpacked for some reason
# pylint: disable=invalid-unary-operand-type

from bvwx import Vec

from seqlogic import ITE, Module


class Hbeb(Module):
    """Half bandwidth elastic buffer."""

    T: type = Vec[8]

    def build(self):
        # Ports
        rd_ready = self.input(name="rd_ready", dtype=Vec[1])
        rd_valid = self.output(name="rd_valid", dtype=Vec[1])
        rd_data = self.output(name="rd_data", dtype=self.T)

        wr_ready = self.output(name="wr_ready", dtype=Vec[1])
        wr_valid = self.input(name="wr_valid", dtype=Vec[1])
        wr_data = self.input(name="wr_data", dtype=self.T)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        # FIFO Control
        full = self.logic(name="full", dtype=Vec[1])

        rd_en = self.logic(name="rd_en", dtype=Vec[1])
        wr_en = self.logic(name="wr_en", dtype=Vec[1])

        # Convert ready/valid to FIFO
        self.assign(rd_valid, full)
        self.expr(rd_en, rd_ready & rd_valid)

        self.expr(wr_ready, ~full)
        self.expr(wr_en, wr_ready & wr_valid)

        # Control
        full_next = self.logic(name="full_next", dtype=Vec[1])
        self.expr(full_next, ITE(full, ~rd_en, wr_en))
        self.dff(full, full_next, clock, rst=reset, rval="1b0")

        # Data
        data = self.logic(name="data", dtype=self.T)

        # Read Port
        self.assign(rd_data, data)

        # Write Port
        self.dff(data, wr_data, clock, en=wr_en)
