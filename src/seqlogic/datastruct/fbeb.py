"""Half bandwidth elastic buffer."""

# PyLint thinks some Packed types are Unpacked for some reason
# pylint: disable=invalid-unary-operand-type

import operator

from seqlogic import Module, Vec

from .fbeb_ctl import FbebCtl


class Fbeb(Module):
    """Full bandwidth elastic buffer."""

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
        rd_addr = self.logic(name="rd_addr", dtype=Vec[1])
        wr_addr = self.logic(name="wr_addr", dtype=Vec[1])

        empty = self.logic(name="empty", dtype=Vec[1])
        full = self.logic(name="full", dtype=Vec[1])

        rd_en = self.logic(name="rd_en", dtype=Vec[1])
        wr_en = self.logic(name="wr_en", dtype=Vec[1])

        # Convert ready/valid to FIFO (full/push, empty/pop)
        self.expr(rd_valid, ~empty)
        self.expr(rd_en, rd_ready & rd_valid)
        self.expr(wr_ready, ~full)
        self.expr(wr_en, wr_ready & wr_valid)

        # Control
        self.submod(
            name="ctl",
            mod=FbebCtl,
        ).connect(
            rd_addr=rd_addr,
            wr_addr=wr_addr,
            empty=empty,
            full=full,
            rd_en=rd_en,
            wr_en=wr_en,
            clock=clock,
            reset=reset,
        )

        # Data
        data = self.logic(name="data", dtype=self.T, shape=(2,))

        # Read Port
        self.combi(rd_data, operator.getitem, data, rd_addr)

        # Write Port
        self.mem_wr_en(data, wr_addr, wr_data, wr_en, clock)
