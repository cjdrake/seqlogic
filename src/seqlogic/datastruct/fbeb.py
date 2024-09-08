"""Half bandwidth elastic buffer."""

# PyLint thinks some Packed types are Unpacked for some reason
# xpylint: disable=invalid-unary-operand-type

import operator

from seqlogic import Module, Vec


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

        rd_addr_next = self.logic(name="rd_addr_next", dtype=Vec[1])
        wr_addr_next = self.logic(name="wr_addr_next", dtype=Vec[1])

        empty = self.logic(name="empty", dtype=Vec[1])
        full = self.logic(name="full", dtype=Vec[1])

        rd_en = self.logic(name="rd_en", dtype=Vec[1])
        wr_en = self.logic(name="wr_en", dtype=Vec[1])

        empty_next = self.logic(name="empty_next", dtype=Vec[1])
        full_next = self.logic(name="full_next", dtype=Vec[1])

        buf = self.logic(name="buf", dtype=self.T, shape=(2,))

        # Convert ready/valid to FIFO
        self.expr(rd_valid, ~empty)
        self.expr(rd_en, rd_ready & rd_valid)

        self.expr(wr_ready, ~full)
        self.expr(wr_en, wr_ready & wr_valid)

        # Control
        self.expr(empty_next, empty & ~wr_en | ~empty & (rd_en & ~wr_en & (rd_addr ^ wr_addr)))
        self.expr(full_next, full & ~rd_en | ~full & (~rd_en & wr_en & (rd_addr ^ wr_addr)))

        self.dff_r(empty, empty_next, clock, reset, rval="1b1")
        self.dff_r(full, full_next, clock, reset, rval="1b0")

        self.expr(rd_addr_next, ~rd_addr)
        self.expr(wr_addr_next, ~wr_addr)

        self.dff_en_r(rd_addr, rd_addr_next, rd_en, clock, reset, rval="1b0")
        self.dff_en_r(wr_addr, wr_addr_next, wr_en, clock, reset, rval="1b0")

        self.mem_wr_en(buf, wr_addr, wr_data, wr_en, clock)
        self.combi(rd_data, operator.getitem, buf, rd_addr)
