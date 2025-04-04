"""
Full Bandwidth Elastic Buffer (FBEB) Control
"""

from bvwx import Vec

from seqlogic import ITE, Module


class FbebCtl(Module):
    """Full Bandwidth Elastic Buffer Control"""

    def build(self):
        # Ports
        rd_addr = self.output(name="rd_addr", dtype=Vec[1])
        wr_addr = self.output(name="wr_addr", dtype=Vec[1])

        empty = self.output(name="empty", dtype=Vec[1])
        full = self.output(name="full", dtype=Vec[1])

        rd_en = self.input(name="rd_en", dtype=Vec[1])
        wr_en = self.input(name="wr_en", dtype=Vec[1])

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        # Read Address
        rd_addr_next = self.logic(name="rd_addr_next", dtype=Vec[1])
        self.expr(rd_addr_next, ~rd_addr)
        self.dff(rd_addr, rd_addr_next, clock, en=rd_en, rst=reset, rval="1b0")

        # Write Address
        wr_addr_next = self.logic(name="wr_addr_next", dtype=Vec[1])
        self.expr(wr_addr_next, ~wr_addr)
        self.dff(wr_addr, wr_addr_next, clock, en=wr_en, rst=reset, rval="1b0")

        # Empty
        empty_next = self.logic(name="empty_next", dtype=Vec[1])
        self.expr(empty_next, ITE(empty, ~wr_en, rd_en & ~wr_en & (rd_addr ^ wr_addr)))
        self.dff(empty, empty_next, clock, rst=reset, rval="1b1")

        # Full
        full_next = self.logic(name="full_next", dtype=Vec[1])
        self.expr(full_next, ITE(full, ~rd_en, ~rd_en & wr_en & (rd_addr ^ wr_addr)))
        self.dff(full, full_next, clock, rst=reset, rval="1b0")
