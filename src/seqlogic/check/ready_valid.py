"""
Ready/Valid Protocol Checker
"""

from seqlogic import Module, Vec

# `define ASSUME(name, expr, clock, reset) \
# `ifdef ASSERT_ON \
# name: assume property ( \
#    @(posedge clock) disable iff (reset !== 1'b0) (expr) \
# ); \
# `endif

# `define ASSERT(name, expr, clock, reset) \
# `ifdef ASSERT_ON \
# name: assert property ( \
#    @(posedge clock) disable iff (reset !== 1'b0) (expr) \
# ); \
# `endif


class CheckReadyValid(Module):
    "TODO(cjdrake): Write docstring."

    T: type = Vec[8]

    TX: bool = False
    TX_READY_STABLE: bool = False

    RX: bool = False
    RX_READY_STABLE: bool = False

    def build(self):
        if self.TX == self.RX:
            s = f"Expected Tx != Rx, got Tx={self.TX}, Rx={self.RX}"
            raise ValueError(s)

        self.input(name="ready", dtype=Vec[1])
        self.input(name="valid", dtype=Vec[1])
        self.input(name="data", dtype=self.T)

        self.input(name="clock", dtype=Vec[1])
        self.input(name="reset", dtype=Vec[1])

        if self.TX:
            # `ASSUME(NeverReadyUnknown, !$isunknown(ready), clk, rst)

            if self.TX_READY_STABLE:
                # `ASSUME(ReadyStable, (ready && !valid) |=> ready, clk, rst)
                pass

            # `ASSERT(NeverValidUnknown, !$isunknown(valid), clk, rst)
            # `ASSERT(NeverDataUnknown, valid |-> !$isunknown(data), clk, rst)
            # `ASSERT(ValidDataStable, (!ready && valid) |=> (valid && $stable(data)), clk, rst)

        if self.RX:
            # `ASSERT(NeverReadyUnknown, !$isunknown(ready), clk, rst)

            if self.RX_READY_STABLE:
                # `ASSERT(ReadyStable, (ready && !valid) |=> ready, clk, rst)
                pass

            # `ASSUME(NeverValidUnknown, !$isunknown(valid), clk, rst)
            # `ASSUME(NeverDataUnknown, valid |-> !$isunknown(data), clk, rst)
            # `ASSUME(ValidDataStable, (!ready && valid) |=> (valid && $stable(data)), clk, rst)
