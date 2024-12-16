"""
Ready/Valid Protocol Checker
"""

from bvwx import Vec

from seqlogic import Module, Packed


def known(p: Vec[1]) -> bool:
    return not p.has_unknown()


def pred_known(p: Vec[1], x: Vec) -> bool:
    return not p or not x.has_unknown()


def foo(ready: Vec[1], valid: Vec[1]) -> bool:
    return bool(ready & ~valid)


def bar(ready: Vec[1], valid: Vec[1]) -> bool:
    return bool(~ready & valid)


async def ready_stable(ready: Packed, clock: Packed) -> bool:
    await clock.posedge()
    return bool(ready.value)


async def valid_data_stable(valid: Packed, data: Packed, clock: Packed) -> bool:
    data_prev = data.value
    await clock.posedge()
    return bool(valid.value and data.value == data_prev)


class CheckReadyValid(Module):
    "Check ready/valid protocol."

    T: type = Vec[8]

    TX: bool = False
    TX_READY_STABLE: bool = False

    RX: bool = False
    RX_READY_STABLE: bool = False

    def build(self):
        if self.TX == self.RX:
            s = f"Expected Tx != Rx, got Tx={self.TX}, Rx={self.RX}"
            raise ValueError(s)

        ready = self.input(name="ready", dtype=Vec[1])
        valid = self.input(name="valid", dtype=Vec[1])
        data = self.input(name="data", dtype=self.T)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        if self.TX:
            # Assume: Ready must never be unknown
            self.assume_immed(
                name="NeverReadyUnknown",
                f=known,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.TX_READY_STABLE:
                # Assume: Ready must remain stable until Valid/Data
                self.assume_seq(
                    name="ReadyStable",
                    p=foo,
                    p_xs=(ready, valid),
                    q=ready_stable,
                    q_xs=(ready, clock),
                    clk=clock,
                    rst=reset,
                )

            # Assert: Valid must never be unknown
            self.assert_immed(
                name="NeverValidUnknown",
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_immed(
                name="NeverDataUnknown",
                f=pred_known,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

            # Assert: Valid/Data must remain stable until Ready
            self.assert_seq(
                name="ValidDataStable",
                p=bar,
                p_xs=(ready, valid),
                q=valid_data_stable,
                q_xs=(valid, data, clock),
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_immed(
                name="NeverValidUnknown",
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_immed(
                name="NeverDataUnknown",
                f=pred_known,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

            # Assume: Valid/Data must remain stable until Ready
            self.assume_seq(
                name="ValidDataStable",
                p=bar,
                p_xs=(ready, valid),
                q=valid_data_stable,
                q_xs=(valid, data, clock),
                clk=clock,
                rst=reset,
            )

            # Assert: Ready must never be unknown
            self.assert_immed(
                name="NeverReadyUnknown",
                f=known,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.RX_READY_STABLE:
                # Assert: Ready must remain stable until Valid/Data
                self.assert_seq(
                    name="ReadyStable",
                    p=foo,
                    p_xs=(ready, valid),
                    q=ready_stable,
                    q_xs=(ready, clock),
                    clk=clock,
                    rst=reset,
                )


class CheckValid(Module):
    "TODO(cjdrake): Write docstring."

    T: type = Vec[8]

    TX: bool = False
    RX: bool = False

    def build(self):
        if self.TX == self.RX:
            s = f"Expected Tx != Rx, got Tx={self.TX}, Rx={self.RX}"
            raise ValueError(s)

        valid = self.input(name="valid", dtype=Vec[1])
        data = self.input(name="data", dtype=self.T)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        if self.TX:
            # Assert: Valid must never be unknown
            self.assert_immed(
                name="NeverValidUnknown",
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_immed(
                name="NeverDataUnknown",
                f=pred_known,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_immed(
                name="NeverValidUnknown",
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_immed(
                name="NeverDataUnknown",
                f=pred_known,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )
