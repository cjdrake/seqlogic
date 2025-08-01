"""
Ready/Valid Protocol Checker
"""

from bvwx import Vec, bits

from seqlogic import BitsConst, Module, Packed

# TODO(cjdrake): Use expect_scalar?
ONE = BitsConst(bits("1b1"))


def known(p: Vec[1]) -> bool:
    return not p.has_unknown()


async def valid_data_stable(valid: Packed, data: Packed, clock: Packed) -> Vec[1]:
    data_prev = data.value
    await clock.posedge()
    return valid.value & (data.value == data_prev)


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
            self.assume_func(
                name="NeverReadyUnknown",
                p=ONE,
                f=known,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.TX_READY_STABLE:
                # Assume: Ready must remain stable until Valid/Data
                self.assume_next(
                    name="ReadyStable",
                    p=(ready & ~valid),
                    q=ready,
                    clk=clock,
                    rst=reset,
                )

            # Assert: Valid must never be unknown
            self.assert_func(
                name="NeverValidUnknown",
                p=ONE,
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_func(
                name="NeverDataUnknown",
                p=valid,
                f=known,
                xs=(data,),
                clk=clock,
                rst=reset,
            )

            # Assert: Valid/Data must remain stable until Ready
            self.assert_seq(
                name="ValidDataStable",
                p=(~ready & valid),
                s=valid_data_stable,
                xs=(valid, data, clock),
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_func(
                name="NeverValidUnknown",
                p=ONE,
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_func(
                name="NeverDataUnknown",
                p=valid,
                f=known,
                xs=(data,),
                clk=clock,
                rst=reset,
            )

            # Assume: Valid/Data must remain stable until Ready
            self.assume_seq(
                name="ValidDataStable",
                p=(~ready & valid),
                s=valid_data_stable,
                xs=(valid, data, clock),
                clk=clock,
                rst=reset,
            )

            # Assert: Ready must never be unknown
            self.assert_func(
                name="NeverReadyUnknown",
                p=ONE,
                f=known,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.RX_READY_STABLE:
                # Assert: Ready must remain stable until Valid/Data
                self.assert_next(
                    name="ReadyStable",
                    p=(ready & ~valid),
                    q=ready,
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
            self.assert_func(
                name="NeverValidUnknown",
                p=ONE,
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_func(
                name="NeverDataUnknown",
                p=valid,
                f=known,
                xs=(data,),
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_func(
                name="NeverValidUnknown",
                p=ONE,
                f=known,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_func(
                name="NeverDataUnknown",
                p=valid,
                f=known,
                xs=(data,),
                clk=clock,
                rst=reset,
            )
