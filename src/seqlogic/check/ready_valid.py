"""
Ready/Valid Protocol Checker
"""

from bvwx import Vec

from seqlogic import Module


class WtfError(Exception):
    pass


def known1(p: Vec[1]) -> bool:
    return not p.has_unknown()


def known2(p: Vec[1], x: Vec) -> bool:
    return not p or not x.has_unknown()


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

        ready = self.input(name="ready", dtype=Vec[1])
        valid = self.input(name="valid", dtype=Vec[1])
        data = self.input(name="data", dtype=self.T)

        clock = self.input(name="clock", dtype=Vec[1])
        reset = self.input(name="reset", dtype=Vec[1])

        async def ready_stable():
            await clock.posedge()
            if not ready.value:
                raise WtfError("WTF")

        async def valid_data_stable():
            data_prev = self.data.value
            await clock.posedge()
            if not (valid.value and data.value == data_prev):
                raise WtfError("WTF")

        if self.TX:
            # Assume: Ready must never be unknown
            self.assume_1(
                name="NeverReadyUnknown",
                f=known1,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.TX_READY_STABLE:
                # Assume: Ready must remain stable until Valid/Data
                self.assume_2(
                    name="ReadyStable",
                    p=lambda: ready.value and not valid.value,
                    q=ready_stable,
                    clk=clock,
                    rst=reset,
                )

            # Assert: Valid must never be unknown
            self.assert_1(
                name="NeverValidUnknown",
                f=known1,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_1(
                name="NeverDataUnknown",
                f=known2,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

            # Assert: Valid/Data must remain stable until Ready
            self.assert_2(
                name="ValidDataStable",
                p=lambda: not ready.value and valid.value,
                q=valid_data_stable,
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_1(
                name="NeverValidUnknown",
                f=known1,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_1(
                name="NeverDataUnknown",
                f=known2,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

            # Assume: Valid/Data must remain stable until Ready
            self.assume_2(
                name="ValidDataStable",
                p=lambda: not ready.value and valid.value,
                q=valid_data_stable,
                clk=clock,
                rst=reset,
            )

            # Assert: Ready must never be unknown
            self.assert_1(
                name="NeverReadyUnknown",
                f=known1,
                xs=(ready,),
                clk=clock,
                rst=reset,
            )

            if self.RX_READY_STABLE:
                # Assert: Ready must remain stable until Valid/Data
                self.assert_2(
                    name="ReadyStable",
                    p=lambda: ready.value and not valid.value,
                    q=ready_stable,
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
            self.assert_1(
                name="NeverValidUnknown",
                f=known1,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assert: Data must never be unknown
            self.assert_1(
                name="NeverDataUnknown",
                f=known2,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.assume_1(
                name="NeverValidUnknown",
                f=known1,
                xs=(valid,),
                clk=clock,
                rst=reset,
            )

            # Assume: Data must never be unknown
            self.assume_1(
                name="NeverDataUnknown",
                f=known2,
                xs=(valid, data),
                clk=clock,
                rst=reset,
            )
