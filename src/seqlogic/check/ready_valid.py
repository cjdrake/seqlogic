"""
Ready/Valid Protocol Checker
"""

from bvwx import Vec

from seqlogic import Module


class WtfError(Exception):
    pass


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
            # Assume: Ready must never be unknown
            self.mon(self.foo(lambda: not self.ready.value.has_unknown()))

            if self.TX_READY_STABLE:
                # Assume: Ready must remain stable until Valid/Data
                self.mon(
                    self.buz(
                        lambda: self.ready.value and not self.valid.value,
                        self.ready_stable,
                    )
                )

            # Assert: Valid must never be unknown
            self.mon(self.foo(lambda: not self.valid.value.has_unknown()))

            # Assert: Data must never be unknown
            self.mon(
                self.bar(
                    lambda: self.valid.value,
                    lambda: not self.data.value.has_unknown(),
                )
            )

            # Assert: Valid/Data must remain stable until Ready
            self.mon(
                self.buz(
                    lambda: not self.ready.value and self.valid.value,
                    self.valid_data_stable,
                )
            )

        if self.RX:
            # Assume: Valid must never be unknown
            self.mon(self.foo(lambda: not self.valid.value.has_unknown()))

            # Assume: Data must never be unknown
            self.mon(
                self.bar(
                    lambda: self.valid.value,
                    lambda: not self.data.value.has_unknown(),
                )
            )

            # Assume: Valid/Data must remain stable until Ready
            self.mon(
                self.buz(
                    lambda: not self.ready.value and self.valid.value,
                    self.valid_data_stable,
                )
            )

            # Assert: Ready must never be unknown
            self.mon(self.foo(lambda: not self.ready.value.has_unknown()))

            if self.RX_READY_STABLE:
                # Assert: Ready must remain stable until Valid/Data
                self.mon(
                    self.buz(
                        lambda: self.ready.value and not self.valid.value,
                        self.ready_stable,
                    ),
                )

    async def ready_stable(self):
        """TODO(cjdrake): Write docstring."""
        await self.clock.posedge()
        if not self.ready.value:
            raise WtfError("WTF")

    async def valid_data_stable(self):
        """TODO(cjdrake): Write docstring."""
        data = self.data.value
        await self.clock.posedge()
        if not (self.valid.value and self.data.value == data):
            raise WtfError("WTF")


class CheckValid(Module):
    "TODO(cjdrake): Write docstring."

    T: type = Vec[8]

    TX: bool = False
    RX: bool = False

    def build(self):
        if self.TX == self.RX:
            s = f"Expected Tx != Rx, got Tx={self.TX}, Rx={self.RX}"
            raise ValueError(s)

        self.input(name="valid", dtype=Vec[1])
        self.input(name="data", dtype=self.T)

        self.input(name="clock", dtype=Vec[1])
        self.input(name="reset", dtype=Vec[1])

        if self.TX:
            # Assert: Valid must never be unknown
            self.mon(self.foo(lambda: not self.valid.value.has_unknown()))
            # Assert: Data must never be unknown
            self.mon(self.bar(lambda: self.valid.value, lambda: not self.data.value.has_unknown()))

        if self.RX:
            # Assume: Valid must never be unknown
            self.mon(self.foo(lambda: not self.valid.value.has_unknown()))
            # Assume: Data must never be unknown
            self.mon(self.bar(lambda: self.valid.value, lambda: not self.data.value.has_unknown()))
            # Assume: Data must never be unknown
            self.mon(self.bar(lambda: self.valid.value, lambda: not self.data.value.has_unknown()))
