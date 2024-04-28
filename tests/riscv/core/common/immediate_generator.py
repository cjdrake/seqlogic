"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.lbool import cat, rep, vec, zeros
from seqlogic.sim import always_comb

from .. import Opcode


class ImmedateGenerator(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()

    def build(self):
        # Ports
        self.immediate = Bits(name="immediate", parent=self, shape=(32,))
        self.inst = Bits(name="inst", parent=self, shape=(32,))

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.inst)
            if self.inst.next[0:7] in [Opcode.LOAD, Opcode.LOAD_FP, Opcode.OP_IMM, Opcode.JALR]:
                self.immediate.next = cat(
                    self.inst.next[20:25], self.inst.next[25:31], rep(self.inst.next[31], 21)
                )
            elif self.inst.next[0:7] in [Opcode.STORE_FP, Opcode.STORE]:
                self.immediate.next = cat(
                    self.inst.next[7:12], self.inst.next[25:31], rep(self.inst.next[31], 21)
                )
            elif self.inst.next[0:7] == Opcode.BRANCH:
                self.immediate.next = cat(
                    zeros(1),
                    self.inst.next[8:12],
                    self.inst.next[25:31],
                    self.inst.next[7],
                    rep(self.inst.next[31], 20),
                )
            elif self.inst.next[0:7] in [Opcode.AUIPC, Opcode.LUI]:
                self.immediate.next = cat(
                    vec("12b0000_0000_0000"),
                    self.inst.next[12:20],
                    self.inst.next[20:31],
                    self.inst.next[31],
                )
            elif self.inst.next[0:7] == Opcode.JAL:
                self.immediate.next = cat(
                    zeros(1),
                    self.inst.next[21:25],
                    self.inst.next[25:31],
                    self.inst.next[20],
                    self.inst.next[12:20],
                    rep(self.inst.next[31], 12),
                )
            else:
                self.immediate.next = zeros(32)
