"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.lbool import Vec, cat, rep, zeros
from seqlogic.sim import always_comb

from .. import Inst, Opcode


class ImmedateGenerator(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self.build()

    def build(self):
        # Ports
        self.immediate = Bits(name="immediate", parent=self, dtype=Vec[32])
        self.inst = Bits(name="inst", parent=self, dtype=Inst)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.inst)
            match self.inst.value.opcode:
                case Opcode.LOAD | Opcode.LOAD_FP | Opcode.OP_IMM | Opcode.JALR:
                    self.immediate.next = cat(
                        self.inst.value[20:25],
                        self.inst.value[25:31],
                        rep(self.inst.value[31], 21),
                    )
                case Opcode.STORE_FP | Opcode.STORE:
                    self.immediate.next = cat(
                        self.inst.value[7:12],
                        self.inst.value[25:31],
                        rep(self.inst.value[31], 21),
                    )
                case Opcode.BRANCH:
                    self.immediate.next = cat(
                        zeros(1),
                        self.inst.value[8:12],
                        self.inst.value[25:31],
                        self.inst.value[7],
                        rep(self.inst.value[31], 20),
                    )
                case Opcode.AUIPC | Opcode.LUI:
                    self.immediate.next = cat(
                        zeros(12),
                        self.inst.value[12:20],
                        self.inst.value[20:31],
                        self.inst.value[31],
                    )
                case Opcode.JAL:
                    self.immediate.next = cat(
                        zeros(1),
                        self.inst.value[21:25],
                        self.inst.value[25:31],
                        self.inst.value[20],
                        self.inst.value[12:20],
                        rep(self.inst.value[31], 12),
                    )
                case _:
                    self.immediate.next = Vec[32].dcs()
