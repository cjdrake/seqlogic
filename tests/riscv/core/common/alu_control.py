"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.lbool import Vec, vec
from seqlogic.sim import always_comb

from .. import AluOp, CtlAlu, Funct3AluLogic, Funct3Branch

# pyright: reportAttributeAccessIssue=false


class AluControl(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self.build()

    def build(self):
        # Ports
        self.alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self.alu_op_type = Bits(name="alu_op_type", parent=self, dtype=CtlAlu)
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, dtype=Vec[3])
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, dtype=Vec[7])

        # State
        self._default_func = Bits(name="default_funct", parent=self, dtype=AluOp)
        self._secondary_func = Bits(name="secondary_funct", parent=self, dtype=AluOp)
        self._branch_func = Bits(name="branch_funct", parent=self, dtype=AluOp)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.inst_funct3)

            match self.inst_funct3.value:
                case Funct3AluLogic.ADD_SUB:
                    self._default_func.next = AluOp.ADD
                    self._secondary_func.next = AluOp.SUB
                case Funct3AluLogic.SLL:
                    self._default_func.next = AluOp.SLL
                case Funct3AluLogic.SLT:
                    self._default_func.next = AluOp.SLT
                case Funct3AluLogic.SLTU:
                    self._default_func.next = AluOp.SLTU
                case Funct3AluLogic.XOR:
                    self._default_func.next = AluOp.XOR
                case Funct3AluLogic.SHIFTR:
                    self._default_func.next = AluOp.SRL
                    self._secondary_func.next = AluOp.SRA
                case Funct3AluLogic.OR:
                    self._default_func.next = AluOp.OR
                case Funct3AluLogic.AND:
                    self._default_func.next = AluOp.AND
                case _:
                    self._default_func.next = AluOp.DC
                    self._secondary_func.next = AluOp.DC

            match self.inst_funct3.value:
                case Funct3Branch.EQ | Funct3Branch.NE:
                    self._branch_func.next = AluOp.SEQ
                case Funct3Branch.LT | Funct3Branch.GE:
                    self._branch_func.next = AluOp.SLT
                case Funct3Branch.LTU | Funct3Branch.GEU:
                    self._branch_func.next = AluOp.SLTU
                case _:
                    self._branch_func.next = AluOp.DC

    @always_comb
    async def p_c_3(self):
        while True:
            await changed(
                self.alu_op_type,
                self.inst_funct3,
                self.inst_funct7,
                self._secondary_func,
                self._default_func,
                self._branch_func,
            )
            match self.alu_op_type.value:
                case CtlAlu.ADD:
                    self.alu_function.next = AluOp.ADD
                case CtlAlu.OP:
                    s = self.inst_funct7.value[5]
                    self.alu_function.next = s.ite(
                        self._secondary_func.value,
                        self._default_func.value,
                    )
                case CtlAlu.OP_IMM:
                    s = self.inst_funct7.value[5]
                    s &= self.inst_funct3.value[0:2].eq(vec("2b01"))
                    self.alu_function.next = s.ite(
                        self._secondary_func.value,
                        self._default_func.value,
                    )
                case CtlAlu.BRANCH:
                    self.alu_function.next = self._branch_func.value
                case _:
                    self.alu_function.next = AluOp.DC
