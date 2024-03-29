"""TODO(cjdrake): Write docstring."""

from seqlogic import Bits, Module, changed
from seqlogic.bits import T, bits
from seqlogic.sim import always_comb

from .constants import AluOp, CtlAlu, Funct3AluLogic, Funct3Branch

# pyright: reportAttributeAccessIssue=false


class AluControl(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.build()

    def build(self):
        """TODO(cjdrake): Write docstring."""
        # Ports
        self.alu_function = Bits(name="alu_function", parent=self, shape=(5,))
        self.alu_op_type = Bits(name="alu_op_type", parent=self, shape=(2,))
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, shape=(3,))
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, shape=(7,))

        # State
        self.default_funct = Bits(name="default_funct", parent=self, shape=(5,))
        self.secondary_funct = Bits(name="seconary_funct", parent=self, shape=(5,))
        self.branch_funct = Bits(name="branch_funct", parent=self, shape=(5,))

    @always_comb
    async def p_c_0(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.inst_funct3)
            match self.inst_funct3.next:
                case Funct3AluLogic.ADD_SUB:
                    self.default_funct.next = AluOp.ADD
                case Funct3AluLogic.SLL:
                    self.default_funct.next = AluOp.SLL
                case Funct3AluLogic.SLT:
                    self.default_funct.next = AluOp.SLT
                case Funct3AluLogic.SLTU:
                    self.default_funct.next = AluOp.SLTU
                case Funct3AluLogic.XOR:
                    self.default_funct.next = AluOp.XOR
                case Funct3AluLogic.SHIFTR:
                    self.default_funct.next = AluOp.SRL
                case Funct3AluLogic.OR:
                    self.default_funct.next = AluOp.OR
                case Funct3AluLogic.AND:
                    self.default_funct.next = AluOp.AND
                case _:
                    self.default_funct.next = AluOp.X

    @always_comb
    async def p_c_1(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.inst_funct3)
            match self.inst_funct3.next:
                case Funct3AluLogic.ADD_SUB:
                    self.secondary_funct.next = AluOp.SUB
                case Funct3AluLogic.SHIFTR:
                    self.secondary_funct.next = AluOp.SRA
                case _:
                    self.secondary_funct.next = AluOp.X

    @always_comb
    async def p_c_2(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(self.inst_funct3)
            match self.inst_funct3.next:
                case Funct3Branch.EQ | Funct3Branch.NE:
                    self.branch_funct.next = AluOp.SEQ
                case Funct3Branch.LT | Funct3Branch.GE:
                    self.branch_funct.next = AluOp.SLT
                case Funct3Branch.LTU | Funct3Branch.GEU:
                    self.branch_funct.next = AluOp.SLTU
                case _:
                    self.branch_funct.next = AluOp.X

    @always_comb
    async def p_c_3(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await changed(
                self.alu_op_type,
                self.inst_funct3,
                self.inst_funct7,
                self.secondary_funct,
                self.default_funct,
                self.branch_funct,
            )
            match self.alu_op_type.next:
                case CtlAlu.ADD:
                    self.alu_function.next = AluOp.ADD
                case CtlAlu.OP:
                    if self.inst_funct7.next[5] == T:
                        self.alu_function.next = self.secondary_funct.next
                    else:
                        self.alu_function.next = self.default_funct.next
                case CtlAlu.OP_IMM:
                    if self.inst_funct7.next[5] == T and self.inst_funct3.next[0:2] == bits("2b01"):
                        self.alu_function.next = self.secondary_funct.next
                    else:
                        self.alu_function.next = self.default_funct.next
                case CtlAlu.BRANCH:
                    self.alu_function.next = self.branch_funct.next
                case _:
                    self.alu_function.next = AluOp.X
