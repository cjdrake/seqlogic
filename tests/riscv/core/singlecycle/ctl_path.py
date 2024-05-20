"""Control Path."""

# pyright: reportAttributeAccessIssue=false

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

from .. import (
    AluOp,
    CtlAlu,
    CtlAluA,
    CtlAluB,
    CtlPc,
    CtlWriteBack,
    Funct3,
    Funct3AluLogic,
    Funct3Branch,
    Opcode,
)
from .control import Control


class CtlPath(Module):
    """Control Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self.build()
        self.connect()

    def build(self):
        # Ports
        self.inst_opcode = Bits(name="inst_opcode", parent=self, dtype=Opcode)
        self.inst_funct3 = Bits(name="inst_funct3", parent=self, dtype=Funct3)
        self.inst_funct7 = Bits(name="inst_funct7", parent=self, dtype=Vec[7])
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        self.alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        self.data_mem_rd_en = Bit(name="data_mem_rd_en", parent=self)
        self.data_mem_wr_en = Bit(name="data_mem_wr_en", parent=self)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        self.alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)

        # State
        self._take_branch = Bit(name="take_branch", parent=self)
        self._alu_op_type = Bits(name="alu_op_type", parent=self, dtype=CtlAlu)
        self._default_func = Bits(name="default_funct", parent=self, dtype=AluOp)
        self._secondary_func = Bits(name="secondary_funct", parent=self, dtype=AluOp)
        self._branch_func = Bits(name="branch_funct", parent=self, dtype=AluOp)

        # Submodules
        self.control = Control(name="control", parent=self)

    def connect(self):
        self.pc_wr_en.connect(self.control.pc_wr_en)
        self.regfile_wr_en.connect(self.control.regfile_wr_en)
        self.alu_op_a_sel.connect(self.control.alu_op_a_sel)
        self.alu_op_b_sel.connect(self.control.alu_op_b_sel)
        self._alu_op_type.connect(self.control.alu_op_type)
        self.data_mem_rd_en.connect(self.control.data_mem_rd_en)
        self.data_mem_wr_en.connect(self.control.data_mem_wr_en)
        self.reg_writeback_sel.connect(self.control.reg_writeback_sel)
        self.next_pc_sel.connect(self.control.next_pc_sel)
        self.control.inst_opcode.connect(self.inst_opcode)
        self.control.take_branch.connect(self._take_branch)

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst_funct3, self.alu_result_equal_zero)
            match self.inst_funct3.value.branch:
                case Funct3Branch.EQ:
                    self._take_branch.next = ~(self.alu_result_equal_zero.value)
                case Funct3Branch.NE:
                    self._take_branch.next = self.alu_result_equal_zero.value
                case Funct3Branch.LT:
                    self._take_branch.next = ~(self.alu_result_equal_zero.value)
                case Funct3Branch.GE:
                    self._take_branch.next = self.alu_result_equal_zero.value
                case Funct3Branch.LTU:
                    self._take_branch.next = ~(self.alu_result_equal_zero.value)
                case Funct3Branch.GEU:
                    self._take_branch.next = self.alu_result_equal_zero.value
                case _:
                    self._take_branch.next = Vec[1].dcs()

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.inst_funct3)

            match self.inst_funct3.value.alu_logic:
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

            match self.inst_funct3.value.branch:
                case Funct3Branch.EQ | Funct3Branch.NE:
                    self._branch_func.next = AluOp.SEQ
                case Funct3Branch.LT | Funct3Branch.GE:
                    self._branch_func.next = AluOp.SLT
                case Funct3Branch.LTU | Funct3Branch.GEU:
                    self._branch_func.next = AluOp.SLTU
                case _:
                    self._branch_func.next = AluOp.DC

    @reactive
    async def p_c_2(self):
        while True:
            await changed(
                self._alu_op_type,
                self.inst_funct3,
                self.inst_funct7,
                self._secondary_func,
                self._default_func,
                self._branch_func,
            )
            match self._alu_op_type.value:
                case CtlAlu.ADD:
                    self.alu_function.next = AluOp.ADD
                case CtlAlu.OP:
                    sel = self.inst_funct7.value[5]
                    self.alu_function.next = sel.ite(
                        self._secondary_func.value,
                        self._default_func.value,
                    )
                case CtlAlu.OP_IMM:
                    a = self.inst_funct7.value[5]
                    b = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SLL)
                    c = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SHIFTR)
                    sel = a & (b | c)
                    self.alu_function.next = sel.ite(
                        self._secondary_func.value,
                        self._default_func.value,
                    )
                case CtlAlu.BRANCH:
                    self.alu_function.next = self._branch_func.value
                case _:
                    self.alu_function.next = AluOp.DC
