"""Control Path."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec
from seqlogic.sim import reactive

from . import (
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


class CtlPath(Module):
    """Control Path Module."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)
        self._build()

    def _build(self):
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

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst_funct3, self.alu_result_equal_zero)
            sel = self.inst_funct3.value.branch
            match sel:
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
                    self._take_branch.xprop(sel)

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.inst_funct3)
            sel = self.inst_funct3.value.alu_logic
            match sel:
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
                    self._default_func.xprop(sel)
                    self._secondary_func.xprop(sel)

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.inst_funct3)
            sel = self.inst_funct3.value.branch
            match sel:
                case Funct3Branch.EQ | Funct3Branch.NE:
                    self._branch_func.next = AluOp.SEQ
                case Funct3Branch.LT | Funct3Branch.GE:
                    self._branch_func.next = AluOp.SLT
                case Funct3Branch.LTU | Funct3Branch.GEU:
                    self._branch_func.next = AluOp.SLTU
                case _:
                    self._branch_func.xprop(sel)

    @reactive
    async def p_c_3(self):
        while True:
            await changed(
                self._alu_op_type,
                self.inst_funct3,
                self.inst_funct7,
                self._secondary_func,
                self._default_func,
                self._branch_func,
            )
            sel = self._alu_op_type.value
            match sel:
                case CtlAlu.ADD:
                    self.alu_function.next = AluOp.ADD
                case CtlAlu.BRANCH:
                    self.alu_function.next = self._branch_func.value
                case CtlAlu.OP:
                    sel = self.inst_funct7.value[5]
                    match sel:
                        case "1b0":
                            self.alu_function.next = self._default_func.value
                        case "1b1":
                            self.alu_function.next = self._secondary_func.value
                        case _:
                            self.alu_function.xprop(sel)
                case CtlAlu.OP_IMM:
                    a = self.inst_funct7.value[5]
                    b = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SLL)
                    c = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SHIFTR)
                    sel = a & (b | c)
                    match sel:
                        case "1b0":
                            self.alu_function.next = self._default_func.value
                        case "1b1":
                            self.alu_function.next = self._secondary_func.value
                        case _:
                            self.alu_function.xprop(sel)
                case _:
                    self.alu_function.xprop(sel)

    @reactive
    async def p_c_4(self):
        while True:
            await changed(self.inst_opcode, self._take_branch)
            sel = self.inst_opcode.value
            match sel:
                case (
                    Opcode.LOAD
                    | Opcode.MISC_MEM  # noqa
                    | Opcode.OP_IMM  # noqa
                    | Opcode.AUIPC  # noqa
                    | Opcode.STORE  # noqa
                    | Opcode.OP  # noqa
                    | Opcode.LUI  # noqa
                ):
                    self.next_pc_sel.next = CtlPc.PC4
                case Opcode.BRANCH:
                    sel = self._take_branch.value
                    match sel:
                        case "1b0":
                            self.next_pc_sel.next = CtlPc.PC4
                        case "1b1":
                            self.next_pc_sel.next = CtlPc.PC_IMM
                        case _:
                            self.next_pc_sel.xprop(sel)
                case Opcode.JALR:
                    self.next_pc_sel.next = CtlPc.RS1_IMM
                case Opcode.JAL:
                    self.next_pc_sel.next = CtlPc.PC_IMM
                case _:
                    self.next_pc_sel.xprop(sel)

    @reactive
    async def p_c_5(self):
        while True:
            await changed(self.inst_opcode)
            sel = self.inst_opcode.value
            match sel:
                case Opcode.LOAD:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b1"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.DATA
                case Opcode.MISC_MEM:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP_IMM:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.OP_IMM
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.AUIPC:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.STORE:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b1"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self._alu_op_type.next = CtlAlu.OP
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.LUI:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.IMM
                case Opcode.BRANCH:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self._alu_op_type.next = CtlAlu.BRANCH
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.JALR:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case Opcode.JAL:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self._alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case _:
                    self.pc_wr_en.xprop(sel)
                    self.regfile_wr_en.xprop(sel)
                    self.alu_op_a_sel.xprop(sel)
                    self.alu_op_b_sel.xprop(sel)
                    self._alu_op_type.xprop(sel)
                    self.data_mem_rd_en.xprop(sel)
                    self.data_mem_wr_en.xprop(sel)
                    self.reg_writeback_sel.xprop(sel)
