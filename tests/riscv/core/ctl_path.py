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

        # Ports
        inst_opcode = Bits(name="inst_opcode", parent=self, dtype=Opcode)
        inst_funct3 = Bits(name="inst_funct3", parent=self, dtype=Funct3)
        inst_funct7 = Bits(name="inst_funct7", parent=self, dtype=Vec[7])
        alu_result_eq_zero = Bit(name="alu_result_eq_zero", parent=self)
        pc_wr_en = Bit(name="pc_wr_en", parent=self)
        regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        data_mem_rd_en = Bit(name="data_mem_rd_en", parent=self)
        data_mem_wr_en = Bit(name="data_mem_wr_en", parent=self)
        reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        alu_func = Bits(name="alu_func", parent=self, dtype=AluOp)
        next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)

        # State
        take_branch = Bit(name="take_branch", parent=self)
        alu_op_type = Bits(name="alu_op_type", parent=self, dtype=CtlAlu)
        default_func = Bits(name="default_funct", parent=self, dtype=AluOp)
        secondary_func = Bits(name="secondary_funct", parent=self, dtype=AluOp)
        branch_func = Bits(name="branch_funct", parent=self, dtype=AluOp)

        # TODO(cjdrake): Remove
        self.inst_opcode = inst_opcode
        self.inst_funct3 = inst_funct3
        self.inst_funct7 = inst_funct7
        self.alu_result_eq_zero = alu_result_eq_zero
        self.pc_wr_en = pc_wr_en
        self.regfile_wr_en = regfile_wr_en
        self.alu_op_a_sel = alu_op_a_sel
        self.alu_op_b_sel = alu_op_b_sel
        self.data_mem_rd_en = data_mem_rd_en
        self.data_mem_wr_en = data_mem_wr_en
        self.reg_writeback_sel = reg_writeback_sel
        self.alu_func = alu_func
        self.next_pc_sel = next_pc_sel
        self.take_branch = take_branch
        self.alu_op_type = alu_op_type
        self.default_func = default_func
        self.secondary_func = secondary_func
        self.branch_func = branch_func

    @reactive
    async def p_c_0(self):
        while True:
            await changed(self.inst_funct3, self.alu_result_eq_zero)
            sel = self.inst_funct3.value.branch
            match sel:
                case Funct3Branch.EQ:
                    self.take_branch.next = ~self.alu_result_eq_zero.value
                case Funct3Branch.NE:
                    self.take_branch.next = self.alu_result_eq_zero.value
                case Funct3Branch.LT:
                    self.take_branch.next = ~self.alu_result_eq_zero.value
                case Funct3Branch.GE:
                    self.take_branch.next = self.alu_result_eq_zero.value
                case Funct3Branch.LTU:
                    self.take_branch.next = ~self.alu_result_eq_zero.value
                case Funct3Branch.GEU:
                    self.take_branch.next = self.alu_result_eq_zero.value
                case _:
                    self.take_branch.xprop(sel)

    @reactive
    async def p_c_1(self):
        while True:
            await changed(self.inst_funct3)
            sel = self.inst_funct3.value.alu_logic
            match sel:
                case Funct3AluLogic.ADD_SUB:
                    self.default_func.next = AluOp.ADD
                case Funct3AluLogic.SLL:
                    self.default_func.next = AluOp.SLL
                case Funct3AluLogic.SLT:
                    self.default_func.next = AluOp.SLT
                case Funct3AluLogic.SLTU:
                    self.default_func.next = AluOp.SLTU
                case Funct3AluLogic.XOR:
                    self.default_func.next = AluOp.XOR
                case Funct3AluLogic.SHIFTR:
                    self.default_func.next = AluOp.SRL
                case Funct3AluLogic.OR:
                    self.default_func.next = AluOp.OR
                case Funct3AluLogic.AND:
                    self.default_func.next = AluOp.AND
                case _:
                    self.default_func.xprop(sel)

            match sel:
                case Funct3AluLogic.ADD_SUB:
                    self.secondary_func.next = AluOp.SUB
                case Funct3AluLogic.SHIFTR:
                    self.secondary_func.next = AluOp.SRA
                case _:
                    self.secondary_func.xprop(sel)

    @reactive
    async def p_c_2(self):
        while True:
            await changed(self.inst_funct3)
            sel = self.inst_funct3.value.branch
            match sel:
                case Funct3Branch.EQ | Funct3Branch.NE:
                    self.branch_func.next = AluOp.SEQ
                case Funct3Branch.LT | Funct3Branch.GE:
                    self.branch_func.next = AluOp.SLT
                case Funct3Branch.LTU | Funct3Branch.GEU:
                    self.branch_func.next = AluOp.SLTU
                case _:
                    self.branch_func.xprop(sel)

    @reactive
    async def p_c_3(self):
        while True:
            await changed(
                self.alu_op_type,
                self.inst_funct3,
                self.inst_funct7,
                self.secondary_func,
                self.default_func,
                self.branch_func,
            )
            sel = self.alu_op_type.value
            match sel:
                case CtlAlu.ADD:
                    self.alu_func.next = AluOp.ADD
                case CtlAlu.BRANCH:
                    self.alu_func.next = self.branch_func.value
                case CtlAlu.OP:
                    sel = self.inst_funct7.value[5]
                    match sel:
                        case "1b0":
                            self.alu_func.next = self.default_func.value
                        case "1b1":
                            self.alu_func.next = self.secondary_func.value
                        case _:
                            self.alu_func.xprop(sel)
                case CtlAlu.OP_IMM:
                    a = self.inst_funct7.value[5]
                    b = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SLL)
                    c = self.inst_funct3.value.alu_logic.eq(Funct3AluLogic.SHIFTR)
                    sel = a & (b | c)
                    match sel:
                        case "1b0":
                            self.alu_func.next = self.default_func.value
                        case "1b1":
                            self.alu_func.next = self.secondary_func.value
                        case _:
                            self.alu_func.xprop(sel)
                case _:
                    self.alu_func.xprop(sel)

    @reactive
    async def p_c_4(self):
        while True:
            await changed(self.inst_opcode, self.take_branch)
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
                    sel = self.take_branch.value
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
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b1"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.DATA
                case Opcode.MISC_MEM:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP_IMM:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.OP_IMM
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.AUIPC:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.STORE:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b1"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.OP:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.OP
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.LUI:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.IMM
                case Opcode.BRANCH:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b0"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.RS2
                    self.alu_op_type.next = CtlAlu.BRANCH
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.ALU
                case Opcode.JALR:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.RS1
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case Opcode.JAL:
                    self.pc_wr_en.next = "1b1"
                    self.regfile_wr_en.next = "1b1"
                    self.alu_op_a_sel.next = CtlAluA.PC
                    self.alu_op_b_sel.next = CtlAluB.IMM
                    self.alu_op_type.next = CtlAlu.ADD
                    self.data_mem_rd_en.next = "1b0"
                    self.data_mem_wr_en.next = "1b0"
                    self.reg_writeback_sel.next = CtlWriteBack.PC4
                case _:
                    self.pc_wr_en.xprop(sel)
                    self.regfile_wr_en.xprop(sel)
                    self.alu_op_a_sel.xprop(sel)
                    self.alu_op_b_sel.xprop(sel)
                    self.alu_op_type.xprop(sel)
                    self.data_mem_rd_en.xprop(sel)
                    self.data_mem_wr_en.xprop(sel)
                    self.reg_writeback_sel.xprop(sel)
