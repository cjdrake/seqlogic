"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, changed
from seqlogic.lbool import Vec, cat, uint2vec, vec, zeros
from seqlogic.sim import always_comb

from .. import TEXT_BASE, AluOp, CtlAluA, CtlAluB, CtlPc, CtlWriteBack, Inst
from ..common.alu import Alu
from ..common.immediate_generator import ImmedateGenerator
from ..common.regfile import RegFile
from ..common.register import Register


class DataPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        self.data_mem_addr = Bits(name="data_mem_addr", parent=self, dtype=Vec[32])
        self.data_mem_wr_data = Bits(name="data_mem_wr_data", parent=self, dtype=Vec[32])
        self.data_mem_rd_data = Bits(name="data_mem_rd_data", parent=self, dtype=Vec[32])

        self.inst = Bits(name="inst", parent=self, dtype=Inst)
        self.pc = Bits(name="pc", parent=self, dtype=Vec[32])

        # Immedate generate
        self.immediate = Bits(name="immediate", parent=self, dtype=Vec[32])

        # PC + 4
        self.pc_plus_4 = Bits(name="pc_plus_4", parent=self, dtype=Vec[32])

        # PC + Immediate
        self.pc_plus_immediate = Bits(name="pc_plus_immediate", parent=self, dtype=Vec[32])

        # Control signals
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bits(name="alu_op_a_sel", parent=self, dtype=CtlAluA)
        self.alu_op_b_sel = Bits(name="alu_op_b_sel", parent=self, dtype=CtlAluB)
        self.alu_function = Bits(name="alu_function", parent=self, dtype=AluOp)
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, dtype=CtlWriteBack)
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, dtype=CtlPc)

        # Select ALU Ops
        self.alu_op_a = Bits(name="alu_op_a", parent=self, dtype=Vec[32])
        self.alu_op_b = Bits(name="alu_op_b", parent=self, dtype=Vec[32])

        # ALU Outputs
        self.alu_result = Bits(name="alu_result", parent=self, dtype=Vec[32])
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)

        # Next PC
        self.pc_next = Bits(name="pc_next", parent=self, dtype=Vec[32])

        # Regfile Write Data
        self.wr_data = Bits(name="wr_data", parent=self, dtype=Vec[32])

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self.rs1_data = Bits(name="rs1_data", parent=self, dtype=Vec[32])
        self.rs2_data = Bits(name="rs2_data", parent=self, dtype=Vec[32])

        # Submodules
        self.alu = Alu(name="alu", parent=self)
        self.immediate_generator = ImmedateGenerator(name="immediate_generator", parent=self)
        pc_init = uint2vec(TEXT_BASE, 32)
        self.program_counter = Register(name="program_counter", parent=self, init=pc_init)
        self.regfile = RegFile(name="regfile", parent=self)

    def connect(self):
        self.data_mem_addr.connect(self.alu_result)
        self.data_mem_wr_data.connect(self.rs2_data)

        self.alu_result.connect(self.alu.result)
        self.alu_result_equal_zero.connect(self.alu.result_equal_zero)
        self.alu.alu_function.connect(self.alu_function)
        self.alu.op_a.connect(self.alu_op_a)
        self.alu.op_b.connect(self.alu_op_b)

        self.immediate.connect(self.immediate_generator.immediate)
        self.immediate_generator.inst.connect(self.inst)

        self.pc.connect(self.program_counter.q)
        self.program_counter.en.connect(self.pc_wr_en)
        self.program_counter.d.connect(self.pc_next)
        self.program_counter.clock.connect(self.clock)
        self.program_counter.reset.connect(self.reset)

        self.regfile.wr_en.connect(self.regfile_wr_en)
        self.regfile.wr_data.connect(self.wr_data)
        self.rs1_data.connect(self.regfile.rs1_data)
        self.rs2_data.connect(self.regfile.rs2_data)
        self.regfile.clock.connect(self.clock)

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.inst)
            self.regfile.rs2_addr.next = self.inst.value.rs2
            self.regfile.rs1_addr.next = self.inst.value.rs1
            self.regfile.wr_addr.next = self.inst.value.rd

    @always_comb
    async def p_c_2(self):
        while True:
            await changed(self.pc)
            self.pc_plus_4.next = self.pc.value + vec("32h0000_0004")

    @always_comb
    async def p_c_3(self):
        while True:
            await changed(self.pc, self.immediate)
            self.pc_plus_immediate.next = self.pc.value + self.immediate.value

    @always_comb
    async def p_c_4(self):
        while True:
            await changed(self.next_pc_sel, self.pc_plus_4, self.pc_plus_immediate, self.alu_result)
            match self.next_pc_sel.value:
                case CtlPc.PC4:
                    self.pc_next.next = self.pc_plus_4.value
                case CtlPc.PC_IMM:
                    self.pc_next.next = self.pc_plus_immediate.value
                case CtlPc.RS1_IMM:
                    self.pc_next.next = cat(zeros(1), self.alu_result.value[1:])
                case _:
                    self.pc_next.next = Vec[32].dcs()

    @always_comb
    async def p_c_5(self):
        while True:
            await changed(self.alu_op_a_sel, self.rs1_data, self.pc)
            match self.alu_op_a_sel.value:
                case CtlAluA.RS1:
                    self.alu_op_a.next = self.rs1_data.value
                case CtlAluA.PC:
                    self.alu_op_a.next = self.pc.value
                case _:
                    self.alu_op_a.next = Vec[32].dcs()

    @always_comb
    async def p_c_6(self):
        while True:
            await changed(self.alu_op_b_sel, self.rs2_data, self.immediate)
            match self.alu_op_b_sel.value:
                case CtlAluB.RS2:
                    self.alu_op_b.next = self.rs2_data.value
                case CtlAluB.IMM:
                    self.alu_op_b.next = self.immediate.value
                case _:
                    self.alu_op_b.next = Vec[32].dcs()

    @always_comb
    async def p_c_7(self):
        while True:
            await changed(
                self.reg_writeback_sel,
                self.alu_result,
                self.data_mem_rd_data,
                self.pc_plus_4,
                self.immediate,
            )
            match self.reg_writeback_sel.value:
                case CtlWriteBack.ALU:
                    self.wr_data.next = self.alu_result.value
                case CtlWriteBack.DATA:
                    self.wr_data.next = self.data_mem_rd_data.value
                case CtlWriteBack.PC4:
                    self.wr_data.next = self.pc_plus_4.value
                case CtlWriteBack.IMM:
                    self.wr_data.next = self.immediate.value
                case _:
                    self.wr_data.next = Vec[32].dcs()
