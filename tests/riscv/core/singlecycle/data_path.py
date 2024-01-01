"""TODO(cjdrake): Write docstring."""

from seqlogic.hier import Module
from seqlogic.logicvec import F, cat, vec, zeros
from seqlogic.sim import notify
from seqlogic.var import Logic, LogicVec

from ..common.adder import Adder
from ..common.alu import Alu
from ..common.immediate_generator import ImmedateGenerator
from ..common.instruction_decoder import InstructionDecoder
from ..common.mux2 import Mux2
from ..common.mux4 import Mux4
from ..common.mux8 import Mux8
from ..common.regfile import RegFile
from ..common.register import Register
from ..misc import COMBI, TASK


class DataPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        """TODO(cjdrake): Write docstring."""
        super().__init__(name, parent)

        self.data_mem_addr = LogicVec(name="data_mem_addr", parent=self, shape=(32,))
        self.data_mem_wr_data = LogicVec(name="data_mem_wr_data", parent=self, shape=(32,))
        self.data_mem_rd_data = LogicVec(name="data_mem_rd_data", parent=self, shape=(32,))

        self.inst = LogicVec(name="inst", parent=self, shape=(32,))
        self.pc = LogicVec(name="pc", parent=self, shape=(32,))

        # Instruction decode
        self.inst_funct7 = LogicVec(name="inst_funct7", parent=self, shape=(7,))
        self.inst_rs2 = LogicVec(name="inst_rs2", parent=self, shape=(5,))
        self.inst_rs1 = LogicVec(name="inst_rs1", parent=self, shape=(5,))
        self.inst_funct3 = LogicVec(name="inst_funct3", parent=self, shape=(3,))
        self.inst_rd = LogicVec(name="inst_rd", parent=self, shape=(5,))
        self.inst_opcode = LogicVec(name="inst_opcode", parent=self, shape=(7,))

        # Immedate generate
        self.immediate = LogicVec(name="immediate", parent=self, shape=(32,))

        # PC + 4
        self.pc_plus_4 = LogicVec(name="pc_plus_4", parent=self, shape=(32,))

        # PC + Immediate
        self.pc_plus_immediate = LogicVec(name="pc_plus_immediate", parent=self, shape=(32,))

        # Control signals
        self.pc_wr_en = Logic(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Logic(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Logic(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Logic(name="alu_op_b_sel", parent=self)
        self.alu_function = LogicVec(name="alu_function", parent=self, shape=(5,))
        self.reg_writeback_sel = LogicVec(name="reg_writeback_sel", parent=self, shape=(3,))
        self.next_pc_sel = LogicVec(name="next_pc_sel", parent=self, shape=(2,))

        # Select ALU Ops
        self.alu_op_a = LogicVec(name="alu_op_a", parent=self, shape=(32,))
        self.alu_op_b = LogicVec(name="alu_op_b", parent=self, shape=(32,))

        # ALU Outputs
        self.alu_result = LogicVec(name="alu_result", parent=self, shape=(32,))
        self.alu_result_equal_zero = Logic(name="alu_result_equal_zero", parent=self)

        # Next PC
        self.pc_next = LogicVec(name="pc_next", parent=self, shape=(32,))

        # Regfile Write Data
        self.wr_data = LogicVec(name="wr_data", parent=self, shape=(32,))

        self.clock = Logic(name="clock", parent=self)
        self.reset = Logic(name="reset", parent=self)

        # State
        self.rs1_data = LogicVec(name="rs1_data", parent=self, shape=(32,))
        self.rs2_data = LogicVec(name="rs2_data", parent=self, shape=(32,))

        self.connect(self.data_mem_addr, self.alu_result)
        self.connect(self.data_mem_wr_data, self.rs2_data)

        # Submodules
        self.adder_pc_plus_4 = Adder(name="adder_pc_plus_4", parent=self, width=32)
        self.connect(self.pc_plus_4, self.adder_pc_plus_4.result)
        # .op_a(32'h0000_0004)
        self.connect(self.adder_pc_plus_4.op_b, self.pc)

        self.adder_pc_plus_immediate = Adder(name="adder_pc_plus_immediate", parent=self, width=32)
        self.connect(self.pc_plus_immediate, self.adder_pc_plus_immediate.result)
        self.connect(self.adder_pc_plus_immediate.op_a, self.pc)
        self.connect(self.adder_pc_plus_immediate.op_b, self.immediate)

        self.alu = Alu(name="alu", parent=self)
        self.connect(self.alu_result, self.alu.result)
        self.connect(self.alu_result_equal_zero, self.alu.result_equal_zero)
        self.connect(self.alu.alu_function, self.alu_function)
        self.connect(self.alu.op_a, self.alu_op_a)
        self.connect(self.alu.op_b, self.alu_op_b)

        self.mux_next_pc = Mux4(name="mux_next_pc", parent=self, width=32)
        self.connect(self.pc_next, self.mux_next_pc.out)
        self.connect(self.mux_next_pc.sel, self.next_pc_sel)
        self.connect(self.mux_next_pc.in0, self.pc_plus_4)
        self.connect(self.mux_next_pc.in1, self.pc_plus_immediate)
        # .in2 ({alu_result[32-1:1], 1'b0})
        # .in3 (32'h0000_0000)

        self.mux_op_a = Mux2(name="mux_op_a", parent=self, width=32)
        self.connect(self.alu_op_a, self.mux_op_a.out)
        self.connect(self.mux_op_a.sel, self.alu_op_a_sel)
        self.connect(self.mux_op_a.in0, self.rs1_data)
        self.connect(self.mux_op_a.in1, self.pc)

        self.mux_op_b = Mux2(name="mux_op_b", parent=self, width=32)
        self.connect(self.alu_op_b, self.mux_op_b.out)
        self.connect(self.mux_op_b.sel, self.alu_op_b_sel)
        self.connect(self.mux_op_b.in0, self.rs2_data)
        self.connect(self.mux_op_b.in1, self.immediate)

        self.mux_reg_writeback = Mux8(name="mux_reg_writeback", parent=self, width=32)
        self.connect(self.wr_data, self.mux_reg_writeback.out)
        self.connect(self.mux_reg_writeback.sel, self.reg_writeback_sel)
        self.connect(self.mux_reg_writeback.in0, self.alu_result)
        self.connect(self.mux_reg_writeback.in1, self.data_mem_rd_data)
        self.connect(self.mux_reg_writeback.in2, self.pc_plus_4)
        self.connect(self.mux_reg_writeback.in3, self.immediate)
        # .in4(32'h0000_0000)
        # .in5(32'h0000_0000)
        # .in6(32'h0000_0000)
        # .in7(32'h0000_0000)

        self.instruction_decoder = InstructionDecoder(name="instruction_decoder", parent=self)
        self.connect(self.instruction_decoder.inst, self.inst)
        self.connect(self.inst_funct7, self.instruction_decoder.inst_funct7)
        self.connect(self.inst_rs2, self.instruction_decoder.inst_rs2)
        self.connect(self.inst_rs1, self.instruction_decoder.inst_rs1)
        self.connect(self.inst_funct3, self.instruction_decoder.inst_funct3)
        self.connect(self.inst_rd, self.instruction_decoder.inst_rd)
        self.connect(self.inst_opcode, self.instruction_decoder.inst_opcode)

        self.immediate_generator = ImmedateGenerator(name="immediate_generator", parent=self)
        self.connect(self.immediate, self.immediate_generator.immediate)
        self.connect(self.immediate_generator.inst, self.inst)

        self.program_counter = Register(
            name="program_counter", parent=self, width=32, init=vec("32h0040_0000")
        )
        self.connect(self.pc, self.program_counter.q)
        self.connect(self.program_counter.en, self.pc_wr_en)
        self.connect(self.program_counter.d, self.pc_next)
        self.connect(self.program_counter.clock, self.clock)
        self.connect(self.program_counter.reset, self.reset)

        self.regfile = RegFile(name="regfile", parent=self)
        self.connect(self.regfile.wr_en, self.regfile_wr_en)
        self.connect(self.regfile.wr_addr, self.inst_rd)
        self.connect(self.regfile.wr_data, self.wr_data)
        self.connect(self.regfile.rs1_addr, self.inst_rs1)
        self.connect(self.rs1_data, self.regfile.rs1_data)
        self.connect(self.regfile.rs2_addr, self.inst_rs2)
        self.connect(self.rs2_data, self.regfile.rs2_data)
        self.connect(self.regfile.clock, self.clock)

        # Processes
        self._procs.add((self.proc_init, TASK))
        self._procs.add((self.proc_mux, COMBI))

    async def proc_init(self):
        """TODO(cjdrake): Write docstring."""
        self.adder_pc_plus_4.op_a.next = vec("32h0000_0004")
        # mux_next_pc.in3(32'h0000_0000)
        self.mux_next_pc.in3.next = zeros((32,))
        # mux_reg_writeback.{in4, in5, in6, in7}
        self.mux_reg_writeback.in4.next = zeros((32,))
        self.mux_reg_writeback.in5.next = zeros((32,))
        self.mux_reg_writeback.in6.next = zeros((32,))
        self.mux_reg_writeback.in7.next = zeros((32,))

    async def proc_mux(self):
        """TODO(cjdrake): Write docstring."""
        while True:
            await notify(self.alu_result.changed)
            # mux_next_pc.in2({alu_result[31:1], 1'b0})
            self.mux_next_pc.in2.next = cat([F, self.alu_result.next[1:32]])
