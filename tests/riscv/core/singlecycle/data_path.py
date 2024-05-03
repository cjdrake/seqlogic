"""TODO(cjdrake): Write docstring."""

from seqlogic import Bit, Bits, Module, Struct, changed
from seqlogic.lbool import cat, uint2vec, vec, zeros
from seqlogic.sim import always_comb, initial

from .. import TEXT_BASE, Inst
from ..common.adder import Adder
from ..common.alu import Alu
from ..common.immediate_generator import ImmedateGenerator
from ..common.mux import Mux
from ..common.regfile import RegFile
from ..common.register import Register


class DataPath(Module):
    """TODO(cjdrake): Write docstring."""

    def __init__(self, name: str, parent: Module | None):
        super().__init__(name, parent)

        self.build()
        self.connect()

    def build(self):
        self.data_mem_addr = Bits(name="data_mem_addr", parent=self, shape=(32,))
        self.data_mem_wr_data = Bits(name="data_mem_wr_data", parent=self, shape=(32,))
        self.data_mem_rd_data = Bits(name="data_mem_rd_data", parent=self, shape=(32,))

        self.inst = Struct(name="inst", parent=self, cls=Inst)
        self.pc = Bits(name="pc", parent=self, shape=(32,))

        # Immedate generate
        self.immediate = Bits(name="immediate", parent=self, shape=(32,))

        # PC + 4
        self.pc_plus_4 = Bits(name="pc_plus_4", parent=self, shape=(32,))

        # PC + Immediate
        self.pc_plus_immediate = Bits(name="pc_plus_immediate", parent=self, shape=(32,))

        # Control signals
        self.pc_wr_en = Bit(name="pc_wr_en", parent=self)
        self.regfile_wr_en = Bit(name="regfile_wr_en", parent=self)
        self.alu_op_a_sel = Bit(name="alu_op_a_sel", parent=self)
        self.alu_op_b_sel = Bit(name="alu_op_b_sel", parent=self)
        self.alu_function = Bits(name="alu_function", parent=self, shape=(5,))
        self.reg_writeback_sel = Bits(name="reg_writeback_sel", parent=self, shape=(3,))
        self.next_pc_sel = Bits(name="next_pc_sel", parent=self, shape=(2,))

        # Select ALU Ops
        self.alu_op_a = Bits(name="alu_op_a", parent=self, shape=(32,))
        self.alu_op_b = Bits(name="alu_op_b", parent=self, shape=(32,))

        # ALU Outputs
        self.alu_result = Bits(name="alu_result", parent=self, shape=(32,))
        self.alu_result_equal_zero = Bit(name="alu_result_equal_zero", parent=self)

        # Next PC
        self.pc_next = Bits(name="pc_next", parent=self, shape=(32,))

        # Regfile Write Data
        self.wr_data = Bits(name="wr_data", parent=self, shape=(32,))

        self.clock = Bit(name="clock", parent=self)
        self.reset = Bit(name="reset", parent=self)

        # State
        self.rs1_data = Bits(name="rs1_data", parent=self, shape=(32,))
        self.rs2_data = Bits(name="rs2_data", parent=self, shape=(32,))

        # Submodules
        self.adder_pc_plus_4 = Adder(name="adder_pc_plus_4", parent=self, width=32)
        self.adder_pc_plus_immediate = Adder(name="adder_pc_plus_immediate", parent=self, width=32)
        self.alu = Alu(name="alu", parent=self)
        self.mux_next_pc = Mux(name="mux_next_pc", parent=self, n=4, shape=(32,))
        self.mux_op_a = Mux(name="mux_op_a", parent=self, n=2, shape=(32,))
        self.mux_op_b = Mux(name="mux_op_b", parent=self, n=2, shape=(32,))
        self.mux_reg_writeback = Mux(name="mux_reg_writeback", n=8, parent=self, shape=(32,))
        self.immediate_generator = ImmedateGenerator(name="immediate_generator", parent=self)
        self.program_counter = Register(
            name="program_counter", parent=self, shape=(32,), init=uint2vec(TEXT_BASE, 32)
        )
        self.regfile = RegFile(name="regfile", parent=self)

    def connect(self):
        self.data_mem_addr.connect(self.alu_result)
        self.data_mem_wr_data.connect(self.rs2_data)

        self.pc_plus_4.connect(self.adder_pc_plus_4.result)
        # .op_a(32'h0000_0004)
        self.adder_pc_plus_4.op_b.connect(self.pc)

        self.pc_plus_immediate.connect(self.adder_pc_plus_immediate.result)
        self.adder_pc_plus_immediate.op_a.connect(self.pc)
        self.adder_pc_plus_immediate.op_b.connect(self.immediate)

        self.alu_result.connect(self.alu.result)
        self.alu_result_equal_zero.connect(self.alu.result_equal_zero)
        self.alu.alu_function.connect(self.alu_function)
        self.alu.op_a.connect(self.alu_op_a)
        self.alu.op_b.connect(self.alu_op_b)

        self.pc_next.connect(self.mux_next_pc.out)
        self.mux_next_pc.sel.connect(self.next_pc_sel)
        self.mux_next_pc.ins[0].connect(self.pc_plus_4)
        self.mux_next_pc.ins[1].connect(self.pc_plus_immediate)
        # .in2 ({alu_result[32-1:1], 1'b0})
        # .in3 (32'h0000_0000)

        self.alu_op_a.connect(self.mux_op_a.out)
        self.mux_op_a.sel.connect(self.alu_op_a_sel)
        self.mux_op_a.ins[0].connect(self.rs1_data)
        self.mux_op_a.ins[1].connect(self.pc)

        self.alu_op_b.connect(self.mux_op_b.out)
        self.mux_op_b.sel.connect(self.alu_op_b_sel)
        self.mux_op_b.ins[0].connect(self.rs2_data)
        self.mux_op_b.ins[1].connect(self.immediate)

        self.wr_data.connect(self.mux_reg_writeback.out)
        self.mux_reg_writeback.sel.connect(self.reg_writeback_sel)
        self.mux_reg_writeback.ins[0].connect(self.alu_result)
        self.mux_reg_writeback.ins[1].connect(self.data_mem_rd_data)
        self.mux_reg_writeback.ins[2].connect(self.pc_plus_4)
        self.mux_reg_writeback.ins[3].connect(self.immediate)
        # .in4(32'h0000_0000)
        # .in5(32'h0000_0000)
        # .in6(32'h0000_0000)
        # .in7(32'h0000_0000)

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

    @initial
    async def p_i_0(self):
        self.adder_pc_plus_4.op_a.next = vec("32h0000_0004")
        # mux_next_pc.in3(32'h0000_0000)
        self.mux_next_pc.ins[3].next = zeros(32)
        # mux_reg_writeback.{in4, in5, in6, in7}
        self.mux_reg_writeback.ins[4].next = zeros(32)
        self.mux_reg_writeback.ins[5].next = zeros(32)
        self.mux_reg_writeback.ins[6].next = zeros(32)
        self.mux_reg_writeback.ins[7].next = zeros(32)

    @always_comb
    async def p_c_0(self):
        while True:
            await changed(self.alu_result)
            # mux_next_pc.in2({alu_result[31:1], 1'b0})
            self.mux_next_pc.ins[2].next = cat(zeros(1), self.alu_result.value[1:32])

    @always_comb
    async def p_c_1(self):
        while True:
            await changed(self.inst)
            self.regfile.rs2_addr.next = self.inst.value.rs2
            self.regfile.rs1_addr.next = self.inst.value.rs1
            self.regfile.wr_addr.next = self.inst.value.rd
