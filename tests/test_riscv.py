"""
Test simulation of a simple RiscV core.

The RiscV core design is copied from: https://github.com/tilk/riscv-simple-sv

Work In Progress
The objective is to figure out seqlogic simulation style/semantics.
We are not presently interested in the details of RISC-V.
It merely serves as a non-trivial example design.
"""

# pylint: disable = protected-access
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

from collections import defaultdict

from seqlogic import get_loop, simify
from seqlogic.lbool import ones, uint2vec, vec, xes, zeros

from .riscv.core import AluOp, CtlAluA, CtlAluB, CtlPc, Inst, Opcode
from .riscv.core.singlecycle.top import Top

loop = get_loop()


T = ones(1)
F = zeros(1)
X32 = xes(32)
Z32 = zeros(32)

DEBUG_REG = vec("32hFFFF_FFF0")

PASS, FAIL, TIMEOUT = 0, 1, 2


def get_mem(name: str) -> list[int]:
    text = []
    with open(name, encoding="utf-8") as f:
        for line in f:
            for part in line.split()[1:]:
                text.append(int(part, base=16))
    return text


def test_singlecycle_dump():
    loop.reset()
    waves = defaultdict(dict)

    # Create module hierarchy
    top = Top(name="top")

    top.dump_waves(waves, r"/top/pc")
    top.dump_waves(waves, r"/top/inst")
    top.dump_waves(waves, r"/top/bus_addr")
    top.dump_waves(waves, r"/top/bus_wr_en")
    top.dump_waves(waves, r"/top/bus_wr_data")
    top.dump_waves(waves, r"/top/bus_rd_en")
    top.dump_waves(waves, r"/top.core/datapath.inst[^/]*")
    top.dump_waves(waves, r"/top/core/datapath.immediate")
    top.dump_waves(waves, r"/top/core/datapath.alu[^/]*")
    top.dump_waves(waves, r"/top/core/datapath.pc[^/]*")
    top.dump_waves(waves, r"/top/core/datapath.reg_writeback_sel")
    top.dump_waves(waves, r"/top/core/datapath.next_pc_sel")
    top.dump_waves(waves, r"/top/core/datapath.regfile_wr_en")
    top.dump_waves(waves, r"/top/core/datapath.wr_data")
    top.dump_waves(waves, r"/top/core/datapath.rs[12]_data")
    top.dump_waves(waves, r"/top/core/datapath.data_mem_addr")
    top.dump_waves(waves, r"/top/core/datapath.data_mem_wr_data")

    simify(top)

    # Initialize instruction memory
    text = get_mem("tests/riscv/tests/add.text")
    for i, d in enumerate(text):
        top.text_memory_bus.text_memory._mem.set_next(i, uint2vec(d, 32))

    loop.run(until=50)

    exp = {
        # Initialize everything to X'es
        -1: {
            # Top
            top._pc: X32,
            top._inst: Inst(),
            top.bus_addr: X32,
            top.bus_wr_en: xes(1),
            top.bus_wr_data: X32,
            top.bus_rd_en: xes(1),
            # Decode
            top.core.datapath.inst: Inst(),
            top.core.datapath.immediate: X32,
            # Control
            top.core.datapath.alu_op_a_sel: CtlAluA.X,
            top.core.datapath.alu_op_b_sel: CtlAluB.X,
            top.core.datapath.reg_writeback_sel: xes(3),
            # ALU
            top.core.datapath.alu_result: X32,
            top.core.datapath.alu_result_equal_zero: xes(1),
            top.core.datapath.alu_function: AluOp.X,
            top.core.datapath.alu_op_a: X32,
            top.core.datapath.alu_op_b: X32,
            # PC
            top.core.datapath.pc_next: X32,
            top.core.datapath.next_pc_sel: CtlPc.X,
            top.core.datapath.pc_plus_4: X32,
            top.core.datapath.pc_plus_immediate: X32,
            top.core.datapath.pc_wr_en: xes(1),
            top.core.datapath.pc: X32,
            # Regfile
            top.core.datapath.regfile_wr_en: xes(1),
            top.core.datapath.wr_data: X32,
            top.core.datapath.rs1_data: X32,
            top.core.datapath.rs2_data: X32,
            # Data Mem
            top.core.datapath.data_mem_addr: X32,
            top.core.datapath.data_mem_wr_data: X32,
        },
        # @(posedge reset)
        5: {
            top._pc: vec("32h0040_0000"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: Z32,
            top.bus_wr_en: F,
            top.bus_wr_data: Z32,
            top.bus_rd_en: F,
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: Z32,
            # Control
            top.core.datapath.alu_op_a_sel: CtlAluA.RS1,
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            top.core.datapath.reg_writeback_sel: vec("3b000"),
            # ALU
            top.core.datapath.alu_result: Z32,
            top.core.datapath.alu_result_equal_zero: T,
            top.core.datapath.alu_function: AluOp.ADD,
            top.core.datapath.alu_op_a: Z32,
            top.core.datapath.alu_op_b: Z32,
            # PC
            top.core.datapath.pc_next: vec("32h0040_0004"),
            top.core.datapath.next_pc_sel: CtlPc.PC4,
            top.core.datapath.pc_plus_4: vec("32h0040_0004"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0000"),
            top.core.datapath.pc_wr_en: T,
            top.core.datapath.pc: vec("32h0040_0000"),
            # Regfile
            top.core.datapath.regfile_wr_en: T,
            top.core.datapath.wr_data: Z32,
            top.core.datapath.rs1_data: Z32,
            top.core.datapath.rs2_data: Z32,
            # Data Mem
            top.core.datapath.data_mem_addr: Z32,
            top.core.datapath.data_mem_wr_data: Z32,
        },
        # @(posedge clock)
        11: {
            top._pc: vec("32h0040_0004"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0008"),
            top.core.datapath.pc_plus_4: vec("32h0040_0008"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0004"),
            top.core.datapath.pc: vec("32h0040_0004"),
            # Regfile
        },
        # @(posedge clock)
        13: {
            top._pc: vec("32h0040_0008"),
            top._inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # PC
            top.core.datapath.pc_next: vec("32h0040_000C"),
            top.core.datapath.pc_plus_4: vec("32h0040_000C"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0008"),
            top.core.datapath.pc: vec("32h0040_0008"),
        },
        # @(posedge clock)
        15: {
            top._pc: vec("32h0040_000C"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # PC
            top.core.datapath.pc_next: vec("32h0040_0010"),
            top.core.datapath.pc_plus_4: vec("32h0040_0010"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_000C"),
            top.core.datapath.pc: vec("32h0040_000C"),
        },
        # @(posedge clock)
        17: {
            top._pc: vec("32h0040_0010"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0002"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0002"),
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0002"),
            top.core.datapath.alu_result_equal_zero: F,
            top.core.datapath.alu_op_b: vec("32h0000_0002"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0014"),
            top.core.datapath.pc_plus_4: vec("32h0040_0014"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0012"),
            top.core.datapath.pc: vec("32h0040_0010"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_0002"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0002"),
        },
        # @(posedge clock)
        19: {
            top._pc: vec("32h0040_0014"),
            top._inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b01100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100110"),
            ),
            top.bus_addr: vec("32h0000_0001"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b01100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100110"),
            ),
            top.core.datapath.immediate: vec("32h0000_04CC"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0001"),
            top.core.datapath.alu_function: AluOp.SEQ,
            top.core.datapath.alu_op_b: Z32,
            # PC
            top.core.datapath.pc_next: vec("32h0040_0018"),
            top.core.datapath.pc_plus_4: vec("32h0040_0018"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_04E0"),
            top.core.datapath.pc: vec("32h0040_0014"),
            # Regfile
            top.core.datapath.regfile_wr_en: F,
            top.core.datapath.wr_data: vec("32h0000_0001"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0001"),
        },
        # @(posedge clock)
        21: {
            top._pc: vec("32h0040_0018"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00001"),
                funct7=vec("7b0000000"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00001"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0001"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_function: AluOp.ADD,
            top.core.datapath.alu_op_b: vec("32h0000_0001"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_001C"),
            top.core.datapath.pc_plus_4: vec("32h0040_001C"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0019"),
            top.core.datapath.pc: vec("32h0040_0018"),
            # Regfile
            top.core.datapath.regfile_wr_en: T,
        },
        # @(posedge clock)
        23: {
            top._pc: vec("32h0040_001C"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00001"),
                funct7=vec("7b0000000"),
            ),
            top.bus_wr_data: vec("32h0000_0100"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00001"),
                funct7=vec("7b0000000"),
            ),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0020"),
            top.core.datapath.pc_plus_4: vec("32h0040_0020"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_001D"),
            top.core.datapath.pc: vec("32h0040_001C"),
            # Regfile
            top.core.datapath.rs2_data: vec("32h0000_0001"),
            # Data Mem
            top.core.datapath.data_mem_wr_data: vec("32h0000_0001"),
        },
        # @(posedge clock)
        25: {
            top._pc: vec("32h0040_0020"),
            top._inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0002"),
            top.bus_wr_data: vec("32h0001_0000"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: Z32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0002"),
            top.core.datapath.alu_op_a: vec("32h0000_0001"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0024"),
            top.core.datapath.pc_plus_4: vec("32h0040_0024"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0020"),
            top.core.datapath.pc: vec("32h0040_0020"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_0002"),
            top.core.datapath.rs1_data: vec("32h0000_0001"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0002"),
        },
        # @(posedge clock)
        27: {
            top._pc: vec("32h0040_0024"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0002"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_op_a: Z32,
            top.core.datapath.alu_op_b: vec("32h0000_0002"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0028"),
            top.core.datapath.pc_plus_4: vec("32h0040_0028"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0026"),
            top.core.datapath.pc: vec("32h0040_0024"),
            # Regfile
            top.core.datapath.rs1_data: Z32,
        },
        # @(posedge clock)
        29: {
            top._pc: vec("32h0040_0028"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00011"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0003"),
            top.bus_wr_data: vec("32h0200_0000"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00011"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0003"),
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0003"),
            top.core.datapath.alu_op_b: vec("32h0000_0003"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_002C"),
            top.core.datapath.pc_plus_4: vec("32h0040_002C"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_002B"),
            top.core.datapath.pc: vec("32h0040_0028"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_0003"),
            top.core.datapath.rs2_data: vec("32h0000_0002"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0003"),
            top.core.datapath.data_mem_wr_data: vec("32h0000_0002"),
        },
        # @(posedge clock)
        31: {
            top._pc: vec("32h0040_002C"),
            top._inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b10100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100101"),
            ),
            top.bus_addr: vec("32h0000_0001"),
            top.bus_wr_data: vec("32h0000_0200"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b10100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100101"),
            ),
            top.core.datapath.immediate: vec("32h0000_04B4"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0001"),
            top.core.datapath.alu_function: AluOp.SEQ,
            top.core.datapath.alu_op_a: vec("32h0000_0002"),
            top.core.datapath.alu_op_b: vec("32h0000_0002"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0030"),
            top.core.datapath.pc_plus_4: vec("32h0040_0030"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_04E0"),
            top.core.datapath.pc: vec("32h0040_002C"),
            # Regfile
            top.core.datapath.regfile_wr_en: F,
            top.core.datapath.wr_data: vec("32h0000_0001"),
            top.core.datapath.rs1_data: vec("32h0000_0002"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0001"),
        },
        # @(posedge clock)
        33: {
            top._pc: vec("32h0040_0030"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00011"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0003"),
            top.bus_wr_data: vec("32h0200_0000"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00011"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0003"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0003"),
            top.core.datapath.alu_function: AluOp.ADD,
            top.core.datapath.alu_op_a: Z32,
            top.core.datapath.alu_op_b: vec("32h0000_0003"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0034"),
            top.core.datapath.pc_plus_4: vec("32h0040_0034"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0033"),
            top.core.datapath.pc: vec("32h0040_0030"),
            # Regfile
            top.core.datapath.regfile_wr_en: T,
            top.core.datapath.wr_data: vec("32h0000_0003"),
            top.core.datapath.rs1_data: Z32,
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0003"),
        },
        # @(posedge clock)
        35: {
            top._pc: vec("32h0040_0034"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00111"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0007"),
            top.bus_wr_data: Z32,
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00111"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0007"),
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0007"),
            top.core.datapath.alu_op_b: vec("32h0000_0007"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0038"),
            top.core.datapath.pc_plus_4: vec("32h0040_0038"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_003B"),
            top.core.datapath.pc: vec("32h0040_0034"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_0007"),
            top.core.datapath.rs2_data: Z32,
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0007"),
            top.core.datapath.data_mem_wr_data: Z32,
        },
        # @(posedge clock)
        37: {
            top._pc: vec("32h0040_0038"),
            top._inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_000A"),
            top.bus_wr_data: vec("32h0007_0000"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: Z32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_000A"),
            top.core.datapath.alu_op_a: vec("32h0000_0003"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_003C"),
            top.core.datapath.pc_plus_4: vec("32h0040_003C"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0038"),
            top.core.datapath.pc: vec("32h0040_0038"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_000A"),
            top.core.datapath.rs1_data: vec("32h0000_0003"),
            top.core.datapath.rs2_data: vec("32h0000_0007"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_000A"),
            top.core.datapath.data_mem_wr_data: vec("32h0000_0007"),
        },
        # @(posedge clock)
        39: {
            top._pc: vec("32h0040_003C"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b01010"),
                funct7=vec("7b0000000"),
            ),
            top.bus_wr_data: Z32,
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11101"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b01010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_000A"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_op_a: Z32,
            top.core.datapath.alu_op_b: vec("32h0000_000A"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0040"),
            top.core.datapath.pc_plus_4: vec("32h0040_0040"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0046"),
            top.core.datapath.pc: vec("32h0040_003C"),
            # Regfile
            top.core.datapath.rs1_data: Z32,
            top.core.datapath.rs2_data: Z32,
            # Data Mem
            top.core.datapath.data_mem_wr_data: Z32,
        },
        # @(posedge clock)
        41: {
            top._pc: vec("32h0040_0040"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00100"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32h0000_0004"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b11100"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00100"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: vec("32h0000_0004"),
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0004"),
            top.core.datapath.alu_op_b: vec("32h0000_0004"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0044"),
            top.core.datapath.pc_plus_4: vec("32h0040_0044"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0044"),
            top.core.datapath.pc: vec("32h0040_0040"),
            # Regfile
            top.core.datapath.wr_data: vec("32h0000_0004"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0004"),
        },
        # @(posedge clock)
        43: {
            top._pc: vec("32h0040_0044"),
            top._inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b11100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100100"),
            ),
            top.bus_addr: vec("32h0000_0001"),
            top.bus_wr_data: vec("32h0000_0A00"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.BRANCH,
                rd=vec("5b11100"),
                funct3=vec("3b001"),
                rs1=vec("5b00011"),
                rs2=vec("5b11101"),
                funct7=vec("7b0100100"),
            ),
            top.core.datapath.immediate: vec("32h0000_049C"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: vec("32h0000_0001"),
            top.core.datapath.alu_function: AluOp.SEQ,
            top.core.datapath.alu_op_a: vec("32h0000_000A"),
            top.core.datapath.alu_op_b: vec("32h0000_000A"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0048"),
            top.core.datapath.pc_plus_4: vec("32h0040_0048"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_04E0"),
            top.core.datapath.pc: vec("32h0040_0044"),
            # Regfile
            top.core.datapath.regfile_wr_en: F,
            top.core.datapath.wr_data: vec("32h0000_0001"),
            top.core.datapath.rs1_data: vec("32h0000_000A"),
            top.core.datapath.rs2_data: vec("32h0000_000A"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32h0000_0001"),
            top.core.datapath.data_mem_wr_data: vec("32h0000_000A"),
        },
        # @(posedge clock)
        45: {
            top._pc: vec("32h0040_0048"),
            top._inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: Z32,
            top.bus_wr_data: Z32,
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd=vec("5b00001"),
                funct3=vec("3b000"),
                rs1=vec("5b00000"),
                rs2=vec("5b00000"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: Z32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_result: Z32,
            top.core.datapath.alu_result_equal_zero: T,
            top.core.datapath.alu_function: AluOp.ADD,
            top.core.datapath.alu_op_a: Z32,
            top.core.datapath.alu_op_b: Z32,
            # PC
            top.core.datapath.pc_next: vec("32h0040_004C"),
            top.core.datapath.pc_plus_4: vec("32h0040_004C"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0048"),
            top.core.datapath.pc: vec("32h0040_0048"),
            # Regfile
            top.core.datapath.regfile_wr_en: T,
            top.core.datapath.wr_data: Z32,
            top.core.datapath.rs1_data: Z32,
            top.core.datapath.rs2_data: Z32,
            # Data Mem
            top.core.datapath.data_mem_addr: Z32,
            top.core.datapath.data_mem_wr_data: Z32,
        },
        # @(posedge clock)
        47: {
            top._pc: vec("32h0040_004C"),
            top._inst: Inst(
                opcode=Opcode.LUI,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b11111"),
                rs2=vec("5b11111"),
                funct7=vec("7b1111111"),
            ),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.LUI,
                rd=vec("5b00010"),
                funct3=vec("3b000"),
                rs1=vec("5b11111"),
                rs2=vec("5b11111"),
                funct7=vec("7b1111111"),
            ),
            top.core.datapath.immediate: vec("32hFFFF_8000"),
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            top.core.datapath.reg_writeback_sel: vec("3b011"),
            # PC
            top.core.datapath.pc_next: vec("32h0040_0050"),
            top.core.datapath.pc_plus_4: vec("32h0040_0050"),
            top.core.datapath.pc_plus_immediate: vec("32h003F_804C"),
            top.core.datapath.pc: vec("32h0040_004C"),
            # Regfile
            top.core.datapath.wr_data: vec("32hFFFF_8000"),
        },
        # @(posedge clock)
        49: {
            top._pc: vec("32h0040_0050"),
            top._inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.bus_addr: vec("32hFFFF_8000"),
            top.bus_wr_data: vec("32hFFFF_8000"),
            # Decode
            top.core.datapath.inst: Inst(
                opcode=Opcode.OP,
                rd=vec("5b00011"),
                funct3=vec("3b000"),
                rs1=vec("5b00001"),
                rs2=vec("5b00010"),
                funct7=vec("7b0000000"),
            ),
            top.core.datapath.immediate: Z32,
            # Control
            top.core.datapath.reg_writeback_sel: vec("3b000"),
            # ALU
            top.core.datapath.alu_result: vec("32hFFFF_8000"),
            top.core.datapath.alu_result_equal_zero: F,
            top.core.datapath.alu_op_b: vec("32hFFFF_8000"),
            # Next PC
            top.core.datapath.pc_next: vec("32h0040_0054"),
            top.core.datapath.pc_plus_4: vec("32h0040_0054"),
            top.core.datapath.pc_plus_immediate: vec("32h0040_0050"),
            top.core.datapath.pc: vec("32h0040_0050"),
            # Regfile
            top.core.datapath.rs2_data: vec("32hFFFF_8000"),
            # Data Mem
            top.core.datapath.data_mem_addr: vec("32hFFFF_8000"),
            top.core.datapath.data_mem_wr_data: vec("32hFFFF_8000"),
        },
    }
    assert waves == exp


def run_riscv_test(name: str) -> int:
    loop.reset()

    # Create module hierarchy
    top = Top(name="top")

    simify(top)

    # Initialize instruction memory
    text = get_mem(f"tests/riscv/tests/{name}.text")
    for i, d in enumerate(text):
        top.text_memory_bus.text_memory._mem.set_next(i, uint2vec(d, 32))

    # Initialize data memory
    data = get_mem(f"tests/riscv/tests/{name}.data")
    for i, d in enumerate(data):
        top.data_memory_bus.data_memory._mem.set_next(i, uint2vec(d, 32))

    # Run the simulation
    for _ in loop.iter(until=10000):
        if top.bus_wr_en.value == T and top.bus_addr.value == DEBUG_REG:
            if top.bus_wr_data.value == vec("32h0000_0001"):
                return PASS
            else:
                return FAIL
    return TIMEOUT


def test_singlecycle_add():
    assert run_riscv_test("add") == PASS


def test_singlecycle_addi():
    assert run_riscv_test("addi") == PASS


def test_singlecycle_and():
    assert run_riscv_test("and") == PASS


def test_singlecycle_andi():
    assert run_riscv_test("andi") == PASS


def test_singlecycle_auipc():
    assert run_riscv_test("auipc") == PASS


def test_singlecycle_beq():
    assert run_riscv_test("beq") == PASS


def test_singlecycle_bge():
    assert run_riscv_test("bge") == PASS


def test_singlecycle_bgeu():
    assert run_riscv_test("bgeu") == PASS


def test_singlecycle_blt():
    assert run_riscv_test("blt") == PASS


def test_singlecycle_bltu():
    assert run_riscv_test("bltu") == PASS


def test_singlecycle_bne():
    assert run_riscv_test("bne") == PASS


def test_singlecycle_jal():
    assert run_riscv_test("jal") == PASS


def test_singlecycle_jalr():
    assert run_riscv_test("jalr") == PASS


def test_singlecycle_lb():
    assert run_riscv_test("lb") == PASS


def test_singlecycle_lbu():
    assert run_riscv_test("lbu") == PASS


def test_singlecycle_lh():
    assert run_riscv_test("lh") == PASS


def test_singlecycle_lhu():
    assert run_riscv_test("lhu") == PASS


def test_singlecycle_lui():
    assert run_riscv_test("lui") == PASS


def test_singlecycle_lw():
    assert run_riscv_test("lw") == PASS


def test_singlecycle_or():
    assert run_riscv_test("or") == PASS


def test_singlecycle_ori():
    assert run_riscv_test("ori") == PASS


def test_singlecycle_sb():
    assert run_riscv_test("sb") == PASS


def test_singlecycle_sh():
    assert run_riscv_test("sh") == PASS


def test_singlecycle_simple():
    assert run_riscv_test("simple") == PASS


def test_singlecycle_sll():
    assert run_riscv_test("sll") == PASS


def test_singlecycle_slli():
    assert run_riscv_test("slli") == PASS


def test_singlecycle_slt():
    assert run_riscv_test("slt") == PASS


def test_singlecycle_slti():
    assert run_riscv_test("slti") == PASS


def test_singlecycle_sltiu():
    assert run_riscv_test("sltiu") == PASS


def test_singlecycle_sltu():
    assert run_riscv_test("sltu") == PASS


def test_singlecycle_sra():
    assert run_riscv_test("sra") == PASS


def test_singlecycle_srai():
    assert run_riscv_test("srai") == PASS


def test_singlecycle_srl():
    assert run_riscv_test("srl") == PASS


def test_singlecycle_srli():
    assert run_riscv_test("srli") == PASS


def test_singlecycle_sub():
    assert run_riscv_test("sub") == PASS


def test_singlecycle_sw():
    assert run_riscv_test("sw") == PASS


def test_singlecycle_xor():
    assert run_riscv_test("xor") == PASS


def test_singlecycle_xori():
    assert run_riscv_test("xori") == PASS
