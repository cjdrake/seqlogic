"""
Test simulation of a simple RiscV core.

The RiscV core design is copied from: https://github.com/tilk/riscv-simple-sv

Work In Progress
The objective is to figure out seqlogic simulation style/semantics.
We are not presently interested in the details of RISC-V.
It merely serves as a non-trivial example design.
"""

# This tracing method requires cross module references to _protected logic
# pylint: disable=protected-access

from collections import defaultdict

from seqlogic import irun, run, u2bv

from .riscv.core import (
    AluOp,
    CtlAluA,
    CtlAluB,
    CtlPc,
    CtlWriteBack,
    Funct3AluLogic,
    Funct3Branch,
    Inst,
    Opcode,
)
from .riscv.core.top import Top

X32 = "32bXXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX"
W32 = "32b----_----_----_----_----_----_----_----"

DEBUG_REG = "32hFFFF_FFF0"

PASS, FAIL, TIMEOUT = 0, 1, 2


def get_mem(name: str) -> list[int]:
    text = []
    with open(name, encoding="utf-8") as f:
        for line in f:
            for part in line.split()[1:]:
                text.append(int(part, base=16))
    return text


def test_dump():
    # Create module hierarchy
    top = Top(name="top")

    waves = defaultdict(dict)
    top.dump_waves(waves, r"/top/pc")
    top.dump_waves(waves, r"/top/inst")
    top.dump_waves(waves, r"/top/bus_addr")
    top.dump_waves(waves, r"/top/bus_wr_en")
    top.dump_waves(waves, r"/top/bus_wr_data")
    top.dump_waves(waves, r"/top/bus_rd_en")
    top.dump_waves(waves, r"/top/core/datapath.immediate")
    top.dump_waves(waves, r"/top/core/datapath.alu[^/]*")
    top.dump_waves(waves, r"/top/core/datapath.pc_next")
    top.dump_waves(waves, r"/top/core/datapath.pc_plus_4")
    top.dump_waves(waves, r"/top/core/datapath.pc_plus_immediate")
    top.dump_waves(waves, r"/top/core/datapath.pc_wr_en")
    top.dump_waves(waves, r"/top/core/datapath.reg_wr_sel")
    top.dump_waves(waves, r"/top/core/datapath.next_pc_sel")
    top.dump_waves(waves, r"/top/core/datapath.reg_wr_en")
    top.dump_waves(waves, r"/top/core/datapath.wr_data")
    top.dump_waves(waves, r"/top/core/datapath.rs[12]_data")
    top.dump_waves(waves, r"/top/core/datapath.data_mem_addr")
    top.dump_waves(waves, r"/top/core/datapath.data_mem_wr_data")

    # Initialize instruction memory
    async def main():
        await top.main()

        text = get_mem("tests/riscv/tests/add.text")
        for i, d in enumerate(text):
            addr = u2bv(i, 10)
            data = u2bv(d, 32)
            top.text_mem_bus.text_mem.mem[addr].next = data

    run(main(), until=50)

    exp = {
        # Initialize everything to X'es
        -1: {
            # Top
            top.pc: X32,
            top.inst: Inst(),
            top.bus_addr: X32,
            top.bus_wr_en: "1bX",
            top.bus_wr_data: X32,
            top.bus_rd_en: "1bX",
            # Decode
            top.core.datapath.immediate: X32,
            # Control
            top.core.datapath.alu_op_a_sel: CtlAluA.X,
            top.core.datapath.alu_op_b_sel: CtlAluB.X,
            top.core.datapath.reg_wr_sel: CtlWriteBack.X,
            # ALU
            top.core.datapath.alu_result: X32,
            top.core.datapath.alu_result_eq_zero: "1bX",
            top.core.datapath.alu_op: AluOp.X,
            top.core.datapath.alu_op_a: X32,
            top.core.datapath.alu_op_b: X32,
            # PC
            top.core.datapath.pc_next: X32,
            top.core.datapath.next_pc_sel: CtlPc.X,
            top.core.datapath.pc_plus_4: X32,
            top.core.datapath.pc_plus_immediate: X32,
            top.core.datapath.pc_wr_en: "1bX",
            # Regfile
            top.core.datapath.reg_wr_en: "1bX",
            top.core.datapath.wr_data: X32,
            top.core.datapath.rs1_data: X32,
            top.core.datapath.rs2_data: X32,
            # Data Mem
            top.core.datapath.data_mem_addr: X32,
            top.core.datapath.data_mem_wr_data: X32,
        },
        # @(posedge reset)
        5: {
            top.pc: "32h0040_0000",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00001",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00000",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0000",
            top.bus_wr_en: "1b0",
            top.bus_wr_data: "32h0000_0000",
            top.bus_rd_en: "1b0",
            # Decode
            top.core.datapath.immediate: "32h0000_0000",
            # Control
            top.core.datapath.alu_op_a_sel: CtlAluA.RS1,
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            top.core.datapath.reg_wr_sel: CtlWriteBack.ALU,
            # ALU
            top.core.datapath.alu_result: "32h0000_0000",
            top.core.datapath.alu_result_eq_zero: "1b1",
            top.core.datapath.alu_op: AluOp.ADD,
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32h0000_0000",
            # PC
            top.core.datapath.pc_next: "32h0040_0004",
            top.core.datapath.next_pc_sel: CtlPc.PC4,
            top.core.datapath.pc_plus_4: "32h0040_0004",
            top.core.datapath.pc_plus_immediate: "32h0040_0000",
            top.core.datapath.pc_wr_en: "1b1",
            # Regfile
            top.core.datapath.reg_wr_en: "1b1",
            top.core.datapath.wr_data: "32h0000_0000",
            top.core.datapath.rs1_data: "32h0000_0000",
            top.core.datapath.rs2_data: "32h0000_0000",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0000",
            top.core.datapath.data_mem_wr_data: "32h0000_0000",
        },
        # @(posedge clock)
        11: {
            top.pc: "32h0040_0004",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00010",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00000",
                funct7="7b0000000",
            ),
            # PC
            top.core.datapath.pc_next: "32h0040_0008",
            top.core.datapath.pc_plus_4: "32h0040_0008",
            top.core.datapath.pc_plus_immediate: "32h0040_0004",
        },
        # @(posedge clock)
        13: {
            top.pc: "32h0040_0008",
            top.inst: Inst(
                opcode=Opcode.OP,
                rd="5b00011",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00001",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            # Decode
            top.core.datapath.immediate: W32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # PC
            top.core.datapath.pc_next: "32h0040_000C",
            top.core.datapath.pc_plus_4: "32h0040_000C",
            top.core.datapath.pc_plus_immediate: W32,
        },
        # @(posedge clock)
        15: {
            top.pc: "32h0040_000C",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11101",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00000",
                funct7="7b0000000",
            ),
            # Decode
            top.core.datapath.immediate: "32h0000_0000",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # PC
            top.core.datapath.pc_next: "32h0040_0010",
            top.core.datapath.pc_plus_4: "32h0040_0010",
            top.core.datapath.pc_plus_immediate: "32h0040_000C",
        },
        # @(posedge clock)
        17: {
            top.pc: "32h0040_0010",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11100",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0002",
            # Decode
            top.core.datapath.immediate: "32h0000_0002",
            # ALU
            top.core.datapath.alu_result: "32h0000_0002",
            top.core.datapath.alu_result_eq_zero: "1b0",
            top.core.datapath.alu_op_b: "32h0000_0002",
            # PC
            top.core.datapath.pc_next: "32h0040_0014",
            top.core.datapath.pc_plus_4: "32h0040_0014",
            top.core.datapath.pc_plus_immediate: "32h0040_0012",
            # Regfile
            top.core.datapath.wr_data: "32h0000_0002",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0002",
        },
        # @(posedge clock)
        19: {
            top.pc: "32h0040_0014",
            top.inst: Inst(
                opcode=Opcode.BRANCH,
                rd="5b01100",
                funct3=Funct3Branch.NE,
                rs1="5b00011",
                rs2="5b11101",
                funct7="7b0100110",
            ),
            top.bus_addr: "32h0000_0001",
            # Decode
            top.core.datapath.immediate: "32h0000_04CC",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: "32h0000_0001",
            top.core.datapath.alu_op: AluOp.SEQ,
            top.core.datapath.alu_op_b: "32h0000_0000",
            # PC
            top.core.datapath.pc_next: "32h0040_0018",
            top.core.datapath.pc_plus_4: "32h0040_0018",
            top.core.datapath.pc_plus_immediate: "32h0040_04E0",
            # Regfile
            top.core.datapath.reg_wr_en: "1b0",
            top.core.datapath.wr_data: "32h0000_0001",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0001",
        },
        # @(posedge clock)
        21: {
            top.pc: "32h0040_0018",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00001",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00001",
                funct7="7b0000000",
            ),
            # Decode
            top.core.datapath.immediate: "32h0000_0001",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_op: AluOp.ADD,
            top.core.datapath.alu_op_b: "32h0000_0001",
            # PC
            top.core.datapath.pc_next: "32h0040_001C",
            top.core.datapath.pc_plus_4: "32h0040_001C",
            top.core.datapath.pc_plus_immediate: "32h0040_0019",
            # Regfile
            top.core.datapath.reg_wr_en: "1b1",
        },
        # @(posedge clock)
        23: {
            top.pc: "32h0040_001C",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00010",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00001",
                funct7="7b0000000",
            ),
            top.bus_wr_data: "32h0000_0100",
            # PC
            top.core.datapath.pc_next: "32h0040_0020",
            top.core.datapath.pc_plus_4: "32h0040_0020",
            top.core.datapath.pc_plus_immediate: "32h0040_001D",
            # Regfile
            top.core.datapath.rs2_data: "32h0000_0001",
            # Data Mem
            top.core.datapath.data_mem_wr_data: "32h0000_0001",
        },
        # @(posedge clock)
        25: {
            top.pc: "32h0040_0020",
            top.inst: Inst(
                opcode=Opcode.OP,
                rd="5b00011",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00001",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0002",
            top.bus_wr_data: "32h0001_0000",
            # Decode
            top.core.datapath.immediate: W32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: "32h0000_0002",
            top.core.datapath.alu_op_a: "32h0000_0001",
            # PC
            top.core.datapath.pc_next: "32h0040_0024",
            top.core.datapath.pc_plus_4: "32h0040_0024",
            top.core.datapath.pc_plus_immediate: W32,
            # Regfile
            top.core.datapath.wr_data: "32h0000_0002",
            top.core.datapath.rs1_data: "32h0000_0001",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0002",
        },
        # @(posedge clock)
        27: {
            top.pc: "32h0040_0024",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11101",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            # Decode
            top.core.datapath.immediate: "32h0000_0002",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32h0000_0002",
            # PC
            top.core.datapath.pc_next: "32h0040_0028",
            top.core.datapath.pc_plus_4: "32h0040_0028",
            top.core.datapath.pc_plus_immediate: "32h0040_0026",
            # Regfile
            top.core.datapath.rs1_data: "32h0000_0000",
        },
        # @(posedge clock)
        29: {
            top.pc: "32h0040_0028",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11100",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00011",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0003",
            top.bus_wr_data: "32h0200_0000",
            # Decode
            top.core.datapath.immediate: "32h0000_0003",
            # ALU
            top.core.datapath.alu_result: "32h0000_0003",
            top.core.datapath.alu_op_b: "32h0000_0003",
            # PC
            top.core.datapath.pc_next: "32h0040_002C",
            top.core.datapath.pc_plus_4: "32h0040_002C",
            top.core.datapath.pc_plus_immediate: "32h0040_002B",
            # Regfile
            top.core.datapath.wr_data: "32h0000_0003",
            top.core.datapath.rs2_data: "32h0000_0002",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0003",
            top.core.datapath.data_mem_wr_data: "32h0000_0002",
        },
        # @(posedge clock)
        31: {
            top.pc: "32h0040_002C",
            top.inst: Inst(
                opcode=Opcode.BRANCH,
                rd="5b10100",
                funct3=Funct3Branch.NE,
                rs1="5b00011",
                rs2="5b11101",
                funct7="7b0100101",
            ),
            top.bus_addr: "32h0000_0001",
            top.bus_wr_data: "32h0000_0200",
            # Decode
            top.core.datapath.immediate: "32h0000_04B4",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: "32h0000_0001",
            top.core.datapath.alu_op: AluOp.SEQ,
            top.core.datapath.alu_op_a: "32h0000_0002",
            top.core.datapath.alu_op_b: "32h0000_0002",
            # PC
            top.core.datapath.pc_next: "32h0040_0030",
            top.core.datapath.pc_plus_4: "32h0040_0030",
            top.core.datapath.pc_plus_immediate: "32h0040_04E0",
            # Regfile
            top.core.datapath.reg_wr_en: "1b0",
            top.core.datapath.wr_data: "32h0000_0001",
            top.core.datapath.rs1_data: "32h0000_0002",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0001",
        },
        # @(posedge clock)
        33: {
            top.pc: "32h0040_0030",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00001",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00011",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0003",
            top.bus_wr_data: "32h0200_0000",
            # Decode
            top.core.datapath.immediate: "32h0000_0003",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_result: "32h0000_0003",
            top.core.datapath.alu_op: AluOp.ADD,
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32h0000_0003",
            # PC
            top.core.datapath.pc_next: "32h0040_0034",
            top.core.datapath.pc_plus_4: "32h0040_0034",
            top.core.datapath.pc_plus_immediate: "32h0040_0033",
            # Regfile
            top.core.datapath.reg_wr_en: "1b1",
            top.core.datapath.wr_data: "32h0000_0003",
            top.core.datapath.rs1_data: "32h0000_0000",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0003",
        },
        # @(posedge clock)
        35: {
            top.pc: "32h0040_0034",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00010",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00111",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0007",
            top.bus_wr_data: "32bXXXXXXXX_00000000_00000000_00000000",
            # Decode
            top.core.datapath.immediate: "32h0000_0007",
            # ALU
            top.core.datapath.alu_result: "32h0000_0007",
            top.core.datapath.alu_op_b: "32h0000_0007",
            # PC
            top.core.datapath.pc_next: "32h0040_0038",
            top.core.datapath.pc_plus_4: "32h0040_0038",
            top.core.datapath.pc_plus_immediate: "32h0040_003B",
            # Regfile
            top.core.datapath.wr_data: "32h0000_0007",
            top.core.datapath.rs2_data: X32,
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0007",
            top.core.datapath.data_mem_wr_data: X32,
        },
        # @(posedge clock)
        37: {
            top.pc: "32h0040_0038",
            top.inst: Inst(
                opcode=Opcode.OP,
                rd="5b00011",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00001",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_000A",
            top.bus_wr_data: "32h0007_0000",
            # Decode
            top.core.datapath.immediate: W32,
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: "32h0000_000A",
            top.core.datapath.alu_op_a: "32h0000_0003",
            # PC
            top.core.datapath.pc_next: "32h0040_003C",
            top.core.datapath.pc_plus_4: "32h0040_003C",
            top.core.datapath.pc_plus_immediate: W32,
            # Regfile
            top.core.datapath.wr_data: "32h0000_000A",
            top.core.datapath.rs1_data: "32h0000_0003",
            top.core.datapath.rs2_data: "32h0000_0007",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_000A",
            top.core.datapath.data_mem_wr_data: "32h0000_0007",
        },
        # @(posedge clock)
        39: {
            top.pc: "32h0040_003C",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11101",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b01010",
                funct7="7b0000000",
            ),
            top.bus_wr_data: "32bXXXXXXXX_XXXXXXXX_00000000_00000000",
            # Decode
            top.core.datapath.immediate: "32h0000_000A",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32h0000_000A",
            # PC
            top.core.datapath.pc_next: "32h0040_0040",
            top.core.datapath.pc_plus_4: "32h0040_0040",
            top.core.datapath.pc_plus_immediate: "32h0040_0046",
            # Regfile
            top.core.datapath.rs1_data: "32h0000_0000",
            top.core.datapath.rs2_data: X32,
            # Data Mem
            top.core.datapath.data_mem_wr_data: X32,
        },
        # @(posedge clock)
        41: {
            top.pc: "32h0040_0040",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b11100",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00100",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0004",
            top.bus_wr_data: X32,
            # Decode
            top.core.datapath.immediate: "32h0000_0004",
            # ALU
            top.core.datapath.alu_result: "32h0000_0004",
            top.core.datapath.alu_op_b: "32h0000_0004",
            # PC
            top.core.datapath.pc_next: "32h0040_0044",
            top.core.datapath.pc_plus_4: "32h0040_0044",
            top.core.datapath.pc_plus_immediate: "32h0040_0044",
            # Regfile
            top.core.datapath.wr_data: "32h0000_0004",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0004",
        },
        # @(posedge clock)
        43: {
            top.pc: "32h0040_0044",
            top.inst: Inst(
                opcode=Opcode.BRANCH,
                rd="5b11100",
                funct3=Funct3Branch.NE,
                rs1="5b00011",
                rs2="5b11101",
                funct7="7b0100100",
            ),
            top.bus_addr: "32h0000_0001",
            top.bus_wr_data: "32h0000_0A00",
            # Decode
            top.core.datapath.immediate: "32h0000_049C",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            # ALU
            top.core.datapath.alu_result: "32h0000_0001",
            top.core.datapath.alu_op: AluOp.SEQ,
            top.core.datapath.alu_op_a: "32h0000_000A",
            top.core.datapath.alu_op_b: "32h0000_000A",
            # PC
            top.core.datapath.pc_next: "32h0040_0048",
            top.core.datapath.pc_plus_4: "32h0040_0048",
            top.core.datapath.pc_plus_immediate: "32h0040_04E0",
            # Regfile
            top.core.datapath.reg_wr_en: "1b0",
            top.core.datapath.wr_data: "32h0000_0001",
            top.core.datapath.rs1_data: "32h0000_000A",
            top.core.datapath.rs2_data: "32h0000_000A",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0001",
            top.core.datapath.data_mem_wr_data: "32h0000_000A",
        },
        # @(posedge clock)
        45: {
            top.pc: "32h0040_0048",
            top.inst: Inst(
                opcode=Opcode.OP_IMM,
                rd="5b00001",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00000",
                rs2="5b00000",
                funct7="7b0000000",
            ),
            top.bus_addr: "32h0000_0000",
            top.bus_wr_data: "32h0000_0000",
            # Decode
            top.core.datapath.immediate: "32h0000_0000",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.IMM,
            # ALU
            top.core.datapath.alu_result: "32h0000_0000",
            top.core.datapath.alu_result_eq_zero: "1b1",
            top.core.datapath.alu_op: AluOp.ADD,
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32h0000_0000",
            # PC
            top.core.datapath.pc_next: "32h0040_004C",
            top.core.datapath.pc_plus_4: "32h0040_004C",
            top.core.datapath.pc_plus_immediate: "32h0040_0048",
            # Regfile
            top.core.datapath.reg_wr_en: "1b1",
            top.core.datapath.wr_data: "32h0000_0000",
            top.core.datapath.rs1_data: "32h0000_0000",
            top.core.datapath.rs2_data: "32h0000_0000",
            # Data Mem
            top.core.datapath.data_mem_addr: "32h0000_0000",
            top.core.datapath.data_mem_wr_data: "32h0000_0000",
        },
        # @(posedge clock)
        47: {
            top.pc: "32h0040_004C",
            top.inst: Inst(
                opcode=Opcode.LUI,
                rd="5b00010",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b11111",
                rs2="5b11111",
                funct7="7b1111111",
            ),
            top.bus_addr: X32,
            top.bus_wr_data: X32,
            # Decode
            top.core.datapath.immediate: "32hFFFF_8000",
            # Control
            top.core.datapath.alu_op_b_sel: CtlAluB.RS2,
            top.core.datapath.reg_wr_sel: CtlWriteBack.IMM,
            # ALU
            top.core.datapath.alu_result: X32,
            top.core.datapath.alu_result_eq_zero: "1bX",
            top.core.datapath.alu_op_a: X32,
            top.core.datapath.alu_op_b: X32,
            # PC
            top.core.datapath.pc_next: "32h0040_0050",
            top.core.datapath.pc_plus_4: "32h0040_0050",
            top.core.datapath.pc_plus_immediate: "32h003F_804C",
            # Regfile
            top.core.datapath.wr_data: "32hFFFF_8000",
            top.core.datapath.rs1_data: X32,
            top.core.datapath.rs2_data: X32,
            # Data Mem
            top.core.datapath.data_mem_addr: X32,
            top.core.datapath.data_mem_wr_data: X32,
        },
        # @(posedge clock)
        49: {
            top.pc: "32h0040_0050",
            top.inst: Inst(
                opcode=Opcode.OP,
                rd="5b00011",
                funct3=Funct3AluLogic.ADD_SUB,
                rs1="5b00001",
                rs2="5b00010",
                funct7="7b0000000",
            ),
            top.bus_addr: "32hFFFF_8000",
            top.bus_wr_data: "32hFFFF_8000",
            # Decode
            top.core.datapath.immediate: W32,
            # Control
            top.core.datapath.reg_wr_sel: CtlWriteBack.ALU,
            # ALU
            top.core.datapath.alu_result: "32hFFFF_8000",
            top.core.datapath.alu_result_eq_zero: "1b0",
            top.core.datapath.alu_op_a: "32h0000_0000",
            top.core.datapath.alu_op_b: "32hFFFF_8000",
            # Next PC
            top.core.datapath.pc_next: "32h0040_0054",
            top.core.datapath.pc_plus_4: "32h0040_0054",
            top.core.datapath.pc_plus_immediate: W32,
            # Regfile
            top.core.datapath.rs1_data: "32h0000_0000",
            top.core.datapath.rs2_data: "32hFFFF_8000",
            # Data Mem
            top.core.datapath.data_mem_addr: "32hFFFF_8000",
            top.core.datapath.data_mem_wr_data: "32hFFFF_8000",
        },
    }
    assert waves == exp


def run_riscv_test(name: str) -> int:
    # Create module hierarchy
    top = Top(name="top")

    async def main():
        await top.main()

        # Initialize instruction memory
        text = get_mem(f"tests/riscv/tests/{name}.text")
        for i, d in enumerate(text):
            addr = u2bv(i, 10)
            data = u2bv(d, 32)
            top.text_mem_bus.text_mem.mem[addr].next = data

        # Initialize data memory
        data = get_mem(f"tests/riscv/tests/{name}.data")
        for i, d in enumerate(data):
            addr = u2bv(i, 10)
            data = u2bv(d, 32).reshape((4, 8))
            top.data_mem_bus.data_mem.mem[addr].next = data

    # Run the simulation
    for _ in irun(main(), until=10000):
        if top.bus_wr_en.value == "1b1" and top.bus_addr.value == DEBUG_REG:
            if top.bus_wr_data.value == "32h0000_0001":
                return PASS
            else:
                return FAIL
    return TIMEOUT


def test_add():
    assert run_riscv_test("add") == PASS


def test_addi():
    assert run_riscv_test("addi") == PASS


def test_and():
    assert run_riscv_test("and") == PASS


def test_andi():
    assert run_riscv_test("andi") == PASS


def test_auipc():
    assert run_riscv_test("auipc") == PASS


def test_beq():
    assert run_riscv_test("beq") == PASS


def test_bge():
    assert run_riscv_test("bge") == PASS


def test_bgeu():
    assert run_riscv_test("bgeu") == PASS


def test_blt():
    assert run_riscv_test("blt") == PASS


def test_bltu():
    assert run_riscv_test("bltu") == PASS


def test_bne():
    assert run_riscv_test("bne") == PASS


def test_jal():
    assert run_riscv_test("jal") == PASS


def test_jalr():
    assert run_riscv_test("jalr") == PASS


def test_lb():
    assert run_riscv_test("lb") == PASS


def test_lbu():
    assert run_riscv_test("lbu") == PASS


def test_lh():
    assert run_riscv_test("lh") == PASS


def test_lhu():
    assert run_riscv_test("lhu") == PASS


def test_lui():
    assert run_riscv_test("lui") == PASS


def test_lw():
    assert run_riscv_test("lw") == PASS


def test_or():
    assert run_riscv_test("or") == PASS


def test_ori():
    assert run_riscv_test("ori") == PASS


def test_sb():
    assert run_riscv_test("sb") == PASS


def test_sh():
    assert run_riscv_test("sh") == PASS


def test_simple():
    assert run_riscv_test("simple") == PASS


def test_sll():
    assert run_riscv_test("sll") == PASS


def test_slli():
    assert run_riscv_test("slli") == PASS


def test_slt():
    assert run_riscv_test("slt") == PASS


def test_slti():
    assert run_riscv_test("slti") == PASS


def test_sltiu():
    assert run_riscv_test("sltiu") == PASS


def test_sltu():
    assert run_riscv_test("sltu") == PASS


def test_sra():
    assert run_riscv_test("sra") == PASS


def test_srai():
    assert run_riscv_test("srai") == PASS


def test_srl():
    assert run_riscv_test("srl") == PASS


def test_srli():
    assert run_riscv_test("srli") == PASS


def test_sub():
    assert run_riscv_test("sub") == PASS


def test_sw():
    assert run_riscv_test("sw") == PASS


def test_xor():
    assert run_riscv_test("xor") == PASS


def test_xori():
    assert run_riscv_test("xori") == PASS
