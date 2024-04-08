"""TODO(cjdrake): Write docstring."""

from seqlogic.lbool import VecEnum

TEXT_BASE = 0x0040_0000
TEXT_BITS = 16
TEXT_SIZE = 2**TEXT_BITS

DATA_BASE = 0x8000_0000
DATA_BITS = 17
DATA_SIZE = 2**DATA_BITS


class Opcode(VecEnum):
    """Instruction opcodes."""

    LOAD = "7b000_0011"
    LOAD_FP = "7b000_0111"
    MISC_MEM = "7b000_1111"
    OP_IMM = "7b001_0011"
    AUIPC = "7b001_0111"
    STORE = "7b010_0011"
    STORE_FP = "7b010_0111"
    OP = "7b011_0011"
    LUI = "7b011_0111"
    OP_FP = "7b101_0011"
    BRANCH = "7b110_0011"
    JALR = "7b110_0111"
    JAL = "7b110_1111"
    SYSTEM = "7b111_0011"


class CtlPc(VecEnum):
    """TODO(cjdrake): Write docstring."""

    PC4 = "2b00"
    PC_IMM = "2b01"
    RS1_IMM = "2b10"
    PC4_BR = "2b11"


class CtlAlu(VecEnum):
    """ALU op types."""

    ADD = "2b00"
    BRANCH = "2b01"
    OP = "2b10"
    OP_IMM = "2b11"


class AluOp(VecEnum):
    """ALU Operations."""

    ADD = "5b0_0001"
    SUB = "5b0_0010"
    SLL = "5b0_0011"
    SRL = "5b0_0100"
    SRA = "5b0_0101"
    SEQ = "5b0_0110"
    SLT = "5b0_0111"
    SLTU = "5b0_1000"
    XOR = "5b0_1001"
    OR = "5b0_1010"
    AND = "5b0_1011"
    MUL = "5b0_1100"
    MULH = "5b0_1101"
    MULHSU = "5b0_1110"
    MULHU = "5b0_1111"
    DIV = "5b1_0000"
    DIVU = "5b1_0001"
    REM = "5b1_0010"
    REMU = "5b1_0011"


class CtlAluA(VecEnum):
    """ALU 1st operand source."""

    RS1 = "1b0"
    PC = "1b1"


class CtlAluB(VecEnum):
    """ALU 2nd operand source."""

    RS2 = "1b0"
    IMM = "1b1"


class Funct3AluLogic(VecEnum):
    """Interpretations of the "funct3" field."""

    ADD_SUB = "3b000"
    SLL = "3b001"
    SLT = "3b010"
    SLTU = "3b011"
    XOR = "3b100"
    SHIFTR = "3b101"
    OR = "3b110"
    AND = "3b111"


class Funct3Branch(VecEnum):
    """Interpretations of the "funct3" field for branches."""

    EQ = "3b000"
    NE = "3b001"
    LT = "3b100"
    GE = "3b101"
    LTU = "3b110"
    GEU = "3b111"


class Funct3AluMul(VecEnum):
    """Interpretations of the "funct3" field for extension M."""

    MUL = "3b000"
    MULH = "3b001"
    MULHSU = "3b010"
    MULHU = "3b011"
    DIV = "3b100"
    DIVU = "3b101"
    REM = "3b110"
    REMU = "3b111"


class CtlWriteBack(VecEnum):
    """Register data sources."""

    ALU = "3b000"
    DATA = "3b001"
    PC4 = "3b010"
    IMM = "3b011"
