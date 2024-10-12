"""IEEE 754 Floating Point Arithmetic.


   1 bit   MSB   w bits  LSB   MSB    t = p - 1 bits    LSB
+--------+-------------------+------------------------------+
|  S     |         E         |  T                           |
| (sign) | (biased exponent) | (trailing significand field) |
+--------+-------------------+------------------------------+
           E[0].......E[w-1]   d[1]..................d[p-1]


+============+=============================+=================================+
|            | IEEE                        | HardFloat Recoded               |
|            | sign  exponent  significand | sign      exponent  significand |
+============+=============================+=================================+
| zeros      |    s         0            0 |    s   000 xx...xx            0 |
| subnormal  |    s         0            F |    s   2^k + 1 - n   F << (n+1) |
| normal     |    s         E            F |    s   2^k + 1 + E            F |
| infinities |    s   11...11            0 |    s   110 xx...xx      xx...xx |
| NaNs       |    s   11...11            F |    s   111 xx...xx            F |
+============+=============================+=================================+
"""

from ...bits import Enum, Struct
from ...bits import Vector as Vec
from ...bits import add, cat, clz, eq, le, sub, xt


class F16(Struct):
    k = 16
    p = 11
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w]
    S: Vec[1]


class F32(Struct):
    k = 32
    p = 24
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w]
    S: Vec[1]


class F64(Struct):
    k = 64
    p = 53
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w]
    S: Vec[1]


class F128(Struct):
    k = 128
    p = 113
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w]
    S: Vec[1]


type Float = F16 | F32 | F64 | F128


class R16(Struct):
    k = 16
    p = 11
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w + 1]
    S: Vec[1]


class R32(Struct):
    k = 32
    p = 24
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w + 1]
    S: Vec[1]


class R64(Struct):
    k = 64
    p = 53
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w + 1]
    S: Vec[1]


class R128(Struct):
    k = 128
    p = 113
    w = k - p
    t = p - 1

    T: Vec[t]
    E: Vec[w + 1]
    S: Vec[1]


type RecFloat = R16 | R32 | R64 | R128


class Special(Enum):
    ZERO = "3b000"
    INF = "3b110"
    NAN = "3b111"


def f2r(x: Float) -> RecFloat:
    """Convert IEEE Float to HardFloat Recoded format."""
    w = x.E.size
    rw = w + 1
    t = x.T.size

    # Recoding constant: 2^(w-1) + 1
    c = cat("1b1", Vec[w - 2].zeros(), "2b01")

    sel = cat(
        eq(x.T, x.T.zeros()),
        eq(x.E, x.E.zeros()),
        eq(x.E, x.E.ones()),
    )
    match sel:
        # Zero
        case "3b011":
            E = cat(Vec[rw - Special.size].dcs(), Special.ZERO)
            T = x.T
        # Subnormal
        case "3b010":
            n = clz(x.T)
            E = sub(c, xt(n, c.size - n.size))
            T = x.T << (n + f"{n.size}d1")
        # Normal
        case "3b000" | "3b001":
            E = add(c, xt(x.E, (c.size - w)))
            T = x.T
        # Infinities
        case "3b101":
            E = cat(Vec[rw - Special.size].dcs(), Special.INF)
            T = x.T
        # NaNs
        case "3b100":
            E = cat(Vec[rw - Special.size].dcs(), Special.NAN)
            T = x.T
        # Propagate Xes
        case _:
            E = Vec[rw].xprop(sel)
            T = Vec[t].xprop(sel)

    match x:
        case F16():
            return R16(T=T, E=E, S=x.S)
        case F32():
            return R32(T=T, E=E, S=x.S)
        case F64():
            return R64(T=T, E=E, S=x.S)
        case F128():
            return R128(T=T, E=E, S=x.S)
        case _:
            assert False


def r2f(x: RecFloat) -> Float:
    """Convert HardFloat Recoded format to IEEE Float."""
    rw = x.E.size
    w = rw - 1
    t = x.T.size

    # Recoding constant: 2^(w-1) + 1
    c = cat("1b1", Vec[w - 2].zeros(), "2b01")

    sel = x.E[(rw - Special.size) : rw]  # noqa
    match sel:
        # Zero
        case Special.ZERO:
            E = Vec[w].zeros()
            T = x.T
        # Normal / Subnormal
        case "3b001" | "3b010" | "3b011" | "3b100" | "3b101":
            subnormal = le(x.E, c)
            match subnormal:
                # Subnormal
                case "1b1":
                    E = Vec[w].zeros()
                    n = sub(c, x.E)
                    T = cat(x.T[1:], "1b1") >> n  # {1,F} >> n
                # Normal
                case "1b0":
                    E = sub(x.E, c)[:w]
                    T = x.T
                # Propagate Xes
                case _:
                    E = Vec[w].xprop(subnormal)
                    T = Vec[t].xprop(subnormal)
        # Infinities
        case Special.INF:
            E = Vec[w].ones()
            T = Vec[t].zeros()
        # NaNs
        case Special.NAN:
            E = Vec[w].ones()
            T = x.T
        # Propagate Xes
        case _:
            E = Vec[w].xprop(sel)
            T = Vec[t].xprop(sel)

    match x:
        case R16():
            return F16(T=T, E=E, S=x.S)
        case R32():
            return F32(T=T, E=E, S=x.S)
        case R64():
            return F64(T=T, E=E, S=x.S)
        case R128():
            return F128(T=T, E=E, S=x.S)
        case _:
            assert False
