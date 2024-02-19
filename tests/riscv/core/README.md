This directory contains a simple RISCV (rv32i) core.

The design is adapted from [riscv-simple-sv](https://github.com/tilk/riscv-simple-sv),
by Marek Materzok.
Its "singlecycle" core architecture was adapted from
[riscv-simple](https://github.com/arthurbeggs/riscv-simple),
by Arthur Matos.

The original source code was written in SystemVerilog.
We have adapted it to Python for use with `seqlogic`'s simulation kernel.
We have made some minor changes for both functional and aesthetic purposes.
