***********************
    Design Elements
***********************

.. autoexception:: seqlogic.DesignError
.. autoexception:: seqlogic.AssumeError
.. autoexception:: seqlogic.AssertError

.. autoclass:: seqlogic.Module

    .. automethod:: seqlogic.Module.build
    .. automethod:: seqlogic.Module.main

    .. autoproperty:: seqlogic.Module.scope

    .. automethod:: seqlogic.Module.dump_waves
    .. automethod:: seqlogic.Module.dump_vcd

    .. automethod:: seqlogic.Module.input
    .. automethod:: seqlogic.Module.output
    .. automethod:: seqlogic.Module.connect
    .. automethod:: seqlogic.Module.logic
    .. automethod:: seqlogic.Module.submod

    .. automethod:: seqlogic.Module.drv
    .. automethod:: seqlogic.Module.mon

    .. automethod:: seqlogic.Module.combi
    .. automethod:: seqlogic.Module.expr
    .. automethod:: seqlogic.Module.assign

    .. automethod:: seqlogic.Module.dff
    .. automethod:: seqlogic.Module.mem_wr

    .. automethod:: seqlogic.Module.assume_func
    .. automethod:: seqlogic.Module.assert_func
    .. automethod:: seqlogic.Module.assume_expr
    .. automethod:: seqlogic.Module.assert_expr
    .. automethod:: seqlogic.Module.assume_seq
    .. automethod:: seqlogic.Module.assert_seq

.. autoclass:: seqlogic.Logic

    .. autoproperty:: seqlogic.Logic.dtype

.. autoclass:: seqlogic.Packed

    .. automethod:: seqlogic.Packed.is_neg
    .. automethod:: seqlogic.Packed.is_posedge
    .. automethod:: seqlogic.Packed.is_negedge
    .. automethod:: seqlogic.Packed.is_pos
    .. automethod:: seqlogic.Packed.is_edge
    .. automethod:: seqlogic.Packed.posedge
    .. automethod:: seqlogic.Packed.negedge
    .. automethod:: seqlogic.Packed.edge

.. autoclass:: seqlogic.Unpacked
