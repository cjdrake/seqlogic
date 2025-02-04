***********************
    Design Elements
***********************

.. autoexception:: seqlogic.DesignError

.. autoclass:: seqlogic.Module

    .. automethod:: seqlogic.Module.build
    .. automethod:: seqlogic.Module.elab

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
    .. automethod:: seqlogic.Module.dff_r
    .. automethod:: seqlogic.Module.dff_en
    .. automethod:: seqlogic.Module.dff_en_r
    .. automethod:: seqlogic.Module.mem_wr

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
