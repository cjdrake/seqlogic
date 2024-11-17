*****************
    Reference
*****************

Bits Data Type
==============

.. autoclass:: seqlogic.bits.Bits

    .. py:property:: size

        Number of bits

    .. automethod:: seqlogic.bits.Bits.cast

    .. automethod:: seqlogic.bits.Bits.xes
    .. automethod:: seqlogic.bits.Bits.zeros
    .. automethod:: seqlogic.bits.Bits.ones
    .. automethod:: seqlogic.bits.Bits.dcs

    .. automethod:: seqlogic.bits.Bits.xprop

    .. autoproperty:: seqlogic.bits.Bits.data

    .. automethod:: seqlogic.bits.Bits.__bool__
    .. automethod:: seqlogic.bits.Bits.__int__

    .. automethod:: seqlogic.bits.Bits.to_uint
    .. automethod:: seqlogic.bits.Bits.to_int

    .. automethod:: seqlogic.bits.Bits.count_xes
    .. automethod:: seqlogic.bits.Bits.count_zeros
    .. automethod:: seqlogic.bits.Bits.count_ones
    .. automethod:: seqlogic.bits.Bits.count_dcs
    .. automethod:: seqlogic.bits.Bits.count_unknown

    .. automethod:: seqlogic.bits.Bits.onehot
    .. automethod:: seqlogic.bits.Bits.onehot0

    .. automethod:: seqlogic.bits.Bits.has_x
    .. automethod:: seqlogic.bits.Bits.has_dc
    .. automethod:: seqlogic.bits.Bits.has_unknown

.. automodule:: seqlogic._bits_enum
    :members:
    :show-inheritance:

.. automodule:: seqlogic._bits_struct
    :members:
    :show-inheritance:

.. automodule:: seqlogic._bits_union
    :members:
    :show-inheritance:

Design Elements
===============

.. automodule:: seqlogic.design
    :members:
    :show-inheritance:

Event Simulation
================

.. automodule:: seqlogic.sim
    :members:
    :show-inheritance:

Utilities
=========

.. autofunction:: seqlogic.util.clog2
