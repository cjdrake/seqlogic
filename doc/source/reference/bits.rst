***************************
    Combinational Logic
***************************

Data Types
==========

.. autoclass:: seqlogic.Bits

    .. py:property:: size

        Number of bits

    .. automethod:: seqlogic.Bits.cast

    .. automethod:: seqlogic.Bits.xes
    .. automethod:: seqlogic.Bits.zeros
    .. automethod:: seqlogic.Bits.ones
    .. automethod:: seqlogic.Bits.dcs

    .. automethod:: seqlogic.Bits.xprop

    .. autoproperty:: seqlogic.Bits.data

    .. automethod:: seqlogic.Bits.__bool__
    .. automethod:: seqlogic.Bits.__int__

    .. automethod:: seqlogic.Bits.to_uint
    .. automethod:: seqlogic.Bits.to_int

    .. automethod:: seqlogic.Bits.count_xes
    .. automethod:: seqlogic.Bits.count_zeros
    .. automethod:: seqlogic.Bits.count_ones
    .. automethod:: seqlogic.Bits.count_dcs
    .. automethod:: seqlogic.Bits.count_unknown

    .. automethod:: seqlogic.Bits.onehot
    .. automethod:: seqlogic.Bits.onehot0

    .. automethod:: seqlogic.Bits.has_x
    .. automethod:: seqlogic.Bits.has_dc
    .. automethod:: seqlogic.Bits.has_unknown

.. autoclass:: seqlogic.Empty
.. autoclass:: seqlogic.Scalar
.. autoclass:: seqlogic.Vector
.. autoclass:: seqlogic.Array

.. autoclass:: seqlogic.Enum
.. autoclass:: seqlogic.Struct
.. autoclass:: seqlogic.Union

Operators
=========

Bitwise
-------

.. autofunction:: seqlogic.not_
.. autofunction:: seqlogic.nor
.. autofunction:: seqlogic.or_
.. autofunction:: seqlogic.nand
.. autofunction:: seqlogic.and_
.. autofunction:: seqlogic.xnor
.. autofunction:: seqlogic.xor
.. autofunction:: seqlogic.mux
.. autofunction:: seqlogic.ite

Unary
-----

.. autofunction:: seqlogic.uor
.. autofunction:: seqlogic.uand
.. autofunction:: seqlogic.uxnor
.. autofunction:: seqlogic.uxor

Arithmetic
----------

.. autofunction:: seqlogic.decode
.. autofunction:: seqlogic.encode_onehot
.. autofunction:: seqlogic.add
.. autofunction:: seqlogic.adc
.. autofunction:: seqlogic.sub
.. autofunction:: seqlogic.sbc
.. autofunction:: seqlogic.neg
.. autofunction:: seqlogic.ngc
.. autofunction:: seqlogic.mul
.. autofunction:: seqlogic.div
.. autofunction:: seqlogic.mod
.. autofunction:: seqlogic.lsh
.. autofunction:: seqlogic.rsh
.. autofunction:: seqlogic.srsh

Word
----

.. autofunction:: seqlogic.xt
.. autofunction:: seqlogic.sxt
.. autofunction:: seqlogic.lrot
.. autofunction:: seqlogic.rrot
.. autofunction:: seqlogic.cat
.. autofunction:: seqlogic.rep
.. autofunction:: seqlogic.pack

Predicate
---------

.. autofunction:: seqlogic.match

.. autofunction:: seqlogic.eq
.. autofunction:: seqlogic.ne

.. autofunction:: seqlogic.lt
.. autofunction:: seqlogic.le
.. autofunction:: seqlogic.gt
.. autofunction:: seqlogic.ge

.. autofunction:: seqlogic.slt
.. autofunction:: seqlogic.sle
.. autofunction:: seqlogic.sgt
.. autofunction:: seqlogic.sge

Factory Functions
=================

.. autofunction:: seqlogic.bits
.. autofunction:: seqlogic.stack
.. autofunction:: seqlogic.u2bv
.. autofunction:: seqlogic.i2bv
