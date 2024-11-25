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
.. TODO(cjdrake): seqlogic.mux
.. autofunction:: seqlogic.ite

Unary
-----

.. autofunction:: seqlogic.uor
.. autofunction:: seqlogic.uand
.. autofunction:: seqlogic.uxnor
.. autofunction:: seqlogic.uxor

Arithmetic
----------

.. TODO(cjdrake): seqlogic.bits.decode
.. TODO(cjdrake): seqlogic.bits.add
.. TODO(cjdrake): seqlogic.bits.adc
.. TODO(cjdrake): seqlogic.bits.sub
.. TODO(cjdrake): seqlogic.bits.sbc
.. TODO(cjdrake): seqlogic.bits.neg
.. TODO(cjdrake): seqlogic.bits.ngc
.. TODO(cjdrake): seqlogic.bits.mul
.. TODO(cjdrake): seqlogic.bits.lsh
.. TODO(cjdrake): seqlogic.bits.rsh
.. TODO(cjdrake): seqlogic.bits.srsh

Word
----

.. TODO(cjdrake): seqlogic.bits.xt
.. TODO(cjdrake): seqlogic.bits.sxt
.. TODO(cjdrake): seqlogic.bits.lrot
.. TODO(cjdrake): seqlogic.bits.rrot
.. TODO(cjdrake): seqlogic.bits.cat
.. TODO(cjdrake): seqlogic.bits.rep
.. TODO(cjdrake): seqlogic.bits.pack

Predicate
---------

.. TODO(cjdrake): seqlogic.bits.match

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

.. TODO(cjdrake): seqlogic.bits.bits
.. TODO(cjdrake): seqlogic.bits.stack
.. TODO(cjdrake): seqlogic.bits.u2bv
.. TODO(cjdrake): seqlogic.bits.i2bv
