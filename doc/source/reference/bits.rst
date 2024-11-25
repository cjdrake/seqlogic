***************************
    Combinational Logic
***************************

Data Types
==========

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

.. autoclass:: seqlogic.bits.Empty
.. autoclass:: seqlogic.bits.Scalar
.. autoclass:: seqlogic.bits.Vector
.. autoclass:: seqlogic.bits.Array

.. TODO(cjdrake): seqlogic.bits.Enum
.. TODO(cjdrake): seqlogic.bits.Struct
.. TODO(cjdrake): seqlogic.bits.Union

Operators
=========

Bitwise
-------

.. autofunction:: seqlogic.bits.not_
.. autofunction:: seqlogic.bits.nor
.. autofunction:: seqlogic.bits.or_
.. autofunction:: seqlogic.bits.nand
.. autofunction:: seqlogic.bits.and_
.. autofunction:: seqlogic.bits.xnor
.. autofunction:: seqlogic.bits.xor
.. TODO(cjdrake): seqlogic.bits.mux
.. autofunction:: seqlogic.bits.ite

Unary
-----

.. autofunction:: seqlogic.bits.uor
.. autofunction:: seqlogic.bits.uand
.. autofunction:: seqlogic.bits.uxnor
.. autofunction:: seqlogic.bits.uxor

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

.. autofunction:: seqlogic.bits.eq
.. autofunction:: seqlogic.bits.ne

.. autofunction:: seqlogic.bits.lt
.. autofunction:: seqlogic.bits.le
.. autofunction:: seqlogic.bits.gt
.. autofunction:: seqlogic.bits.ge

.. autofunction:: seqlogic.bits.slt
.. autofunction:: seqlogic.bits.sle
.. autofunction:: seqlogic.bits.sgt
.. autofunction:: seqlogic.bits.sge

Factory Functions
=================

.. TODO(cjdrake): seqlogic.bits.bits
.. TODO(cjdrake): seqlogic.bits.stack
.. TODO(cjdrake): seqlogic.bits.u2bv
.. TODO(cjdrake): seqlogic.bits.i2bv
