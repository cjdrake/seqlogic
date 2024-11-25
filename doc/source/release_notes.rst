*********************
    Release Notes
*********************

This section lists new features, API changes, and bug fixes.
For a complete history, see the Git commit log.

Version 0.37.0
==============

Add a ``Module.parameterize`` class method.
For modules that have parameters,
this creates a new class for the specific parameter values,
which extends from the generic base class.

For example::

    >>> class RCA(Module):
    ...     N: int = 8
    ...     def build(self):
    ...         ...

    >>> RCA.N
    8
    >>> RCA_32 = RCA.parameterize(N=32)
    >>> RCA_32.N
    32

This updates how submodules are instantiated.

Previously::

    self.submod(
        name="rca32",
        mod=RCA,
        N=32,
    ).connect(
        s=s,
        ci=ci,
        a=a,
        b=b,
        co=co,
    )

Now this works::

    self.submod(
        name="rca32",
        mod=RCA.parameterize(N=32),
    ).connect(
        s=s,
        ci=ci,
        a=a,
        b=b,
        co=co,
    )

Version 0.36.0
==============

Chose to host documentation on `Read The Docs <https://rtfd.org>`_.

Minor improvements to reference documentation.

Added an example Johnson Counter notebook.

Moved ``clz`` function from ``bits`` module to ``algorithms.count`` module.

Added capability to update variables using ``x.next = <int/bool>``.

Previously::

    async def drive(valid: Vec[1], data: Vec[8]):
        valid.next = "1b1"
        data.next = "8d42"

Now this works::

    async def drive(valid: Vec[1], data: Vec[8]):
        valid.next = 1
        data.next = 42

Version 0.35.0
==============

Changed VCD ``VarType`` used by bit vectors from ``reg`` to ``logic``.
See `PyVCD Changelog`_ version 0.4.1 for details.

.. _PyVCD Changelog: https://github.com/westerndigitalcorporation/pyvcd/blob/master/CHANGELOG.rst
