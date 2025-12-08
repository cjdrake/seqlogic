.. _release_notes:

#####################
    Release Notes
#####################

This section lists new features, API changes, and bug fixes.
For a complete history, see the Git commit log.


Version 0.51.0
==============

Minor changes for ``bvwx`` v0.17.


Version 0.50.0
==============

Mostly updates related to backward-incompatible ``deltacycle`` changes.
Lots of miscellaneous type annotation work.


Version 0.49.0
==============

Renamed/Consolidated checker methods:

* assume_impl => assume_func
* assert_impl => assert_func
* assume_immed => assume_func
* assert_immed => assert_func

Gave checkers new options to choose negedge async reset.

Added ``Const``, ``BitsConst``, and ``IntConst`` to top level namespace

Various other refactoring and simplification.


Version 0.48.0
==============

Minor changes for ``deltacycle`` v0.12.0.


Version 0.47.0
==============

Implemented new checker feature.
Modules contain new methods:

* assume_impl
* assert_impl
* assume_immed
* assert_immed
* assume_seq
* assert_seq

Checkers can raise new assertions:

* AssumeError
* AssertError

Created a basic ready/valid checker,
and attached it to HBEB, FBEB, and pipe register tests.


Version 0.46.0
==============

Updating to ``deltacycle`` 0.11 API,
incorporating several top-level API changes.
Also some miscellaneous bug fixes and performance improvements.


Version 0.45.0
==============

Update how module parameterization works.
Get rid of ``Module.parameterize`` and ``Module.paramz`` methods.
Now, if the module has parameters,
use its constructor to create a parameterized module.

For example::

    class MyModule(Module):
        N: int = 8

        def build(self):
            ...

    my_mod = MyModule(N=32)


Version 0.44.0
==============

Updated tooling to use ``uv`` and ``ruff``.


Version 0.43.0
==============

Minor changes required for ``deltacycle`` 0.3.0.


Version 0.42.0
==============

Minor changes related to ``bvwx`` and ``deltacycle`` dependencies.


Version 0.41.0
==============

Lots of breaking changes in this release.

Breaking changes to design constructs:
* Replaced ``dff_*`` with a single ``dff`` method
* Replaced ``mem_wr_*`` with a single ``mem_wr`` method

Some minor, but important fixes to simulator internals.
The ``Region`` class is now in design module.
The simulator kernel no longer has to choose whether to use next state or
present state; this is now left to the implementation.
Now that concerns are better separated,
it might be valuable to break out the sim.py module into a separate package.

Renamed the ``Module.elab`` method to ``Module.main``.

Shorted the ``Module.parameterize`` method name to just ``Module.paramz``.
(Though the old name is still usable as an alias).

Got rid of the opinionated requirement for module internals to prefix all
names with ``_`` character.
Simpler, more explicit, cleaner. Obviously better.

The ``bvwx`` version 0.5 dependency introduces "logical" operators,
which can make certain expressions easier to implement.


Version 0.40.0
==============

Added a `float` design component.
Experimental. Not yet documented or tested.

Migrated ``bits`` data type from local to external library, ``bvwx``.
Latest version added ``impl`` operator,
and implemented a few miscellaneous bug fixes and improvements.

Added a ``ctz`` (count trailing zeros) function to ``algorithms.count`` module.


Version 0.39.0
==============

Add ``encode_onehot`` and ``encode_priority`` functions to the bits module.


Version 0.38.0
==============

Lots of new documentation, and improved type hints.

Implemented ``Empty.__getitem__`` for consistency with ``Scalar``,
``Vector``, and ``Array``.

Allow ``Array`` slice operator to take string literals as indices.

Relax ``Bits`` ``add`` and ``mul`` operators.
Input sizes no longer need to match.

Implement new ``div`` and ``mod`` operators for basic unsigned
division and modulus.


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
