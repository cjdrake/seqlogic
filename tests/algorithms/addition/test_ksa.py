"""Test Kogge Stone Addition (KSA) algorithm."""

import os

from seqlogic import u2bv
from seqlogic.algorithm.addition.ksa import add

DIR = os.path.dirname(__file__)

N_START = 1
N_STOP = 6


def test_functional():
    for n in range(N_START, N_STOP):
        for i in range(2**n):
            for j in range(2**n):
                for k in range(2**1):
                    # Inputs
                    a = u2bv(i, n)
                    b = u2bv(j, n)
                    ci = u2bv(k, 1)

                    # Outputs
                    s, co = add(a, b, ci)

                    # Check outputs
                    q, r = divmod(i + j + k, 2**n)
                    assert s.to_uint() == r
                    assert co.to_uint() == q
