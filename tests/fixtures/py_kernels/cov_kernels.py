"""Kernels for coverage-guided verify tests (v0.11).

Like ``cffi_kernels.py``, these functions ARE executed by the py_harness
differential tests -- keep them pure and side-effect free.

- ``rare_branch``: a branch that random inputs (uniform in
  [-1000, 1000]) and the boundary sweep can never hit -- the guided
  round must aim the ``1234567`` literal at it.
- ``loopy``: a plain loop kernel used for loop-body steering tests.
"""


def rare_branch(x: int) -> int:
    # NOTE: the else-branch is the identity on purpose -- the boundary
    # sweep feeds I64_MAX/I64_MIN, and any arithmetic there would trip
    # the harness's int64-closure rule (Python bignum vs C wraparound).
    if x == 1234567:
        return x * 2
    return x


def loopy(items: list[int]) -> int:
    s = 0
    for v in items:
        s += v
    return s
