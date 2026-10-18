"""Annotated kernels for the Python->C cffi slice (v0.7).

Unlike the other fixtures in this dir, these functions ARE executed: the
py_harness differential tests import this module and run it. All pure
functions here must stay side-effect free (no I/O, no globals) so the
differential comparison is meaningful.

- dot / norm / matvec / fib / count_upper: pure, fully annotated.
- log_sample: deliberately IMPURE (mutates module-level LOG).
- poly_plain: deliberately UNANNOTATED (cffi glue must refuse it with a
  clear error naming the offending parameters).
"""

import math

LOG: list = []


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def matvec(a: list[float], n: int, x: list[float], y: list[float]) -> None:
    """y = A @ x with A stored row-major (n x n); y is IN/OUT."""
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[i * n + j] * x[j]
        y[i] = s


def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def count_upper(s: str) -> int:
    n = 0
    for ch in s:
        if "A" <= ch <= "Z":
            n += 1
    return n


def toggle(b: bool) -> bool:
    return not b


def log_sample(xs: list[int], v: int) -> None:
    """Deliberately impure: mutates the module-level LOG list."""
    LOG.append((list(xs), v))


def poly_plain(x, coeffs):
    """Deliberately unannotated: glue must refuse with a clear error."""
    total = 0.0
    power = 1.0
    for c in coeffs:
        total += c * power
        power *= x
    return total
