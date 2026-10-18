"""Pure numeric kernels (fixture for test_pymine.py). Never executed here."""

import math
from math import sqrt as msqrt


def dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def matmul(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for m in range(p * p, n + 1, p):
                is_prime[m] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]


def norm(v):
    return math.sqrt(dot(v, v))


def unit(v):
    n = norm(v)
    return [x / n for x in v]


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def hypot2(x, y):
    return msqrt(x * x + y * y)


def poly(x, coeffs):
    """Unannotated on purpose: annotations must not be required."""
    total = 0.0
    power = 1.0
    for c in coeffs:
        total += c * power
        power *= x
    return total


def typed_dot(a: list, b: list) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def vsum(*args, **kwargs):
    """Variadic signatures are rejected with a clear reason (no C equivalent)."""
    s = 0.0
    for a in args:
        s += a
    for v in kwargs.values():
        s += v
    return s


def gsum(xs):
    """Pure generator-expression kernel: the genexp must be walked (v0.9)."""
    return sum(x * x for x in xs)


def clamp(x, lo=-1.0, hi=1.0):
    assert lo <= hi, "bad bounds"
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
