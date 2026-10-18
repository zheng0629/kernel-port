"""Closure / generator / lambda shapes (fixture for test_pymine.py)."""


def make_adder(n):
    def adder(x):
        return x + n

    return adder


def with_lambda(xs):
    return sorted(xs, key=lambda v: -v)


def gen(n):
    yield n


def outer_state(xs):
    total = 0

    def bump(x):
        nonlocal total
        total += x

    for x in xs:
        bump(x)
    return total


def multiplies_out(out, xs, k):
    """Out-parameter mutation: allowed by the out-param convention."""
    for i, x in enumerate(xs):
        out[i] = x * k
