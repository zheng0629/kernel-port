"""Impure functions (fixture for test_pymine.py). Each must be rejected
with a concrete reason. Never executed here."""

import os
import random
import time

LOG = []
COUNTER = 0
SCALE = 2.0  # ALL_CAPS literal: readable as a constant


class _App:
    def route(self, path):
        def deco(fn):
            return fn

        return deco


app = _App()


def save_results(path, data):
    with open(path, "w") as fh:
        fh.write(str(data))


def random_walk(n):
    x = 0.0
    for _ in range(n):
        x += random.random()
    return x


def bump():
    global COUNTER
    COUNTER += 1
    return COUNTER


def cache_put(key, value):
    LOG.append((key, value))


def stamp():
    return time.time()


def env_home():
    return os.environ["HOME"]


def dynamic(name):
    return getattr(__builtins__, name)


def run(code):
    eval(code)


def scaled(x):
    return x * SCALE  # pure: reads an ALL_CAPS literal constant


def reads_log():
    return len(LOG)  # impure: reads a mutable module-level variable


@app.route("/predict")
def predict(x):
    return x * 2.0  # impure: decorator implies framework behavior


def raiser(x):
    if x < 0:
        raise ValueError("negative")
    return x


class CustomError(Exception):
    pass


_ERR = ValueError("prebuilt")


def raiser_custom(x):
    if x < 0:
        raise CustomError("negative")
    return x


def raiser_prebuilt(x):
    if x < 0:
        raise _ERR
    return x


def raiser_bare(x):
    try:
        return 1 / x
    except ZeroDivisionError:
        raise


def genexp_print(xs):
    return sum(x * 2 for x in xs if print(x) or True)


def genexp_open(xs):
    return sum(1 for _ in xs if open("/tmp/x").write("y"))
