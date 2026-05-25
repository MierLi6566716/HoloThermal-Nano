
from __future__ import print_function

import os
import numpy as np


def run_units(units, config):
    n = int(config["matrix_size"])
    seed = int(os.getpid() % 100000)
    rng = np.random.RandomState(seed)

    a = rng.rand(n, n)
    b = rng.rand(n, n)
    out = None

    for _ in range(int(units)):
        out = np.dot(a, b)
        a = out
