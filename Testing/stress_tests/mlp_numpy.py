
from __future__ import print_function

import os
import numpy as np


def run_units(units, config):
    samples = int(config["mlp_samples"])
    features = int(config["mlp_features"])
    hidden = int(config["mlp_hidden"])
    outputs = int(config["mlp_outputs"])

    seed = int(os.getpid() % 100000)
    rng = np.random.RandomState(seed)

    x = rng.randn(samples, features)
    w1 = rng.randn(features, hidden)
    w2 = rng.randn(hidden, outputs)

    for _ in range(int(units)):
        h = np.dot(x, w1)
        h[h < 0.0] = 0.0
        y = np.dot(h, w2)
        y = 1.0 / (1.0 + np.exp(-y))
        g2 = np.dot(h.T, y)
        back = np.dot(y, w2.T)
        back[back < 0.0] = 0.0
        g1 = np.dot(x.T, back)
        w2 = w2 - 0.000001 * g2
        w1 = w1 - 0.000001 * g1
