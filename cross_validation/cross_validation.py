# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:48:43 2021

@author: DarthReca
"""

import numpy as np


def shuffle_sample(features: np.ndarray, labels: np.ndarray, samples: int):
    dataset = np.vstack([features, labels])
    rng = np.random.default_rng(seed=0)
    rng.shuffle(dataset, axis=1)

    sampled = []
    sample_size = int(features.shape[1] / samples)
    for i in range(samples):
        start = i * sample_size
        s = dataset[:, start : (start + sample_size)]
        sampled.append((s[:-1], s[-1]))

    return sampled
