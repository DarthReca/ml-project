# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:10 2021

@author: DarthReca
"""

import numpy as np
from scipy.stats import norm


def gaussianize(features: np.ndarray, rankOver: np.ndarray = np.zeros([1])) -> np.ndarray:
    """Perform gaussianization on features."""
    if np.add.reduce(rankOver, axis=None) == 0:
        rankOver = features
    r, c = features.shape
    n = rankOver.shape[1]
    transformed = np.empty([r, c])
    for i in range(c):
        rank = 1 + (features[:, i].reshape([r, 1]) < rankOver).sum(axis=1)
        transformed[:, i] = norm.ppf(rank / (n + 2))
    return transformed
