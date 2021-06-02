# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:10 2021

@author: DarthReca
"""

import numpy as np
from scipy.stats import norm

def gaussianize(features: np.ndarray):
    """Perform gaussianization on features."""
    r, c = features.shape
    transformed = np.empty([r, c])
    for i in range(c):
        rank = 1 + (features[:, i].reshape([r, 1]) < features).sum(axis=1)
        transformed[:, i] = norm.ppf(rank / (c + 2))
    return transformed
    
    
    