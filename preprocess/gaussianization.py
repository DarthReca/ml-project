# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:10 2021

@author: DarthReca
"""

import numpy as np
from scipy.stats import norm

class Gaussianizer:
    
    def fit_gaussianize(self, features: np.ndarray) -> np.ndarray:
        self.rankOver = features
        r, c = features.shape
        n = self.rankOver.shape[1]
        transformed = np.empty([r, c])
        for i in range(c):
            rank = 1 + (features[:, i].reshape([r, 1]) < self.rankOver).sum(axis=1)
            transformed[:, i] = norm.ppf(rank / (n + 2))
        return transformed
    
    def gaussianize(self, features: np.ndarray) -> np.ndarray:
        """Perform gaussianization on features."""
        r, c = features.shape
        n = self.rankOver.shape[1]
        transformed = np.empty([r, c])
        for i in range(c):
            rank = 1 + (features[:, i].reshape([r, 1]) < self.rankOver).sum(axis=1)
            transformed[:, i] = norm.ppf(rank / (n + 2))
        return transformed
