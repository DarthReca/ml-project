# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:11:10 2021

@author: DarthReca
"""

import numpy as np
from sklearn.stats import norm

def gaussianize(features: np.ndarray):
    pass

def rank_feature(x: float, features: np.ndarray):
    n = features.shape[1]
    rank = 0.00
    for x_i in features:
        rank += 1
        if x < x_i:
            rank += 1
    return rank / (n + 2)
    
    
    