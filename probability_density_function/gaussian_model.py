# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:14:45 2021

@author: gino9
"""

import numpy as np

class GaussianModel:
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        
    def compute_parameters(self) -> None:
        self.mean = self.features.mean(axis=0)
        self.variance = self.features.var(axis=0)
    
    def _log_density(self, x) -> float:
        return -1/2*np.log(2*np.pi)\
                - 1/2*np.log(self.variance)\
                    - (x - self.mean)/(2*self.variance)