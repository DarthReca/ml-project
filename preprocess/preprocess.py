# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:24:09 2021

@author: DarthReca
"""

import sys

import numpy as np

from dimensionality_reduction import within_class_covariance

sys.path.append("..")

class Preprocessor:
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.means = features.mean(axis=1)
        self.std = features.std(axis=1)
        self.cov = np.cov(features, bias=True)
        
        f = self.center_features(features)
        f = self.standardize_variance(f)
        #f = self.whiten_covariance(f)
        f = self.normalize_lenght(f)
        
        return f

    def transform(self, features: np.ndarray) -> np.ndarray:
        f = self.center_features(features)
        f = self.standardize_variance(f)
        #f = self.whiten_covariance(f)
        f = self.normalize_lenght(f)
        
        return f

        
    def center_features(self, features: np.ndarray) -> np.ndarray:
        return features - np.vstack(self.means)
    
    
    def standardize_variance(self, features: np.ndarray) -> np.ndarray:
        return features / np.vstack(self.std)
    
    
    def whiten_covariance(self, features: np.ndarray) -> np.ndarray:
        W = np.sqrt(self.cov)
        r, n = features.shape
        uncorrelated = np.empty([r, n])
        for i in range(n):
            uncorrelated[:, i] = W.dot(features[:, i])
        return uncorrelated
    
    
    def normalize_lenght(self, features: np.ndarray) -> np.ndarray:
        normalized = np.empty(features.shape)
        for i in range(features.shape[1]):
            sample = features[:, i]
            sample /= np.linalg.norm(sample)
            normalized[:, i] = sample
        return normalized
    
