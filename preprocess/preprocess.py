# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:24:09 2021

@author: DarthReca
"""

import sys

import numpy as np
from dimensionality_reduction import within_class_covariance

sys.path.append("..")


def center_features(features: np.ndarray) -> np.ndarray:
    means = features.mean(axis=1)
    return features - np.vstack(means)


def standardize_variance(features: np.ndarray) -> np.ndarray:
    variance = features.std(axis=1)
    return features / np.vstack(variance)


def whiten_covariance(features: np.ndarray) -> np.ndarray:
    cov = np.cov(features, bias=True)
    eig_vals, eig_vec = np.linalg.eig(cov)
    uncorrelated = features.T.dot(eig_vec)
    uncorrelated /= np.sqrt(eig_vals + 1e-5)
    return uncorrelated.T


def normalize_lenght(features: np.ndarray) -> np.ndarray:
    normalized = np.empty(features.shape)
    for i in range(features.shape[1]):
        sample = features[:, i]
        sample /= np.linalg.norm(sample)
        normalized[:, i] = sample
    return normalized
