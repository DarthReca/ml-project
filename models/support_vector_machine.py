# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:55:54 2021

@author: gino9
"""


import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class SupportVectorMachine:
    def __init__(self):
        self.k = 10
        self.C = 0.1

    def _binary_cross_entropy(
        self, samples: np.ndarray, z_labels: np.ndarray
    ) -> np.ndarray:
        lab_count = z_labels.shape[0]
        lab = np.broadcast_to(z_labels, [lab_count, lab_count])
        return lab * lab.T * np.dot(samples.T, samples)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        samples_count = features.shape[1]

        z_labels = 2 * labels - 1
        mapped_features = np.vstack([features, np.repeat(self.k, samples_count)])

        cross_entropy = self._binary_cross_entropy(mapped_features, z_labels)

        b = [(0, self.C) for _ in range(samples_count)]
        sol = fmin_l_bfgs_b(
            self._inverse_obj_funct,
            np.zeros(samples_count),
            args=[cross_entropy],
            bounds=b,
            factr=1.0,
        )

        self.w = np.zeros(mapped_features.shape[0])
        for i in range(samples_count):
            self.w += sol[0][i] * z_labels[i] * mapped_features[:, i]

    def predict(self, features: np.ndarray):
        samples_count = features.shape[1]

        X = np.vstack([features, np.repeat(self.k, samples_count)])
        scores = np.dot(self.w.T, X)

        return (scores > 0).astype(np.int32)

    def _primal_objective(self, w, C, X, Z):
        J = 0.5 * np.linalg.norm(w) ** 2
        for i in range(X.shape[1]):
            J += C * max(0, 1 - Z[i] * np.dot(w.T, X[:, i]))
        return J

    def _inverse_obj_funct(self, alpha: np.ndarray, cross_entropy: np.ndarray):
        samples_count = alpha.shape[0]

        L = 0.5 * np.dot(np.dot(alpha.T, cross_entropy), alpha) - np.dot(
            alpha.T, np.ones(samples_count)
        )
        gradient = np.dot(cross_entropy, alpha) - np.ones(samples_count)
        return L, gradient
