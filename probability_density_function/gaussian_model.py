# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:14:45 2021

@author: gino9
"""

from typing import List

import numpy as np


class GaussianModel:

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the model with given features matrix and labels array.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray

        """
        self.means = []
        self.covariances = []
        for i in range(2):
            selected_datas = features[:, labels == i]
            mean = np.vstack(selected_datas.mean(axis=1, dtype=np.float64))
            cov = np.cov(selected_datas, bias=True)
            self.means.append(mean)
            self.covariances.append(cov)

    def _log_likelihood_sample(self, x: np.ndarray,
                               mu: np.ndarray,
                               sigma: np.ndarray) -> float:
        m = x.shape[0]
        sign, sigma_log_det = np.linalg.slogdet(sigma)
        sigma_inv = np.linalg.inv(sigma)
        dc = x - mu

        return (-m / 2 * np.log(2 * np.pi) - 0.5 * sigma_log_det
                - 0.5 * np.dot(dc.T, np.dot(sigma_inv, dc)).item())

    def _log_likelihood(self, features: np.ndarray) -> np.ndarray:
        n = features.shape[1]
        result = np.empty([2, n])

        for i in range(n):
            likelihood = np.empty(2)
            for lab in range(2):
                curr_cov = self.covariances[lab]
                curr_mean = self.means[lab]
                curr_sample = np.row_stack(features[:, i])
                likelihood[lab] = self._log_likelihood_sample(
                    curr_sample, curr_mean, curr_cov)
            result[:, i] = likelihood

        return result

    def predict(self, features: np.ndarray, prior_prob: float) -> np.ndarray:
        """
        Apply model on feautures and predict label.

        Parameters
        ----------
        features : np.ndarray

        prior_prob : float

        Returns
        -------
        predictions: ndarray

        """
        threshold = -np.log(prior_prob / (1 - prior_prob))
        likelihood = self._log_likelihood(features)
        ratio = likelihood[1, :] / likelihood[0, :]
        return (ratio > threshold).astype(np.int32)
