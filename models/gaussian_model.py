# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:14:45 2021

@author: gino9
"""

import sys
from typing import Tuple, Union

import data_result_analysis as dra
import dimensionality_reduction as dr
import numpy as np

sys.path.append("..")


class GaussianModel:
    """Gaussian Model class.

    Parameters
    ----------
    threshold: float
        threshold for likelihood that separate classes.
    """

    def __init__(self, threshold: float):
        """
        Gaussian Model class.

        Parameters
        ----------
        threshold : float
            threshold for likelihood that separate classes.
        """
        self.threshold = threshold

    def set_prior(self, prior: float) -> None:
        self.threshold = -np.log(prior/(1 - prior))

    def set_threshold(self, threshold: float) -> None:
        """
        Set threshold for model.

        Parameters
        ----------
        threshold : float
        """
        self.threshold = threshold

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        naive: bool = False,
        tied_cov: bool = False,
    ) -> None:
        """
        Train the model with given features matrix and labels array.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray

        naive : optional, bool
            Use naive assumption. Default False.

        tied_cov : optional, bool
            Use tied covariance assumption. Default false.

        """
        self.means = []
        self.covariances = []
        for i in range(2):
            selected_datas = features[:, labels == i]
            mean = np.vstack(selected_datas.mean(axis=1, dtype=np.float64))
            cov = np.cov(selected_datas, bias=True)
            # Naive assumption
            if naive:
                cov = np.diag(np.diag(cov))
            self.means.append(mean)
            self.covariances.append(cov)
        if tied_cov:
            self.covariances = [dr.within_class_covariance(features, labels, 2)]

    def _log_likelihood_sample(
        self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> float:
        m = x.shape[0]
        sign, sigma_log_det = np.linalg.slogdet(sigma)
        sigma_inv = np.linalg.inv(sigma)
        dc = x - mu

        return (
            -m / 2 * np.log(2 * np.pi)
            - 0.5 * sigma_log_det
            - 0.5 * np.dot(dc.T, np.dot(sigma_inv, dc)).item()
        )

    def _log_likelihood(self, features: np.ndarray) -> np.ndarray:
        n = features.shape[1]
        result = np.empty([2, n])

        for i in range(n):
            likelihood = np.empty(2)
            for lab in range(2):
                curr_cov = self.covariances[0]
                if len(self.covariances) != 1:
                    curr_cov = self.covariances[lab]
                curr_mean = self.means[lab]
                curr_sample = np.row_stack(features[:, i])
                likelihood[lab] = self._log_likelihood_sample(
                    curr_sample, curr_mean, curr_cov
                )
            result[:, i] = likelihood

        return result

    def predict(
        self, features: np.ndarray, return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply model on feautures and predict label.

        Parameters
        ----------
        features : np.ndarray

        return_scores: optional, bool

        Returns
        -------
        predictions: ndarray

        ratio: optional, ndarray
            returned if return_score == True

        """
        likelihood = self._log_likelihood(features)
        ratio = likelihood[1, :] / likelihood[0, :]
        prediction = (ratio > self.threshold).astype(np.int32)
        if return_scores:
            return prediction, ratio
        return prediction
