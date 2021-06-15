# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:28:09 2021

@author: darthreca
"""

from typing import Tuple, Union

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class LogisticRegression:
    """
    Class for logistic regression.

    Attributes
    ----------
    lamb: float
        lambda hyperparameter
    """

    def __init__(self, lamb: float, prior_true: float, quadratic: bool = False):
        """
        Class for logistic regression.

        Parameters
        ----------
        lamb : float
            lambda hyperparameter.
        prior_true: float
            prior probability true class
        quadratic: optional, bool
            Use quadratic logistic regression
        """
        self.l = lamb
        self.prior_true = prior_true
        self.quadratic = quadratic
        self.threshold = 0.0

    def _log_reg_obj(self, v: np.ndarray, features: np.ndarray, labels: np.ndarray):
        w, b = v[:-1], v[-1]
        n = features.shape[1]
        z = 2 * labels - 1

        pos_f = features[:, labels == 1]
        neg_f = features[:, labels == 0]
        nt = pos_f.shape[1]
        nf = neg_f.shape[1]

        summatory_neg = 0
        for i in range(nf):
            x_i = features[:, i]
            exponent = -z[i] * (b + w.dot(x_i))
            summatory_neg += np.log1p(np.exp(exponent))
        summatory_pos = 0
        for i in range(nt):
            x_i = features[:, i]
            exponent = -z[i] * (b + w.dot(x_i))
            summatory_pos += np.log1p(np.exp(exponent))

        summatory = (
            1 - self.prior_true
        ) / nf * summatory_neg + self.prior_true / nt * summatory_pos

        return self.l / 2 * np.linalg.norm(w) ** 2 + summatory

    def _map_to_quad_space(self, features: np.ndarray) -> np.ndarray:
        r, n = features.shape
        mapped = np.empty([r*r+r ,n])
        for i in range(n):
            x_i = features[:, i].reshape([r ,1])
            mat = x_i.dot(x_i.T)
            mat = mat.flatten("F")
            mapped[:, i] = np.vstack([mat.reshape([r**2, 1]), x_i])[:, 0]
        return mapped

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train model with features, labels and lambda.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray
        """
        x0 = np.zeros(features.shape[0] + 1)
        mapped = features
        
        r, n = features.shape
        if self.quadratic :
            mapped = self._map_to_quad_space(features)
            x0 = np.zeros([r*r+r+1])
        
        self.obj_funct = fmin_l_bfgs_b(
            self._log_reg_obj,
            x0,
            args=[mapped, labels],
            approx_grad=True,
            factr=1e7
        )

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def predict(
        self, features: np.ndarray, return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict labels from features.

        Parameters
        ----------
        features : np.ndarray

        return_scores: optional, bool
            Default is False.

        Returns
        -------
        labels: ndarray
            Predicted labels.
        scores: optional, ndarray
            Returned if `return_scores` == True

        """
        mapped = features
        if self.quadratic:
            mapped = self._map_to_quad_space(features)
        w, b = self.obj_funct[0][:-1], self.obj_funct[0][-1]
        n = mapped.shape[1]
        scores = np.full(n, b)
        for i in range(n):
            x_i = mapped[:, i]
            scores[i] += np.dot(w.T, x_i)
        pred = (scores >= self.threshold).astype(np.int32)
        if return_scores:
            return pred, scores
        return pred
