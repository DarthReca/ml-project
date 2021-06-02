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

    def __init__(self, lamb: float, prior_true: float):
        """
        Class for logistic regression.

        Parameters
        ----------
        lamb : float
            lambda hyperparameter.
        """
        self.l = lamb
        self.prior_true = prior_true

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

        return self.l / 2 * np.linalg.norm(w) ** 2 + 1 / n * summatory

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train model with features, labels and lambda.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray
        """
        x0 = np.zeros(features.shape[0] + 1)
        self.obj_funct = fmin_l_bfgs_b(
            self._log_reg_obj,
            x0,
            args=[features, labels],
            approx_grad=True,
            maxiter=1000,
            maxfun=1000,
        )

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
        w, b = self.obj_funct[0][:-1], self.obj_funct[0][-1]
        n = features.shape[1]
        scores = np.full(n, b)
        for i in range(n):
            x_i = features[:, i]
            scores[i] += np.dot(w.T, x_i)
        pred = (scores > 0).astype(np.int32)
        if return_scores:
            return pred, scores
        return pred
