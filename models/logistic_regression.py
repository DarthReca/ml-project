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

    def __init__(self, lamb: float):
        """
        Class for logistic regression.

        Parameters
        ----------
        lamb : float
            lambda hyperparameter.
        """
        self.l = lamb

    def _log_reg_obj(self, v: np.ndarray, features: np.ndarray, labels: np.ndarray):
        w, b = v[:-1], v[-1]
        n = features.shape[1]

        summatory = 0
        for i in range(n):
            x_i = features[:, i]

            log_sigmoid = np.log1p(np.exp(-b - np.dot(w.T, x_i)))
            minus_log_sigmoid = np.log1p(np.exp(b + np.dot(w.T, x_i)))

            summatory += labels[i] * log_sigmoid + (1 - labels[i]) * minus_log_sigmoid

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
            self._log_reg_obj, x0, args=[features, labels], approx_grad=True
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
        scores = np.empty(n)
        for i in range(n):
            x_i = features[:, i]
            scores[i] = np.dot(w.T, x_i) + b
        pred = (scores > 0).astype(np.int32)
        if return_scores:
            return pred, scores
        return pred
