# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:55:54 2021

@author: DarthReca
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class SupportVectorMachine:
    """
    Class for SVM.

    Attributes
    ----------
    k: float
        hyperparameter
    C: float
        hyperparameter
    kernel_type: str
        Choose between 'polynomial' or 'radial basis function'.
        Default is 'polynomial'.
    kernel_grade: float
        Exponent of kernel
        Default is 1.0
    pol_kernel_c: float
        If kernel_type is 'polynomial' this the c hyperparameter of polynomial kernel
        Default is 0.0
    """

    def __init__(
        self,
        k: float,
        C: float,
        kernel_type: str = "polynomial",
        kernel_grade: float = 1.0,
        pol_kernel_c: float = 0.0,
    ):
        """
        Class for SVM.

        Attributes
        ----------
        k: float
            hyperparameter
        C: float
            hyperparameter
        kernel_type: optional, str
            Choose between 'polynomial' or 'radial basis function'.
            Default is 'polynomial'.
        kernel_grade: optional, float
            Exponent of kernel.
            Default is 1.0
        pol_kernel_c: optional, float
            If kernel_type is 'polynomial' this the c hyperparameter of polynomial kernel.
            Default is 0.0.
        """
        self.k = k
        self.C = C

        self.kernel_grade = kernel_grade
        self.pol_kernel_c = pol_kernel_c
        self.kernel_funct = self._polynomial_kernel
        if kernel_type == "radial basis function":
            self.kernel_funct = self._radial_basis_function_kernel

    def _polynomial_kernel(self, sample_i: np.ndarray, sample_j: np.ndarray) -> float:
        return (np.dot(sample_i.T, sample_j) + self.pol_kernel_c) ** self.kernel_grade

    def _radial_basis_function_kernel(
        self, sample_i: np.ndarray, sample_j: np.ndarray
    ) -> float:
        norm = np.linalg.norm(sample_i - sample_j) ** 2
        return np.exp(-self.kernel_grade * norm)

    def _binary_cross_entropy(
        self,
        samples: np.ndarray,
        z_labels: np.ndarray,
    ) -> np.ndarray:
        lab_count = z_labels.shape[0]
        entropy = np.empty([lab_count, lab_count])

        for i in range(lab_count):
            for j in range(lab_count):
                entropy[i, j] = z_labels[i] * z_labels[j]
                entropy[i, j] *= (
                    self.kernel_funct(samples[:, i], samples[:, j]) + self.k ** 2
                )

        return entropy

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the model on given features and labels.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray

        """
        samples_count = features.shape[1]
        self.z_labels = 2 * labels - 1

        cross_entropy = self._binary_cross_entropy(features, self.z_labels)

        b = [(0, self.C) for _ in range(samples_count)]
        sol = fmin_l_bfgs_b(
            self._inverse_obj_funct,
            np.zeros(samples_count),
            args=[cross_entropy],
            bounds=b,
            factr=1.0,
        )

        # Keep only useful values
        to_keep = sol[0] > 0

        self.z_labels = self.z_labels[to_keep]
        self.samples = features[:, to_keep]
        self.alpha = sol[0][to_keep]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels of given features.

        Parameters
        ----------
        features : np.ndarray

        Returns
        -------
        predictons: np.ndarray
        """
        samples_count = features.shape[1]

        scores = np.empty(samples_count)
        for t in range(samples_count):
            current_sample = features[:, t]
            summatory = 0.0
            for i in range(self.samples.shape[1]):
                kern = (
                    self.kernel_funct(self.samples[:, i], current_sample) + self.k ** 2
                )
                summatory += self.alpha[i] * self.z_labels[i] * kern
            scores[t] = summatory

        return (scores > 0).astype(np.int32)

    def _inverse_obj_funct(self, alpha: np.ndarray, cross_entropy: np.ndarray):
        samples_count = alpha.shape[0]

        L = 0.5 * np.dot(np.dot(alpha.T, cross_entropy), alpha) - np.dot(
            alpha.T, np.ones(samples_count)
        )
        gradient = np.dot(cross_entropy, alpha) - np.ones(samples_count)
        return L, gradient
