# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:52:40 2021

@author: DarthReca
"""

from typing import List, Tuple

import numpy as np
from scipy.special import logsumexp


class GaussianMixtureModel:
    """
    Class for Gaussian Mixture Model.

    Parameters
    ----------
    eigvals_bound : float, optional
        Inferior limit for eigvals of covariance. The default is 0.01.
    displacement_factor : float, optional
        LBG displacement factor. The default is 0.1.
    precision : float, optional
        Precision for estimating GMM. The default is 1e-6.
    """

    def __init__(
        self,
        eigvals_bound: float = 0.01,
        displacement_factor: float = 0.1,
        precision: float = 1e-6,
    ):
        """
        Class for Gaussian Mixture Model.

        Parameters
        ----------
        eigvals_bound : float, optional
            Inferior limit for eigvals of covariance. The default is 0.01.
        displacement_factor : float, optional
            LBG displacement factor. The default is 0.1.
        precision : float, optional
            Precision for estimating GMM. The default is 1e-6.
        """
        self.eigvals_bound = eigvals_bound
        self.displacement_factor = displacement_factor
        self.precision = precision

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

    def _bound_cov(self, cov: np.ndarray) -> np.ndarray:
        u, s, _ = np.linalg.svd(cov)
        s[s < self.eigvals_bound] = self.eigvals_bound
        s = np.diag(s)
        return u.dot(s.dot(u.T))

    def _log_joints_gmm(self, x: np.ndarray, gmm: List) -> np.ndarray:
        d, n = x.shape
        m = len(gmm)
        s = np.empty([m, n])  # joint log densities
        for i in range(n):
            for g in range(m):
                mu = gmm[g][1]
                cov = gmm[g][2]
                ll = self._log_likelihood_sample(x[:, i].reshape([d, 1]), mu, cov)
                s[g, i] = ll + np.log(gmm[g][0])
        return s

    def _estimate_gmm(
        self,
        features: np.ndarray,
        initial_estimate: List,
        tied: bool = False,
        diag: bool = False,
    ) -> List:
        m = len(initial_estimate)
        n = features.shape[1]
        r = features.shape[0]

        current_params = initial_estimate.copy()
        ll_current = np.NaN

        while True:

            # E Step

            ll_previous = ll_current

            joints = self._log_joints_gmm(features, current_params)
            marginals = logsumexp(joints, axis=0)

            ll_current = marginals.sum() / n

            if np.abs(ll_current - ll_previous) < self.precision:
                return current_params

            joints -= marginals
            post_prob = np.exp(joints)

            # M STEP

            Z = post_prob.sum(axis=1)
            F = np.zeros([m, features.shape[0]])
            S = []
            for g in range(m):
                S.append(np.zeros([r, r]))
                for i in range(n):
                    x = features[:, i].reshape([r, 1])
                    F[g] += (post_prob[g, i] * x.T)[0]
                    S[g] += post_prob[g, i] * x.dot(x.T)

            for g in range(m):
                w_t1 = Z[g] / Z.sum()
                mu_t1 = (F[g] / Z[g]).reshape([r, 1])
                cov_t1 = (S[g] / Z[g]) - mu_t1.dot(mu_t1.T)

                if diag:
                    cov_t1 = np.diag(np.diag(cov_t1))

                cov_t1 = self._bound_cov(cov_t1)

                current_params[g] = (w_t1, mu_t1, cov_t1)

            # Tied Covariance
            if tied:
                tied_cov = np.zeros([r, r])
                for g in range(m):
                    tied_cov += Z[g] * current_params[g][2]
                tied_cov /= n
                tied_cov = self._bound_cov(tied_cov)
                for g in range(m):
                    mu = current_params[g][1]
                    w = current_params[g][0]
                    current_params[g] = (w, mu, tied_cov)

    def _lbg(self, features: np.ndarray, num_gaussians: int):
        n = features.shape[0]

        mean = features.mean(axis=1).reshape([n, 1])
        cov = np.cov(features)
        cov = self._bound_cov(cov)

        gmm_1 = [(1.0, mean, cov)]

        for _ in range(int(num_gaussians / 2)):
            gmm = []
            for p in gmm_1:
                w = p[0] / 2
                cov = p[2]

                u, s, _ = np.linalg.svd(cov)
                d = u[:, 0:1] * s[0] ** 0.5 * self.displacement_factor
                mean = p[1]

                gmm.append((w, mean + d, cov))
                gmm.append((w, mean - d, cov))
            gmm_1 = self._estimate_gmm(features, gmm)

        return gmm_1

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_gaussians: int,
        labels_count: int,
    ) -> None:
        """
        Train model with given features and labels.

        Parameters
        ----------
        features : np.ndarray

        labels : np.ndarray

        num_gaussians : int
            number of components for each gaussian mixture.
        labels_count : int

        """
        self.gaussian_mixtures = []
        for l in range(labels_count):
            curr = features[:, labels == l]
            params = self._lbg(curr, num_gaussians)
            self.gaussian_mixtures.append(params)

    def predict(self, features: np.ndarray):
        """Predict labels for given features."""
        labels_count = len(self.gaussian_mixtures)
        log_densities = np.empty([labels_count, features.shape[1]])
        for l in range(labels_count):
            gmm = self.gaussian_mixtures[l]
            gmm_joints = self._log_joints_gmm(features, gmm)
            gmm_marginals = logsumexp(gmm_joints, axis=0)
            log_densities[l] = gmm_marginals
        joints = log_densities + np.log(1 / 3)
        marginal = logsumexp(joints, axis=0)
        joints -= marginal
        return joints.argmax(axis=0)
