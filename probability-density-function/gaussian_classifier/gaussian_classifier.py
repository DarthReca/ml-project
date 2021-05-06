# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:30:11 2021

@author: gino9
"""

import sys
from typing import List, Tuple, Union

import numpy as np
from scipy.special import logsumexp

from dimensionality_reduction import within_class_covariance

from .likelihood import mvg_likelihood, mvg_log_likelihood

sys.path.append("..")


def gaussian_estimate_stats(
    datas: np.ndarray, labels: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each class compute mean and covariance matrix.

    Parameters
    ----------
    datas : ndarray
        size(c, n), where c is classes count and n is samples count

    labels : ndarray
        size (1, n), where n is samples count

    Returns
    -------
    mean : List[ndarray]
        mean[i] refers to class i.
    cov : List[ndarray]
        cov[i] refers to class i.
    """
    if labels.shape != (1, datas.shape[1]):
        raise ValueError("labels shape must be (1, {})".format(datas.shape[1]))

    means: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    for i in range(3):
        selected_datas = datas[:, labels[0] == i]
        mean = np.row_stack(selected_datas.mean(axis=1, dtype=np.float64))
        cov = np.cov(selected_datas)
        means.append(mean)
        covariances.append(cov)
    return means, covariances


def naive_gaussian_estimate_stats(
    datas: np.ndarray, labels: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each class compute mean and covariance matrix, using naive assumption.

    Parameters
    ----------
    datas : np.ndarray
        size(c, n), where c is classes count and n is samples count
    labels : np.ndarray
        size (1, n), where n is samples count

    Returns
    -------
    mean : List[ndarray]
        mean[i] refers to class i.
    cov : List[ndarray]
        cov[i] refers to class i.
    """
    mean, cov = gaussian_estimate_stats(datas, labels)
    cov = [np.diag(np.diag(x)) for x in cov]
    return mean, cov


def tied_cov_estimate_stats(
    datas: np.ndarray, labels: np.ndarray, label_count: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    For each class compute mean and covariance matrix, using tied covariance assumption.

    Parameters
    ----------
    datas : np.ndarray
        size(c, n), where c is classes count and n is samples count
    labels : np.ndarray
        size (1, n), where n is samples count
    label_count : int

    Returns
    -------
    mean : List[ndarray]
        mean[i] refers to class i.
    cov : List[ndarray]
        cov[i] refers to class i.

    """
    wc_cov = within_class_covariance(datas, labels, label_count)
    mean, _ = gaussian_estimate_stats(datas, labels)
    return mean, [wc_cov]


def class_posterior_prob(likelihood: np.ndarray) -> np.ndarray:
    """Return posterior probability for each class starting from likelihood."""
    joint_prob = 1 / 3 * likelihood
    marginal_densities = joint_prob.sum(axis=0)
    post_prob = []
    for i in range(marginal_densities.shape[0]):
        post_prob.append(joint_prob[:, i] / marginal_densities[i])
    return np.column_stack(post_prob)


def log_class_posterior_prob(
        log_likelihood: np.ndarray,
        prior_prob: float) -> np.ndarray:
    """Return log posterior probability for each class from log likelihood."""
    joint_prob = log_likelihood + np.log(prior_prob)
    marginal_densities = logsumexp(joint_prob, axis=0)
    post_prob = []
    for i in range(marginal_densities.shape[0]):
        post_prob.append(joint_prob[:, i] - marginal_densities[i])
    return np.column_stack(post_prob)


def apply_model(
    data: np.ndarray,
    mean: List[np.ndarray],
    covariance: List[np.ndarray],
    label_count: int,
) -> Union[np.signedinteger, np.ndarray]:
    """Apply model on given data."""
    likelihood = mvg_log_likelihood(data, mean, covariance, label_count)
    posterior_prob = log_class_posterior_prob(likelihood, 1 / 3)
    return posterior_prob.argmax(axis=0)
