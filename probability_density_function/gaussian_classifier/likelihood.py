# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:40:19 2021

@author: DarthReca
"""

from typing import List, Tuple

import numpy as np


def mvg_log_likelihood_sample(x: np.ndarray, mu: np.ndarray,
                              sigma: np.ndarray) -> float:
    """Compute likelihood of mvg for a single sample."""
    m = x.shape[0]
    sign, sigma_log_det = np.linalg.slogdet(sigma)
    sigma_inv = np.linalg.inv(sigma)
    dc = x - mu

    return (-m / 2 * np.log(2 * np.pi) - 0.5 * sigma_log_det
            - 0.5 * np.dot(dc.T, np.dot(sigma_inv, dc)).item())


def mvg_log_likelihood(datas: np.ndarray,
                       mean: List[np.ndarray],
                       covariance: List[np.ndarray],
                       label_count: int) -> np.ndarray:
    """
    Calculate log(likelihood) for all classes with given data.

    Parameters
    ----------
    datas : ndarray
        size of (a, n) where a is attributes count and n is samples count

    mean: list
        list of ndarray with size (l, 1) where l is label count

    covariance: ndarray
        list of ndarray with size (l, l)

    label_count: int

    Returns
    -------
    result : ndarray
        element i,j is likelihood of sample j for class i

    """
    a = datas.shape[0]
    if len(mean) != label_count:
        raise ValueError("mean must contain {} ndarray".format(label_count))
    if len(covariance) != label_count and len(covariance) != 1:
        raise ValueError(
            "covariance must contain {} ndarray".format(label_count))
    for i in range(label_count):
        if mean[i].shape != (a, 1):
            raise ValueError("mean[{}] shape must be ({}, 1)".format(i, a))
        if len(covariance) > i and covariance[i].shape != (a, a):
            raise ValueError("cov[{}] shape must be ({}, {})".format(i, a, a))

    result: List[np.ndarray] = []
    for i in range(datas.shape[1]):
        likelihood = np.zeros((label_count, 1))
        for lab in range(label_count):
            curr_cov = covariance[0] if len(
                covariance) == 1 else covariance[lab]
            likelihood[lab] = mvg_log_likelihood_sample(
                np.row_stack(datas[:, i]), mean[lab], curr_cov)
        result.append(likelihood)

    return np.column_stack(result)


def mvg_likelihood(datas: np.ndarray,
                   mean: List[np.ndarray],
                   covariance: List[np.ndarray],
                   label_count: int) -> np.ndarray:
    """
    Calculate likelihood for all classes with given data.

    Parameters
    ----------
    datas : ndarray
        size of (c, n) where c is classes count and n is samples count
    mean: list
        list of ndarray with size (c, 1)

    covariance: ndarray
        list of ndarray with size (c, c)

    Returns
    -------
    result : ndarray
        element i,j is log_likelihood of sample j for class i

    """
    return np.exp(mvg_log_likelihood(datas, mean, covariance, label_count))
