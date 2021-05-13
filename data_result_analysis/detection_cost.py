# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:53:59 2021

@author: darth
"""

from typing import Tuple

import numpy as np

from .confusion_matrix import confusion_matrix


def dcf(
    conf_m: np.ndarray,
    prior_prob_true: float,
    cost_fn: float,
    cost_fp: float,
    normalize: bool = True,
) -> float:
    """
    Compute bayes risk for given confusion matrix.

    Parameters
    ----------
    conf_m : np.ndarray
        Confusion matrix.
    prior_prob_true : float
        Prior probability of true class.
    cost_fn : float
        Cost of false negative.
    cost_fp : float
        Cost of false positive.
    normalize : optional, bool
        Default value is True
    Returns
    -------
    bayes_risk: float

    """
    fnr = conf_m[0, 1] / conf_m[:, 1].sum()
    fpr = conf_m[1, 0] / conf_m[:, 0].sum()

    risk = prior_prob_true * cost_fn * fnr + (1 - prior_prob_true) * cost_fp * fpr

    optimal_cost = min(cost_fn * prior_prob_true, cost_fp * (1 - prior_prob_true))

    if normalize:
        risk /= optimal_cost

    return risk


def min_norm_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    prior_prob_true: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """
    Compute minimum normalized bayes risk.

    Parameters
    ----------
    scores : np.ndarray
        Computed scores (likelihood ratio) for samples.
    labels : np.ndarray
        Labels associated with samples.
    prior_prob_true : float
        Prior probability of true class.
    cost_fn : float
        Cost of false negative.
    cost_fp : float
        Cost of false positive.

    Returns
    -------
    min_dcf: float

    """
    risks = []

    for t in scores:
        pred = (scores > t).astype(int)
        cm = confusion_matrix(labels, pred)
        risks.append(dcf(cm, prior_prob_true, cost_fn, cost_fp))

    return min(risks)
