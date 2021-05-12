# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:53:59 2021

@author: darth
"""

import numpy as np
from typing import Tuple
from .confusion_matrix import confusion_matrix

def detection_cost_function(
        conf_m: np.ndarray,
        prior_prob_true: float,
        cost_fn: float,
        cost_fp: float,
        normalize: bool = True
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

    br = ( prior_prob_true * cost_fn * fnr +
        (1 - prior_prob_true) * cost_fp * fpr )

    optimal_cost = np.minimum(
        cost_fn * prior_prob_true,
        cost_fp * (1 - prior_prob_true)
    )

    if normalize:
        br /= optimal_cost

    return br

def min_detection_cost_function(
        llr_labels: Tuple[np.ndarray, np.ndarray],
        prior_prob_true: float,
        cost_fn: float,
        cost_fp: float) -> float:
    """m."""
    llr, labels = llr_labels
    threshold = np.arange(-100, 100, 1)
    
    min_br = []
    for t in threshold:
        selected_class = (llr > t).astype(int)
        cm = confusion_matrix(labels, selected_class)
        br = detection_cost_function(cm, prior_prob_true, cost_fn, cost_fp)
        min_br.append(br)

    return min(min_br)