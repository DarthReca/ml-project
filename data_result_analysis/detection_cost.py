# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:53:59 2021

@author: darth
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .confusion_matrix import confusion_matrix
from .error_curves import roc_det_curves


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
    plot_roc_det: bool = False,
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
    conf_matrixes = []

    for t in np.sort(scores, kind="mergesort"):
        pred = (scores > t).astype(int)
        cm = confusion_matrix(labels, pred)
        conf_matrixes.append(cm)
        risks.append(dcf(cm, prior_prob_true, cost_fn, cost_fp))

    if plot_roc_det:
        roc_det_curves(conf_matrixes)

    return min(risks)


def bayes_error_plot(
    scores: np.ndarray,
    true_labels: np.ndarray,
    range_values: Tuple[float, float] = (0, 0),
) -> None:
    """
    Show bayes errors plot.

    Parameters
    ----------
    scores : np.ndarray
        Scores obtained through a model.
    true_labels : np.ndarray
        Real labels.
    range_values : Tuple[float, float], optional
        The range of prior log odds to plot. The default is (0, 0).
    """
    max_value = range_values[1]
    min_value = range_values[0]
    if range_values == (0, 0):
        max_value = scores.max()
        min_value = scores.min()
    prior_log_odds = -np.linspace(min_value, max_value, 20)
    
    dcfs = []
    min_dcfs = []
    priors = []

    for t in prior_log_odds:
        prior = 1 / (1 + np.exp(-t))
        priors.append(prior)
        pred = (scores > -t).astype(int)
        cm = confusion_matrix(true_labels, pred)
        dcfs.append(dcf(cm, prior, 1, 1))
        min_dcfs.append(min_norm_dcf(scores, true_labels, prior, 1, 1))

    plt.plot(prior_log_odds, dcfs, label="DCF", color="r")
    plt.plot(prior_log_odds, min_dcfs, label="min DCF", color="b")
    plt.xlabel("Prior log odds")
    plt.ylabel("DCF")

    plt.xticks(prior_log_odds, rotation="vertical")

    plt.legend()
    plt.show()
