# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:16:11 2021

@author: DarthReca
"""

from typing import Tuple

import numpy as np


def tpr_fnr(cm: np.ndarray) -> Tuple[float, float]:
    """Compute true positive rate and false negative rate."""
    tpr = cm[1, 1] / cm[:, 1].sum()
    return tpr, 1 - tpr


def tnr_fpr(cm: np.ndarray) -> Tuple[float, float]:
    """Compute true negative rate and false positive rate."""
    fpr = cm[1, 0] / cm[:, 0].sum()
    return 1 - fpr, fpr


def precision(cm: np.ndarray) -> float:
    """Compute precision (how many positive are classified correctly)."""
    tp = cm[1, 1]
    fp = cm[1, 0]
    return tp / (tp + fp)


def recall(cm: np.ndarray) -> float:
    """Compute recall (how well positive class is predicted)."""
    tp = cm[1, 1]
    fn = cm[0, 1]
    return tp / (tp + fn)


def f_beta_score(confusion_matrix: np.ndarray, beta: float) -> float:
    """
    Combine recall and precision for unbalanced dataset.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        DESCRIPTION.
    beta : float
        Best values are: 
            - 1 (mportant fn, fp
            - 2 important false negative
            - 0.5 important false positive

    Returns
    -------
    float

    """
    prec = precision(confusion_matrix)
    rec = recall(confusion_matrix)
    return ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec)

def matthew_corr_coeff(cm: np.ndarray) -> float:
    """-1 not good 1 good"""
    pass

def confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix from predicted labels and true labels.

    Parameters
    ----------
    true_labels : ndarray
        Real labels to make test on.
    predicted_labels : ndarray
        Labels predicted by model.

    Returns
    -------
    confusion_matrix : ndarray
        element[i, j] is predicted as part of i class, but its class is j
    """
    # To be sure we have 1-D array
    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()

    labels_count = np.amax(true_labels).item() + 1
    conf_mat = np.empty([labels_count, labels_count], dtype=np.int32)
    for pred in range(labels_count):
        for real in range(labels_count):
            tl = true_labels == real
            pl = predicted_labels == pred
            conf_mat[pred][real] = np.logical_and(tl, pl).sum()
    return conf_mat
