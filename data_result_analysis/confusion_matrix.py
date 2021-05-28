# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:16:11 2021

@author: DarthReca
"""

import numpy as np


def confusion_matrix(
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
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
