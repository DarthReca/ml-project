# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:23:41 2021

@author: DarthReca
"""

import numpy as np
from scipy.linalg import eigh  # type: ignore


def within_class_covariance(
    data: np.ndarray, labels: np.ndarray, label_count: int
) -> np.ndarray:
    """
    Compute within class covariance of data.

    Parameters
    ----------
    data : np.ndarray

    labels : np.ndarray

    label_count : int


    Returns
    -------
    within class covariance: ndarray

    """
    sw = np.zeros((data.shape[0], data.shape[0]))
    for i in range(label_count):
        selected = data[:, labels == i]
        sw += np.cov(selected, bias=True) * float(selected.shape[1])
    return sw / float(data.shape[1])


def between_class_covariance(
    data: np.ndarray, labels: np.ndarray, label_count: int
) -> np.ndarray:
    """
    Compute between class covariance of data.

    Parameters
    ----------
    data : np.ndarray

    labels : np.ndarray

    label_count : int


    Returns
    -------
    between class covariance: ndarray

    """
    sb = np.zeros((data.shape[0], data.shape[0]))
    mu = np.row_stack(data.mean(axis=1))
    for i in range(label_count):
        selected = data[:, labels == i]
        muc = np.row_stack(selected.mean(axis=1))
        muc -= mu
        sb += float(selected.shape[1]) * np.dot(muc, muc.T)
    return sb / float(data.shape[1])


def lda(
    data: np.ndarray, labels: np.ndarray, label_count: int, dimensions: int
) -> np.ndarray:
    """
    Perform linear discriminant analysis over data.

    Parameters
    ----------
    data : np.ndarray

    labels : np.ndarray

    label_count : int

    dimensions : int
        Dimensions to obtain.

    Returns
    -------
    compressed_data: ndarray
        Input data projected over new subspace

    """
    sw = within_class_covariance(data, labels, label_count)
    sb = between_class_covariance(data, labels, label_count)
    _, U = eigh(sb, sw)
    W = np.fliplr(U)[:, :dimensions]
    return np.dot(W.T, data)
