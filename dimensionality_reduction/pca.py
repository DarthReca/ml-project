# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:28:01 2021

@author: DarthReca
"""

import numpy as np


def pca(data: np.ndarray, m: int) -> np.ndarray:
    """
    Perform principal componenet anaylisis on data reducing to m dimensions.

    Parameters
    ----------
    data : np.ndarray

    m : int
        dimesions to obtain

    Raises
    ------
    Exception
        if m is more than data class count

    Returns
    -------
    reduced: ndarray

    """
    if m > data.shape[1]:
        raise Exception("m must be less than {}".format(data.shape[1]))
    C = np.cov(data, bias=True)
    _, U = np.linalg.eigh(C)
    P = np.fliplr(U)[:, :m]
    return np.dot(P.T, data)
