# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:36:57 2021

@author: darth
"""

import numpy as np
from pathlib import Path
from typing import Tuple

path = Path(__file__)

"""
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
"""
labels_name = [
    "mean ip",
    "standard deviation ip",
    "excess kurtosis ip",
    "skewness ip",
    "mean ds",
    "standard deviation ds",
    "excess kurtosis ds",
    "skewness ds"
]

def load_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data.

    Returns
    -------
    data: ndarray
        Rows represent attributes, Column samples
        Dimension (a, s) where a is attributes count, s is samples count.
    labels: ndarray
        Array of labels with dimension (n, )

    """
    
    dataset = np.loadtxt(path.with_name("Train.txt"), delimiter=",")
    return dataset[:, :-1].T, dataset[:, -1]

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data.

    Returns
    -------
    data: ndarray
        Rows represent attributes, Column samples
        Dimension (a, s) where a is attributes count, s is samples count.
    labels: ndarray
        Array of labels with dimension (n, )

    """
    dataset = np.loadtxt(path.with_name("Test.txt"), delimiter=",")
    return dataset[:, :-1].T, dataset[:, -1]