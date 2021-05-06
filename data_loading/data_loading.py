# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:36:57 2021

@author: darth
"""

from pathlib import Path
from typing import Tuple

import numpy as np

path = Path(__file__)

labels_names = [
    "Integrated profile mean",
    "Integrated profile SD",
    "Integrated profile EK",
    "Integrated profile skewness",
    "DM-SNR curve mean",
    "DM-SNR curve SD",
    "DM-SNR curve EK",
    "DM-SNR curve skewness",
    "Class"
]


def load_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data and divide it in labels and features.

    Returns
    -------
    features : ndarray

    labels: ndarray
    """
    dataset = np.loadtxt(
        path.with_name("Train.txt"),
        delimiter=",").T
    return dataset[:-1, :], dataset[-1, :].astype(np.int32)


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data and divide it in labels and features.

    Returns
    -------
    features : ndarray

    labels: ndarray
    """
    dataset = np.loadtxt(
        path.with_name("Test.txt"),
        delimiter=",").T
    return dataset[:-1, :], dataset[-1, :].astype(np.int32)
