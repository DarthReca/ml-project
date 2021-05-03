# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:36:57 2021

@author: darth
"""

import numpy as np
from pathlib import Path
from typing import Tuple

path = Path(__file__)

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
    return dataset[:, :-2].T, dataset[:, -1]

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
    return dataset[:, :-2].T, dataset[:, -1]