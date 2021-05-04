# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:36:57 2021

@author: darth
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

path = Path(__file__)

column_names = [
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
    "Class"
]


def load_train_data() -> pd.DataFrame:
    """
    Load training data as Dataframe.

    Returns
    -------
    dataset : DataFrame
    """
    dataset = pd.read_csv(
        path.with_name("Train.txt"),
        names=column_names,
        index_col=False)
    return dataset


def load_test_data() -> pd.DataFrame:
    """
    Load test data as Dataframe.

    Returns
    -------
    dataset : DataFrame

    """
    dataset = pd.read_csv(path.with_name("Test.txt"),
                          names=column_names,
                          index_col=False)
    return dataset
