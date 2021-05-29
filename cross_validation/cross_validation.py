# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:48:43 2021

@author: DarthReca
"""

from typing import List, Tuple

import numpy as np


def shuffle_sample(
    features: np.ndarray, labels: np.ndarray, samples: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Shuffle feautures and labels consistently and create samples.

    Parameters
    ----------
    features : np.ndarray

    labels : np.ndarray

    samples : int
        Number of partitions.

    Returns
    -------
    sampled_features : List[np.ndarray]

    sampled_labels : List[np.ndarray]

    """
    dataset = np.vstack([features, labels])
    rng = np.random.default_rng()
    rng.shuffle(dataset, axis=1)

    s_features = []
    s_labels = []
    sample_size = int(features.shape[1] / samples)
    for i in range(samples):
        start = i * sample_size
        s = dataset[:, start: (start + sample_size)]
        s_features.append(s[:-1])
        s_labels.append(s[-1].astype(np.int32))

    return s_features, s_labels


def train_validation_sets(
    sampled_f: np.ndarray, sampled_l: np.ndarray, validation_index: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    val_feat = sampled_f[validation_index]
    val_lab = sampled_l[validation_index]
    samples = len(sampled_f)

    tr_feat = np.hstack(
        [sampled_f[x] for x in range(samples) if not validation_index == x]
    )
    tr_lab = np.hstack(
        [sampled_l[x] for x in range(samples) if not validation_index == x]
    )
    return (tr_feat, tr_lab), (val_feat, val_lab)
