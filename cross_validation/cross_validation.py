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
    
    positive_samples = dataset[:, dataset[-1] == 1]
    negative_samples = dataset[:, dataset[-1] == 0]
    
    pos_perc = positive_samples.shape[1]/dataset.shape[1]
    
    rng = np.random.default_rng(seed=0)
    rng.shuffle(positive_samples, axis=1)
    rng.shuffle(negative_samples, axis=1)

    s_features = []
    s_labels = []
    
    sample_size = int(features.shape[1] / samples)
    
    positive_size = int(sample_size * pos_perc)
    negative_size = sample_size - positive_size
    
    for i in range(samples):
        start_positive = i * positive_size
        start_negative = i * negative_size
        
        end_positive = start_positive + positive_size
        end_negative = start_negative + negative_size
        
        ps = positive_samples[:, start_positive: end_positive]
        ns = negative_samples[:, start_negative: end_negative]
        
        s = np.hstack([ps, ns])
        
        s_features.append(s[:-1])
        s_labels.append(s[-1].astype(np.int32))

    return s_features, s_labels


def train_validation_sets(
    sampled_f: np.ndarray, sampled_l: np.ndarray, validation_index: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Merge samples to obtain train and validation sets.

    Parameters
    ----------
    sampled_f : np.ndarray
        sampled feautures obtained from `shuffle sample`.
    sampled_l : np.ndarray
        sampled labels obtained from `shuffle_sample`.
    validation_index : int
        index of the validation set.

    Returns
    -------
    (tr_set, val_set)
        Each set is a tuple with features and labels.

    """    
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
