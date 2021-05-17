# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:58:53 2021

@author: Hossein.JvdZ
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import itertools as it

import data_loading as dl

sys.path.append("..")


def plot_attributes(features: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot features in a grid of hystograms, divided by labels.

    Parameters
    ----------
    features : np.ndarray

    labels : np.ndarray

    """
    # 2 columns each containing 4 figures, total 8 features
    fig, axes = plt.subplots(2, 4, figsize=(12, 4))

    wrong = features[:, labels == 0]  # Non-accurate data
    pulsar = features[:, labels == 1]  # Accurate data

    ax = axes.ravel()  # flat axes with numpy ravel
    # def plot_test_data():
    for i in range(features.shape[0]):
        # resulation of each axes
        _, bins = np.histogram(features[i, :], bins=35)
        # red color to show false prediction
        ax[i].hist(wrong[i, :], bins=bins, color="r", alpha=0.2)
        # green color to show true prediction
        ax[i].hist(pulsar[i, :], bins=bins, color="g", alpha=0.5)
        # increase fontsize to 16 for better report image
        ax[i].set_title(dl.labels_names[i])
        # the x-axis co-ordinates are not so useful, as we just want to look how
        # well separated the histograms are
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].set_yticks(())
        ax[i].axis(ymax=200)

    ax[0].legend(["Non-Pulsar", "Pulsar"], loc="best", fontsize=10)
    plt.tight_layout()  # let's make good plots
    plt.show()


def scatter_attributes(features: np.ndarray, labels: np.ndarray) -> None:
    features_count = features.shape[0]

    for c in list(it.combinations(range(features_count), 2)):
        _, ax = plt.subplots()

        true_features = features[:, labels == 1]
        false_features = features[:, labels == 0]


        ax.scatter(false_features[c[0]], false_features[c[1]], 
                   label="Non-Pulsar", alpha=0.3,
                   edgecolors="red", facecolors="none")
        ax.scatter(true_features[c[0]], true_features[c[1]], 
                   label="Pulsar", alpha=1,
                   edgecolors="blue", facecolors="none")
        
        ax.set_xlabel(dl.labels_names[c[0]])
        ax.set_ylabel(dl.labels_names[c[1]])

        ax.legend()
        plt.show()
