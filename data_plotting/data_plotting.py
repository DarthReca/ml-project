# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:58:53 2021

@author: Hossein.JvdZ
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_loading as dl

sys.path.append("..")

def plot_attributes(data: pd.DataFrame):

    target_dataframe = data['Class']  # Possibility of existence

    # Emit Possibility of existence the Pulsar to use it just in
    # target_dataframe
    pulsar_star = data.drop(columns='Class')

    # 2 columns each containing 4 figures, total 8 features
    fig, axes = plt.subplots(2, 4, figsize=(12, 4))

    wrong = pulsar_star[target_dataframe == 0]  # Non-accurate data
    pulsar = pulsar_star[target_dataframe == 1]  # Accurate data

    ax = axes.ravel()  # flat axes with numpy ravel
    # def plot_test_data():
    for i in range(pulsar_star.columns.size):
        # resulation of each axes
        _, bins = np.histogram(pulsar_star.iloc[:, i], bins=35)
        # red color to show false prediction
        ax[i].hist(wrong.iloc[:, i], bins=bins, color='r', alpha=0.2)
        # green color to show true prediction
        ax[i].hist(pulsar.iloc[:, i], bins=bins, color='g', alpha=0.5)
        # increase fontsize to 16 for better report image
        ax[i].set_title(pulsar_star.columns[i], fontsize=18)
        # the x-axis co-ordinates are not so useful, as we just want to look how
        # well separated the histograms are
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].set_yticks(())
        ax[i].axis(ymax=200)

    ax[0].legend(['Wrong', 'Pulsar'], loc='best', fontsize=10)
    plt.tight_layout()  # let's make good plots
    plt.show()
