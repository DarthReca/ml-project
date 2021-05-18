# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:22:43 2021

@author: DarthReca
"""

import data_loading as dl
import data_plotting as dp
import data_result_analysis as dra
import dimensionality_reduction as dr
import numpy as np


def main():
    test_features, test_labels = dl.load_test_data()
    train_features, train_labels = dl.load_train_data()

    """
    priors = np.linspace(0.1, 0.5, num=20)
    cms = []
    for p in priors:
        pred = np.load("./saved_data/gaussian-{}.npy".format(p))
        cms.append(dra.confusion_matrix(test_labels, pred))
    dra.thresholds_error_rates(priors, cms)
    """


if __name__ == "__main__":
    main()