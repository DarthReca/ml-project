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
import preprocess as prep


def main():
    train_features, train_labels = dl.load_train_data()
    
    train_features = prep.apply_all_preprocess(train_features)   

    dp.plot_attributes(train_features, train_labels)
    # dp.scatter_attributes(train_features, train_labels)


if __name__ == "__main__":
    main()
