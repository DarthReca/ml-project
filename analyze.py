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

def analize_correlation():
    train_features, train_labels = dl.load_train_data()
    
    pcc = np.corrcoef(train_features)
    
    dp.plot_matrix(pcc, "Reds")
    
    train_features = prep.apply_all_preprocess(train_features)   
    
    pcc = np.corrcoef(train_features)
    
    dp.plot_matrix(pcc, "Reds")
    
def analize_risk():
    train_features, train_labels = dl.load_train_data()


def main():
    analize_correlation()


if __name__ == "__main__":
    main()
