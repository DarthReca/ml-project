# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:22:43 2021

@author: DarthReca
"""

import data_loading as dl
import data_plotting as dp
import data_result_analysis as dra
import dimensionality_reduction as dr
import cross_validation as cv
import numpy as np
import preprocess as prep

def analize_correlation():
    train_features, train_labels = dl.load_train_data()
    pcc = np.corrcoef(train_features)
    dp.plot_matrix(pcc, "Greens")
    # Preprocessed
    preprocessed = prep.apply_all_preprocess(train_features)   
    pcc = np.corrcoef(preprocessed)
    dp.plot_matrix(pcc, "Reds")
    #Gaussianized
    gaussianized = prep.gaussianize(train_features)
    pcc = np.corrcoef(gaussianized)
    dp.plot_matrix(pcc, "Greens")
    #Preprocessed and Gaussianized
    gaussianized = prep.gaussianize(preprocessed)
    pcc = np.corrcoef(gaussianized)
    dp.plot_matrix(pcc, "Reds")
    
def analize_class_correlation():
    train_features, train_labels = dl.load_train_data()
    train_features = prep.apply_all_preprocess(train_features)
    
    pcc = np.corrcoef(train_features[:, train_labels == 1])
    dp.plot_matrix(pcc, "Blues")
    
    pcc = np.corrcoef(train_features[:, train_labels == 0])
    dp.plot_matrix(pcc, "Reds")

    
def analize_gaussianization():
    features, labels = dl.load_train_data()

    gaussianized = prep.gaussianize(features)
    dp.plot_attributes(gaussianized, labels)
    
    preprocessed = prep.apply_all_preprocess(features)
    gaussianized = prep.gaussianize(preprocessed)
    dp.plot_attributes(gaussianized, labels)



if __name__ == "__main__":
    analize_class_correlation()
