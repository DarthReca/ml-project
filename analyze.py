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
from data_result_analysis import tpr_fnr, tnr_fpr
import matplotlib.pyplot as plt

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

def analize_features_separation():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
    
    dp.scatter_attributes(features, labels)


def analize_rocs():
    # Remember to load saved_data/conf_m_comparison
    fprs0 = []
    tprs0 = []
    fnrs0 = []
    for j in range(len(lin_svm_cms)):
        line = lin_svm_cms[j]
        fprs0.append([])
        tprs0.append([])
        fnrs0.append([])
        for i in range(len(line)):
            cm = line[i]
            tpr, fnr = tpr_fnr(cm)
            _, fpr = tnr_fpr(cm)
    
            fprs0[j].append(fpr)
            tprs0[j].append(tpr)
            fnrs0[j].append(fnr)
        
    fprs1 = []
    tprs1 = []
    fnrs1 = []
    for j in range(len(quad_svm_cms)):
        line = quad_svm_cms[j]
        fprs1.append([])
        tprs1.append([])
        fnrs1.append([])
        for i in range(len(line)):
            cm = line[i]
            tpr, fnr = tpr_fnr(cm)
            _, fpr = tnr_fpr(cm)
    
            fprs1[j].append(fpr)
            tprs1[j].append(tpr)
            fnrs1[j].append(fnr)

    _, (roc) = plt.subplots(constrained_layout=True)

    # ROC
    roc.set_title("ROC")
    for i in range(len(lin_svm_cms)):
        roc.plot(fprs0[i], tprs0[i], label="svm", color="blue")
        roc.plot(fprs1[i], tprs1[i], label="quad", color="red")

    roc.grid(True)
    roc.set_xlim(0, 1)
    roc.set_ylim(0, 1)

    roc.set_xlabel("False positive rate")
    roc.set_ylabel("True positive rate")
    
    roc.legend()

    plt.show()


if __name__ == "__main__":
    analize_features_separation()
