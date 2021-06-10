# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:49:03 2021

@author: DarthReca
"""

import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np
import preprocess as prep

def analize_risk():
    train_features, train_labels = dl.load_train_data()
    
    # train_features = prep.apply_all_preprocess(train_features)

    s_f, s_l = cv.shuffle_sample(train_features, train_labels, 5)
    
    norm_dcf = np.empty([5, 10])
    low_dcf = np.empty([5, 10])
    high_dcf = np.empty([5, 10])
    
    gaussians = np.logspace(1, 10, num=10, base=2)
    
    model = models.GaussianMixtureModel()
    for i in range(5):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(s_f, s_l, i)
        
        # Gaussianization doesn't reduce the risk
        # tr_feat = prep.gaussianize(tr_feat)
        # val_feat = prep.gaussianize(val_feat, tr_feat)
        
        for j in range(gaussians.shape[0]):
            n = gaussians[j]
            
            model.fit(tr_feat, tr_lab, n,2)
            pred, scores = model.predict(val_feat, True)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
            
            model.fit(tr_feat, tr_lab, n ,2)
            pred, scores = model.predict(val_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
            
            model.fit(tr_feat, tr_lab, n,2)
            pred, scores = model.predict(val_feat, True)
            high_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    pass

if __name__ == '__main__':
    analize_risk()