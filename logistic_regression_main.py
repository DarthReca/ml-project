# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:45:45 2021

@author: gino9
"""

import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np
import preprocess as prep

def main():
    features, labels = dl.load_train_data()
    
    features = prep.apply_all_preprocess(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        lams = np.linspace(-100, 100, 10)
        conf_ms = []
        for l in lams:
            log_regr = models.LogisticRegression(l)
            log_regr.fit(tr_feat, tr_lab)
            pred = log_regr.predict(ts_feat)
            cm = dra.confusion_matrix(ts_lab, pred)
            conf_ms.append(cm)
            print(dra.matthews_corr_coeff(cm))
        dra.thresholds_error_rates(lams, conf_ms)

if __name__ == '__main__':
    main()    