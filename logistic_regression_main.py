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
import matplotlib.pyplot as plt
        
def plot_risk():
    fig, ax = plt.subplots()
    
    ax.plot(lams, norm_dcf.mean(axis=0), label="prior 0.5")
    ax.plot(lams, low_dcf.mean(axis=0), label="prior 0.1")
    ax.plot(lams, high_dcf.mean(axis=0), label="prior 0.9")

    ax.set_xscale("log")
    ax.legend()    
    plt.show()

def analize_risk():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
        
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    lams = np.linspace(1e-5, 1e5, 20)

    low_dcf = np.empty([k, 20])
    norm_dcf = np.empty([k, 20])
    high_dcf = np.empty([k, 20])
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        # Gaussianization doesn't reduce the risk
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)
        
        for j in range(20):
            log_regr = models.LogisticRegression(lams[j], 0.5)
            log_regr.fit(tr_feat, tr_lab)
            pred, scores = log_regr.predict(ts_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    pass

def print_min_risk():
    select_l = 1e-5
    k = 5

    features, labels = dl.load_train_data()
    # Some benefits only for pi = 0.9
    features = prep.apply_all_preprocess(features)
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    
    low_lr = models.LogisticRegression(select_l, 0.1)
    norm_lr = models.LogisticRegression(select_l, 0.5)
    high_lr = models.LogisticRegression(select_l, 0.9)
    
    min_dcf_1 = np.empty([k , 3])
    min_dcf_5 = np.empty([k , 3])
    min_dcf_9 = np.empty([k , 3])
    
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        low_lr.fit(tr_feat, tr_lab)
        norm_lr.fit(tr_feat, tr_lab)
        high_lr.fit(tr_feat, tr_lab)
        
        _, l_scores = low_lr.predict(ts_feat, True)
        _, n_scores = norm_lr.predict(ts_feat, True)
        _, h_scores = high_lr.predict(ts_feat, True)
        
        # print("pi_t = 0.1")
        min_dcf_1[i, 0] = dra.min_norm_dcf(l_scores, ts_lab, 0.1, 1, 1)
        min_dcf_1[i, 1] = dra.min_norm_dcf(l_scores, ts_lab, 0.5, 1, 1)
        min_dcf_1[i, 2] = dra.min_norm_dcf(l_scores, ts_lab, 0.9, 1, 1)
        
        # print("pi_t = 0.5")
        min_dcf_5[i, 0] = dra.min_norm_dcf(n_scores, ts_lab, 0.1, 1, 1)
        min_dcf_5[i, 1] = dra.min_norm_dcf(n_scores, ts_lab, 0.5, 1, 1)
        min_dcf_5[i, 2] = dra.min_norm_dcf(n_scores, ts_lab, 0.9, 1, 1)

        # print("pi_t = 0.9")
        min_dcf_9[i, 0] = dra.min_norm_dcf(h_scores, ts_lab, 0.1, 1, 1)
        min_dcf_9[i, 1] = dra.min_norm_dcf(h_scores, ts_lab, 0.5, 1, 1)
        min_dcf_9[i, 2] = dra.min_norm_dcf(h_scores, ts_lab, 0.9, 1, 1)
        
    print("pi_t = 0.1")
    print("0.1, 0.5, 0.9", min_dcf_1.mean(axis=0))
    print("pi_t = 0.5")
    print("0.1, 0.5, 0.9", min_dcf_5.mean(axis=0))
    print("pi_t = 0.9")
    print("0.1, 0.5, 0.9", min_dcf_9.mean(axis=0))


if __name__ == '__main__':
    plot_risk()    