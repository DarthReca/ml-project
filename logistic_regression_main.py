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
 
def calibrate_score():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    k_scores = []
    k_labs = []
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )  
        
        log_reg = models.LogisticRegression(1e-5, 0.1)
        
        log_reg.fit(tr_feat, tr_lab)
        pred, scores = log_reg.predict(ts_feat, True)
        
        k_scores.append(scores)
        k_labs.append(ts_lab)


    scores = np.hstack(k_scores)
    scores_labs = np.hstack(k_labs)
    sampled_s, sampled_slab = cv.shuffle_sample(scores, scores_labs, 2)
    
    (tr_scores, tr_slab), (val_scores, val_slab) = cv.train_validation_sets(
        sampled_s, sampled_slab, 1)
    
    scores_count = tr_scores.shape[1]
    
    dcfs = np.empty(scores_count)
    for ti in range(scores_count):
       t = tr_scores[0][ti]
       pred_s = (tr_scores >= t).astype(np.int32)
       cm = dra.confusion_matrix(tr_slab, pred_s)
       dcfs[ti] = dra.dcf(cm, 0.1, 1, 1)
    
    selected_thresh = tr_scores[0, dcfs.argmin()]

    pred_s = (val_scores >= selected_thresh).astype(np.int32)
    cm = dra.confusion_matrix(val_slab, pred_s)
    
    print(dra.dcf(cm, 0.1, 1, 1))
    print(selected_thresh)
    print(dra.min_norm_dcf(scores, scores_labs, 0.1, 1, 1))
    pass        

def log_regr_bayes_err_plot():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )   
    
        log_reg = models.LogisticRegression(1e-5, 0.1, True)
        
        log_reg.fit(tr_feat, tr_lab)
        pred, scores = log_reg.predict(ts_feat, True)
         
        dra.bayes_error_plot(scores, ts_lab)
    pass
   

def log_regr_dcf():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    dcf_5 = np.empty([k, 2])    
    dcf_1 = np.empty([k, 2])    
    dcf_9 = np.empty([k, 2])    
    
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )  
        
        log_regr = models.LogisticRegression(1e-5, 0.5)
        log_regr.set_threshold(0.0)
        log_regr.fit(tr_feat, tr_lab)
        pred, scores = log_regr.predict(ts_feat, True)
        cm = dra.confusion_matrix(ts_lab, pred)
        dcf_5[i, 0] = dra.dcf(cm, 0.5, 1, 1) 
        dcf_5[i, 1] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)

        log_regr = models.LogisticRegression(1e-5, 0.1)
        log_regr.set_threshold(-np.log(0.1/0.9))
        log_regr.fit(tr_feat, tr_lab)
        pred, scores = log_regr.predict(ts_feat, True)
        cm = dra.confusion_matrix(ts_lab, pred)
        dcf_1[i, 0] = dra.dcf(cm, 0.1, 1, 1) 
        dcf_1[i, 1] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)

        log_regr = models.LogisticRegression(1e-5, 0.9)
        log_regr.set_threshold(-np.log(0.9/0.1))
        log_regr.fit(tr_feat, tr_lab)
        pred, scores = log_regr.predict(ts_feat, True)
        cm = dra.confusion_matrix(ts_lab, pred)
        dcf_9[i, 0] = dra.dcf(cm, 0.9, 1, 1) 
        dcf_9[i, 1] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    pass

def log_regr_roc():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
        
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    t = np.linspace(-15, 15, 20)
    log_regr = models.LogisticRegression(1e-5, 0.5)
    cms = []
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        cms.append([])
        for j in range(t.shape[0]):
            log_regr.set_threshold(t[j])
            log_regr.fit(tr_feat, tr_lab)
            pred, score = log_regr.predict(ts_feat, True)
            cms[i].append(dra.confusion_matrix(ts_lab, pred))
        dra.roc_det_curves(cms[i])
    pass
        

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
            log_regr = models.LogisticRegression(lams[j], 0.1)
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
    #norm_lr = models.LogisticRegression(select_l, 0.5)
    #high_lr = models.LogisticRegression(select_l, 0.9)
    
    min_dcf_1 = np.empty([k , 3])
    min_dcf_5 = np.empty([k , 3])
    min_dcf_9 = np.empty([k , 3])
    
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)
        
        low_lr.fit(tr_feat, tr_lab)
        # norm_lr.fit(tr_feat, tr_lab)
        # high_lr.fit(tr_feat, tr_lab)
        
        _, l_scores = low_lr.predict(ts_feat, True)
        # _, n_scores = norm_lr.predict(ts_feat, True)
        # _, h_scores = high_lr.predict(ts_feat, True)
        
        # print("pi_t = 0.1")
        min_dcf_1[i, 0] = dra.min_norm_dcf(l_scores, ts_lab, 0.1, 1, 1)
        min_dcf_1[i, 1] = dra.min_norm_dcf(l_scores, ts_lab, 0.5, 1, 1)
        min_dcf_1[i, 2] = dra.min_norm_dcf(l_scores, ts_lab, 0.9, 1, 1)
        
        # print("pi_t = 0.5")
        """
        min_dcf_5[i, 0] = dra.min_norm_dcf(n_scores, ts_lab, 0.1, 1, 1)
        min_dcf_5[i, 1] = dra.min_norm_dcf(n_scores, ts_lab, 0.5, 1, 1)
        min_dcf_5[i, 2] = dra.min_norm_dcf(n_scores, ts_lab, 0.9, 1, 1)

        # print("pi_t = 0.9")
        min_dcf_9[i, 0] = dra.min_norm_dcf(h_scores, ts_lab, 0.1, 1, 1)
        min_dcf_9[i, 1] = dra.min_norm_dcf(h_scores, ts_lab, 0.5, 1, 1)
        min_dcf_9[i, 2] = dra.min_norm_dcf(h_scores, ts_lab, 0.9, 1, 1)
        """
    print("pi_t = 0.1")
    print("0.1, 0.5, 0.9", min_dcf_1.mean(axis=0))
    print("pi_t = 0.5")
    print("0.1, 0.5, 0.9", min_dcf_5.mean(axis=0))
    print("pi_t = 0.9")
    print("0.1, 0.5, 0.9", min_dcf_9.mean(axis=0))


if __name__ == '__main__':
    print_min_risk()    
