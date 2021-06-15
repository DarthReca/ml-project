# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:19:54 2021

@author: DarthReca
"""

import data_loading as dl
import models
import preprocess as prep
import data_result_analysis as dra
import data_plotting as dp
import cross_validation as cv
import numpy as np

def gaussian_min_dcf():
    features, labels = dl.load_test_data()
    features = prep.apply_all_preprocess(features)
    
    s_f, s_l = cv.shuffle_sample(features, labels, 5)
    
    norm_dcf = np.empty(5)
    low_dcf = np.empty(5)
    high_dcf = np.empty(5)
    
    
    model = models.GaussianModel(0.0)
    for i in range(5):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(s_f, s_l, i)
        
        # Gaussianization doesn't reduce the risk
        tr_feat = prep.gaussianize(tr_feat)
        val_feat = prep.gaussianize(val_feat, tr_feat)
        
        model.set_prior(0.1)
        model.fit(tr_feat, tr_lab, tied_cov=True)
        pred, scores = model.predict(val_feat, True)
       
        norm_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
        low_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
        high_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    
    print("pi_t = 0.1")
    print("0.1, 0.5, 0.9", 
          low_dcf.mean(axis=0), 
          norm_dcf.mean(axis=0),
          high_dcf.mean(axis=0))

def log_reg_min_dcf():
    select_l = 1e-5
    k = 5

    features, labels = dl.load_test_data()
    features = prep.apply_all_preprocess(features)
    
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    
    low_lr = models.LogisticRegression(select_l, 0.1)
    
    min_dcf_1 = np.empty([k , 3])
    
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        tr_feat = prep.gaussianize(tr_feat)
        ts_feat = prep.gaussianize(ts_feat, tr_feat)
        
        low_lr.fit(tr_feat, tr_lab)
        _, l_scores = low_lr.predict(ts_feat, True)
        
        # print("pi_t = 0.1")
        min_dcf_1[i, 0] = dra.min_norm_dcf(l_scores, ts_lab, 0.1, 1, 1)
        min_dcf_1[i, 1] = dra.min_norm_dcf(l_scores, ts_lab, 0.5, 1, 1)
        min_dcf_1[i, 2] = dra.min_norm_dcf(l_scores, ts_lab, 0.9, 1, 1)
        
    print("pi_t = 0.1")
    print("0.1, 0.5, 0.9", min_dcf_1.mean(axis=0))

def svm_min_dcf():
    features, labels = dl.load_test_data()

    features = prep.apply_all_preprocess(features)
    grade = 10.0

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    min_dcf_1 = np.empty(k)
    min_dcf_5 = np.empty(k)
    min_dcf_9 = np.empty(k)
    svm = models.SupportVectorMachine(
        kernel_type="radial basis function",
        k=1, C=1e-3, 
        prior_true=0.1, 
        kernel_grade=grade, 
        pol_kernel_c=1.0
    )
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        # Not so useful
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)

        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(ts_feat, True)

        min_dcf_1[i] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
        min_dcf_5[i] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
        min_dcf_9[i] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)

    print("pi_t = 0.1", min_dcf_1.mean(axis=0))
    print("pi_t = 0.5", min_dcf_5.mean(axis=0))
    print("pi_t = 0.9", min_dcf_9.mean(axis=0))

def gmm_min_dcf():
    train_features, train_labels = dl.load_test_data()
    
    # train_features = prep.apply_all_preprocess(train_features)

    s_f, s_l = cv.shuffle_sample(train_features, train_labels, 5)
    
    norm_dcf = np.empty(5)
    low_dcf = np.empty(5)
    high_dcf = np.empty(5)
        
    model = models.GaussianMixtureModel(precision=1e-2)
    for i in range(5):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(s_f, s_l, i)
        
        # tr_feat = prep.gaussianize(tr_feat)
        # val_feat = prep.gaussianize(val_feat, tr_feat)
        
        n = 4
            
        model.set_prior(0.1)
        model.fit(tr_feat, tr_lab, n,2)
        pred, scores = model.predict(val_feat, True)
        
        norm_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
        low_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
        high_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    
    print("0.1 = ", low_dcf.mean(axis=0)) 
    print("0.5 = ", norm_dcf.mean(axis=0))
    print("0.9 =", high_dcf.mean(axis=0))

def log_regr_lambda_estimation():
    features, labels = dl.load_test_data()
    features = prep.apply_all_preprocess(features)
        
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    lams = np.linspace(1e-5, 1e5, 10)

    low_dcf = np.empty([k, 10])
    norm_dcf = np.empty([k, 10])
    high_dcf = np.empty([k, 10])
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        # Gaussianization doesn't reduce the risk
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)
        
        for j in range(10):
            log_regr = models.LogisticRegression(lams[j], 0.1)
            log_regr.fit(tr_feat, tr_lab)
            pred, scores = log_regr.predict(ts_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    pass

def log_regr_roc():
    features, labels = dl.load_test_data()
    features = prep.apply_all_preprocess(features)
        
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    t = np.linspace(-15, 15, 10)
    log_regr = models.LogisticRegression(0, 0.1)
    cms = []
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)
        cms.append([])
        for j in range(t.shape[0]):
            log_regr.set_threshold(t[j])
            log_regr.fit(tr_feat, tr_lab)
            pred, score = log_regr.predict(ts_feat, True)
            cms[i].append(dra.confusion_matrix(ts_lab, pred))
        dra.roc_det_curves(cms[i])
    pass

def calibrate_score():
    features, labels = dl.load_test_data()
    features = prep.apply_all_preprocess(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    k_scores = []
    k_labs = []
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )  
        
        log_reg = models.LogisticRegression(0, 0.1)
        
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
    
    pred_s_theo = (val_scores >= -np.log(0.1/0.9)).astype(np.int32)
    cm_t = dra.confusion_matrix(val_slab, pred_s_theo)
    
    print("Threorethical threshold dcf: ", dra.dcf(cm_t, 0.1, 1, 1))
    print("Actual threshold dcf:", dra.dcf(cm, 0.1, 1, 1))
    print(selected_thresh)
    print(dra.min_norm_dcf(val_scores[0], val_slab, 0.1, 1, 1))
    pass     

def main():
    features, labels = dl.load_test_data()
    
    features = prep.apply_all_preprocess(features)    
    
    train_f, train_l = dl.load_train_data()
    
    train_f = prep.apply_all_preprocess(train_f)

    
    gm = models.LogisticRegression(0, 0.1)
       
    gm.set_threshold(-0.141)

    gm.fit(train_f, train_l)
    
    pred = gm.predict(features)
    
    cm = dra.confusion_matrix(labels, pred)
    
    print(dra.matthews_corr_coeff(cm))


if __name__ == "__main__":
    main()
