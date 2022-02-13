# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:19:54 2021

@author: DarthReca
"""

import numpy as np

import cross_validation as cv
import data_loading as dl
import data_plotting as dp
import data_result_analysis as dra
import models
import preprocess as prep


def gaussian_min_dcf():
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
    
    model = models.GaussianModel(0.0)
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    # ts_feat = gaussianizer.gaussianize(ts_feat)

    model.set_prior(0.1)
    model.fit(tr_feat, tr_lab, tied_cov=True)
    pred, scores = model.predict(ts_feat, True)
   
    norm_dcf = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
    low_dcf = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
    high_dcf = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    
    print("0.1", low_dcf)
    print("0.5", norm_dcf)
    print("0.9", high_dcf)

def log_reg_min_dcf():
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
    
    model = models.LogisticRegression(0, 0.1)
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    # ts_feat = gaussianizer.gaussianize(ts_feat)

    model.fit(tr_feat, tr_lab)
    pred, scores = model.predict(ts_feat, True)
   
    norm_dcf = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
    low_dcf = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
    high_dcf = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    
    print("0.1", low_dcf)
    print("0.5", norm_dcf)
    print("0.9", high_dcf)

def svm_min_dcf():
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
    
    model = models.SupportVectorMachine(
        k=1.0, C=1e-3, prior_true=0.1, 
        kernel_type="radial basis function", 
        kernel_grade=0.1,
        pol_kernel_c=1.0)
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    ts_feat = gaussianizer.gaussianize(ts_feat)

    model.fit(tr_feat, tr_lab)
    pred, scores = model.predict(ts_feat, True)
   
    norm_dcf = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
    low_dcf = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
    high_dcf = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    
    print("0.1", low_dcf)
    print("0.5", norm_dcf)
    print("0.9", high_dcf)

def gmm_min_dcf():
    n = 64
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
    
    model = models.GaussianMixtureModel(precision=1e-1)
        
    # tr_feat = preprocessor.fit_transform(tr_feat)
    # ts_feat = preprocessor.transform(ts_feat)
    
    # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    # ts_feat = gaussianizer.gaussianize(ts_feat)
    
    model.fit(tr_feat, tr_lab, n, 2)
    pred, scores = model.predict(ts_feat, True)
       
    norm_dcf = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
    low_dcf = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
    high_dcf = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    
    print("0.1", low_dcf)
    print("0.5", norm_dcf)
    print("0.9", high_dcf)

def log_regr_lambda_estimation():
    
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    lams = np.linspace(1e-5, 1e5, 10)  
    low_dcf = np.empty([1, 10])
    norm_dcf = np.empty([1, 10])
    high_dcf = np.empty([1, 10])
    
    for j in range(10):
        log_regr = models.LogisticRegression(lams[j], 0.1)
        log_regr.fit(tr_feat, tr_lab)
        pred, scores = log_regr.predict(ts_feat, True)
        low_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
        norm_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
        high_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)

def svm_C_estimation():
    
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    lams = np.linspace(1e-3, 1e2, 10)  
    low_dcf = np.empty([1, 10])
    norm_dcf = np.empty([1, 10])
    high_dcf = np.empty([1, 10])
    
    for j in range(10):
        log_regr = models.SupportVectorMachine(
        k=1.0, C=lams[j], prior_true=0.1, 
        kernel_type="polynomial", 
        kernel_grade=2.0,
        pol_kernel_c=1.0)
        
        log_regr.fit(tr_feat, tr_lab)
        pred, scores = log_regr.predict(ts_feat, True)
        low_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
        norm_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
        high_dcf[0, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)

def bayes_errors_plot():
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()        
    
    model = models.SupportVectorMachine(k=1.0, 
        C=1e-3, prior_true=0.1, 
        kernel_type="polynomial", 
        kernel_grade=1.0,
        pol_kernel_c=1.0)
    
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    # ts_feat = gaussianizer.gaussianize(ts_feat)
    
        
    model.fit(tr_feat, tr_lab)
    pred, scores = model.predict(ts_feat, True)
         
    dra.bayes_error_plot(scores, ts_lab)


def calibrate_score():
    ts_feat, ts_lab = dl.load_test_data()
    tr_feat, tr_lab = dl.load_train_data()
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()        
    
    
    model = models.LogisticRegression(0, 0.1)
        
    tr_feat = preprocessor.fit_transform(tr_feat)
    ts_feat = preprocessor.transform(ts_feat)
    
    tr_feat = gaussianizer.fit_gaussianize(tr_feat)
    ts_feat = gaussianizer.gaussianize(ts_feat)

    model.fit(tr_feat, tr_lab)
    pred, scores = model.predict(ts_feat, True)
    scores_labs = ts_lab

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
    
    selected_thresh = tr_scores[0, dcfs.argmin()]#

    pred_s = (val_scores >= selected_thresh).astype(np.int32)
    cm = dra.confusion_matrix(val_slab, pred_s)
    
    pred_s_theo = (val_scores >= -np.log(0.1/0.9)).astype(np.int32)
    cm_t = dra.confusion_matrix(val_slab, pred_s_theo)
    
    print("Threorethical threshold dcf: ", dra.dcf(cm_t, 0.1, 1, 1))
    print("Actual threshold dcf:", dra.dcf(cm, 0.1, 1, 1))
    print(selected_thresh)
    print(dra.min_norm_dcf(val_scores[0], val_slab, 0.1, 1, 1))

def main():
    train_f, train_l = dl.load_train_data()
    test_f, test_l = dl.load_test_data()
    
    preprocessor = prep.Preprocessor()
    gaussianizer = prep.Gaussianizer()
    
    train_f = preprocessor.fit_transform(train_f)
    test_f = preprocessor.transform(test_f)
    
    train_f = gaussianizer.fit_gaussianize(train_f)
    test_f = gaussianizer.gaussianize(test_f)
    
    lr = models.LogisticRegression(0, 0.1)
    lr.fit(train_f, train_l)
    pred = lr.predict(test_f)
    
    cm = dra.confusion_matrix(test_l, pred)
    
    print(dra.matthews_corr_coeff(cm))

if __name__ == "__main__":
    main()
