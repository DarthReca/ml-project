# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:05:46 2021

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
    
    # Preprocess is useless
    # train_features = prep.apply_all_preprocess(train_features)

    s_f, s_l = cv.shuffle_sample(train_features, train_labels, 5)
    
    for i in range(5):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(s_f, s_l, i)
        
        model = models.GaussianModel(0.0)
        model.set_prior(0.5)
        model.fit(tr_feat, tr_lab, naive=True)
        pred, scores = model.predict(val_feat, True)
        print("0.5:", dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1))
        
        model.set_prior(0.1)
        model.fit(tr_feat, tr_lab, naive=True)
        pred, scores = model.predict(val_feat, True)
        print("0.1", dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1))
        
        model.set_prior(0.9)
        model.fit(tr_feat, tr_lab, naive=True)
        pred, scores = model.predict(val_feat, True)
        print("0.9", dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1))

def main() -> None:
    """Only main."""
    features, labels = dl.load_train_data()
    gaussian_model = models.GaussianModel(0.0)

    features = prep.apply_all_preprocess(features)

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    for i in range(k):

        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        gaussian_model.fit(tr_feat, tr_lab, naive=True)
        threshs = np.linspace(-5, 5, 10)
        conf_ms = []
        for i in threshs:
            gaussian_model.set_threshold(i)
            pred, scores = gaussian_model.predict(ts_feat, True)
            cm = dra.confusion_matrix(ts_lab, pred)
            conf_ms.append(cm)

        dra.thresholds_error_rates(threshs, conf_ms)


if __name__ == "__main__":
    analize_risk()
