# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:45:45 2021

@author: DarthReca
"""

import matplotlib.pyplot as plt
import numpy as np

import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import dimensionality_reduction as dr
import models
import preprocess as prep


def calibrate_score():
    features, labels = dl.load_train_data()

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    k_scores = []
    k_labs = []
    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        log_reg = models.LogisticRegression(0, 0.1)

        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)

        tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        val_feat = gaussianizer.gaussianize(val_feat)

        log_reg.fit(tr_feat, tr_lab)
        pred, scores = log_reg.predict(val_feat, True)

        k_scores.append(scores)
        k_labs.append(val_lab)

    scores = np.hstack(k_scores)
    scores_labs = np.hstack(k_labs)
    sampled_s, sampled_slab = cv.shuffle_sample(scores, scores_labs, 2)

    (tr_scores, tr_slab), (val_scores, val_slab) = cv.train_validation_sets(
        sampled_s, sampled_slab, 1
    )

    scores_count = tr_scores.shape[1]

    dcfs = np.empty(scores_count)
    for ti in range(scores_count):
        t = tr_scores[0][ti]
        pred_s = (tr_scores >= t).astype(np.int32)
        cm = dra.confusion_matrix(tr_slab, pred_s)
        dcfs[ti] = dra.dcf(cm, 0.1, 1, 1)

    selected_thresh = tr_scores[0, dcfs.argmin()]  #

    pred_s = (val_scores >= selected_thresh).astype(np.int32)
    cm = dra.confusion_matrix(val_slab, pred_s)

    pred_s_theo = (val_scores >= -np.log(0.1 / 0.9)).astype(np.int32)
    cm_t = dra.confusion_matrix(val_slab, pred_s_theo)

    print("Threorethical threshold dcf: ", dra.dcf(cm_t, 0.1, 1, 1))
    print("Actual threshold dcf:", dra.dcf(cm, 0.1, 1, 1))
    print(selected_thresh)
    print(dra.min_norm_dcf(val_scores[0], val_slab, 0.1, 1, 1))


def bayes_err_plot():
    features, labels = dl.load_train_data()

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)

        # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        # val_feat = gaussianizer.gaussianize(val_feat)

        log_reg = models.LogisticRegression(0, 0.1)

        log_reg.fit(tr_feat, tr_lab)
        pred, scores = log_reg.predict(val_feat, True)

        dra.bayes_error_plot(scores, val_lab)


def actual_dcf():
    features, labels = dl.load_train_data()

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    dcf_5 = np.empty([k, 2])
    dcf_1 = np.empty([k, 2])
    dcf_9 = np.empty([k, 2])

    log_regr = models.LogisticRegression(0, 0.1)

    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)

        tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        val_feat = gaussianizer.gaussianize(val_feat)

        log_regr.fit(tr_feat, tr_lab)

        pred, scores = log_regr.predict(val_feat, True)
        cm = dra.confusion_matrix(val_lab, pred)

        dcf_5[i, 0] = dra.dcf(cm, 0.5, 1, 1)
        dcf_5[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)

        dcf_1[i, 0] = dra.dcf(cm, 0.1, 1, 1)
        dcf_1[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)

        dcf_9[i, 0] = dra.dcf(cm, 0.9, 1, 1)
        dcf_9[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)


def log_regr_roc():
    features, labels = dl.load_train_data()

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    t = np.linspace(-12, 7, 20)
    log_regr = models.LogisticRegression(0, 0.1)
    cms = []
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        tr_feat = preprocessor.fit_transform(tr_feat)
        ts_feat = preprocessor.transform(ts_feat)

        # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        # ts_feat = gaussianizer.gaussianize(ts_feat)

        cms.append([])
        for j in range(t.shape[0]):
            log_regr.set_threshold(t[j])
            log_regr.fit(tr_feat, tr_lab)
            pred, score = log_regr.predict(ts_feat, True)
            cms[i].append(dra.confusion_matrix(ts_lab, pred))
        dra.roc_det_curves(cms[i])


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

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    lams = np.linspace(1e-5, 1e5, 10)

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    low_dcf = np.empty([k, 10])
    norm_dcf = np.empty([k, 10])
    high_dcf = np.empty([k, 10])
    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)

        # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        # val_feat = gaussianizer.gaussianize(val_feat)

        for j in range(10):
            log_regr = models.LogisticRegression(lams[j], 0.1, True)
            log_regr.fit(tr_feat, tr_lab)
            pred, scores = log_regr.predict(val_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)


def means():
    print("0.9:", high_dcf[:, 0].mean())
    print("0.5:", norm_dcf[:, 0].mean())
    print("0.1:", low_dcf[:, 0].mean())


def misc_dcf():
    print("0.9", (dcf_9[:, 0] - dcf_9[:, 1]).mean())
    print("0.5", (dcf_5[:, 0] - dcf_5[:, 1]).mean())
    print("0.1", (dcf_1[:, 0] - dcf_1[:, 1]).mean())


if __name__ == "__main__":
    plot_risk()
