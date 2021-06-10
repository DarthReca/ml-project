import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import matplotlib.pyplot as plt
import models
import numpy as np
import preprocess as prep

def linear_svm_roc():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    
    cms = []
    t = np.linspace(-2, 2, 20)
    
    svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.5,
                                      kernel_grade=1.0, pol_kernel_c=1.0)
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        cms.append([])
        for j in range(t.shape[0]):
            svm.set_threshold(t[j])
            svm.fit(tr_feat, tr_lab)
            pred, scores = svm.predict(ts_feat, True)
            cms[i].append(dra.confusion_matrix(ts_lab, pred))
        dra.roc_det_curves(cms[i])
    pass

def quadratic_svm_roc():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    
    cms = []
    t = np.linspace(-2, 2, 20)
    
    svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.5,
                                      kernel_grade=2.0, pol_kernel_c=1.0)
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        cms.append([])
        for j in range(t.shape[0]):
            svm.set_threshold(t[j])
            svm.fit(tr_feat, tr_lab)
            pred, scores = svm.predict(ts_feat, True)
            cms[i].append(dra.confusion_matrix(ts_lab, pred))
        dra.roc_det_curves(cms[i])
    pass

def plot_risk():
    fig, ax = plt.subplots()

    ax.plot(C, norm_dcf.mean(axis=0), label="prior 0.5")
    ax.plot(C, low_dcf.mean(axis=0), label="prior 0.1")
    ax.plot(C, high_dcf.mean(axis=0), label="prior 0.9")

    ax.set_xscale("log")
    ax.legend()
    plt.show()


def analize_risk_C():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    C = np.linspace(1e-3, 1e2, 10)
    # Tested Pol: 1.0, 2.0
    grade = 2.
    # Tested RBF: 0.1, 1.0, 10
    gamma = 0.1

    low_dcf = np.empty([k, 10])
    norm_dcf = np.empty([k, 10])
    high_dcf = np.empty([k, 10])

    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        # Not so useful
        # tr_feat = prep.gaussianize(tr_feat)
        # ts_feat = prep.gaussianize(ts_feat, tr_feat)

        for j in range(10):
            svm = models.SupportVectorMachine(
                k=1.0, C=C[j], prior_true=0.5, 
                kernel_grade=gamma, pol_kernel_c=1.0,
                kernel_type="radial basis function"
            )
            svm.fit(tr_feat, tr_lab)
            pred, scores = svm.predict(ts_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    pass


def print_min_risk():
    features, labels = dl.load_train_data()

    features = prep.apply_all_preprocess(features)
    grade = 10

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    min_dcf_1 = np.empty(k)
    min_dcf_5 = np.empty(k)
    min_dcf_9 = np.empty(k)
    svm = models.SupportVectorMachine(
        k=1, C=1e-3, prior_true=0.5, kernel_grade=grade, pol_kernel_c=1.0
    )
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        # Not so useful
        tr_feat = prep.gaussianize(tr_feat)
        ts_feat = prep.gaussianize(ts_feat, tr_feat)

        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(ts_feat, True)

        min_dcf_1[i] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
        min_dcf_5[i] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
        min_dcf_9[i] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)

    print("pi_t = 0.1", min_dcf_1.mean(axis=0))
    print("pi_t = 0.5", min_dcf_5.mean(axis=0))
    print("pi_t = 0.9", min_dcf_9.mean(axis=0))


if __name__ == "__main__":
    quadratic_svm_roc()
