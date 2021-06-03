import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np
import preprocess as prep

def analize_risk():
    features, labels = dl.load_train_data()
    features = prep.apply_all_preprocess(features)
    features = prep.gaussianize(features)
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    C = np.linspace(1e-3, 1e2, 10)

    low_dcf = np.empty([k, 10])
    norm_dcf = np.empty([k, 10])
    high_dcf = np.empty([k, 10])
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        
        for j in range(10):
            svm = models.SupportVectorMachine(1, C[j], kernel_grade=2)
            svm.fit(tr_feat, tr_lab)
            pred, scores = svm.predict(ts_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, ts_lab, 0.9, 1, 1)
    pass

def main():
    features, labels = dl.load_train_data()
    
    features = prep.apply_all_preprocess(features)
        
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        ks = np.linspace(-100, 100, 10)
        conf_ms = []
        for kh in ks:
            svm = models.SupportVectorMachine(kh, 1.0, 'polynomial', 2.0, 1.0)
            svm.fit(tr_feat, tr_lab)
            pred = svm.predict(ts_feat)
            cm = dra.confusion_matrix(ts_lab, pred)
            conf_ms.append(cm)
            print(dra.matthews_corr_coeff(cm))
        dra.thresholds_error_rates(ks, conf_ms)
    
    
if __name__ == '__main__':
    analize_risk()