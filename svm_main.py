import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import matplotlib.pyplot as plt
import models
import numpy as np
import preprocess as prep  
    
def calibrate_score():
    features, labels = dl.load_train_data()
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    
    preprocessor = prep.Preprocessor()
    
    k_scores = []
    k_labs = []
    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )  
        
        svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.1,
                                          kernel_type="polynomial",
                                          kernel_grade=1.0, pol_kernel_c=1.0)
        
        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)
        
        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(val_feat, True)
        
        k_scores.append(scores)
        k_labs.append(val_lab)


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
    
    print("Threorethical threshold", dra.dcf(cm_t, 0.1, 1, 1))
    print("Actual threshold dcf:", dra.dcf(cm, 0.1, 1, 1))
    print(selected_thresh)
    print(dra.min_norm_dcf(val_scores[0], val_slab, 0.1, 1, 1))
    pass 

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
    
        svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.1,
                                          kernel_type="polynomial",
                                          kernel_grade=1.0, pol_kernel_c=1.0)
        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(val_feat, True)
         
        dra.bayes_error_plot(scores, val_lab)
    pass

def actual_dcf():
    features, labels = dl.load_train_data()
    
    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    dcf_5 = np.empty([k, 2])    
    dcf_1 = np.empty([k, 2])    
    dcf_9 = np.empty([k, 2])    

    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )   
        
        svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.1,
                                          kernel_type="polynomial",
                                          kernel_grade=2.0, pol_kernel_c=1.0)
        
        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)
        
        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(val_feat, True)
        cm = dra.confusion_matrix(val_lab, pred)
        
        dcf_5[i, 0] = dra.dcf(cm, 0.5, 1, 1) 
        dcf_5[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)

        dcf_1[i, 0] = dra.dcf(cm, 0.1, 1, 1) 
        dcf_1[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
        
        dcf_9[i, 0] = dra.dcf(cm, 0.9, 1, 1)
        dcf_9[i, 1] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    pass

def svm_roc():
    features, labels = dl.load_train_data()
    preprocessor = prep.Preprocessor()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)
    
    cms = []
    t = np.linspace(-1.5, -0.5, 20)
    
    svm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.1,
                                      kernel_type="polynomial",
                                      kernel_grade=1.0, pol_kernel_c=1.0)
    for i in range(k):
        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )
        tr_feat = preprocessor.fit_transform(tr_feat)
        ts_feat = preprocessor.transform(ts_feat)
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
    features, labels = dl.load_test_data()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    C = np.linspace(1e-3, 1e2, 10)
    # Tested Pol: 1.0, 2.0
    grade = 100.0
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    low_dcf = np.empty([k, 10])
    norm_dcf = np.empty([k, 10])
    high_dcf = np.empty([k, 10])

    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        for j in range(10):
            svm = models.SupportVectorMachine(
                k=1.0, C=C[j], prior_true=0.1, 
                kernel_grade=grade, pol_kernel_c=1.0,
                kernel_type="radial basis function"
            )
            
            tr_feat = preprocessor.fit_transform(tr_feat)
            val_feat = preprocessor.transform(val_feat)
            
            # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
            # val_feat = gaussianizer.gaussianize(val_feat)
            
            svm.fit(tr_feat, tr_lab)
            pred, scores = svm.predict(val_feat, True)
            low_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
            norm_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
            high_dcf[i, j] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    pass

def analize_single():
    features, labels = dl.load_test_data()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    # Tested Pol: 1.0, 2.0
    grade = 0.01
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()

    low_dcf = np.empty([k])
    norm_dcf = np.empty([k])
    high_dcf = np.empty([k])

    for i in range(k):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        
        svm = models.SupportVectorMachine(
            k=1.0, C=1e-3, prior_true=0.1, 
            kernel_grade=grade, pol_kernel_c=1.0,
            kernel_type="radial basis function"
        )
        
        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)
        
        # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        # val_feat = gaussianizer.gaussianize(val_feat)
        
        svm.fit(tr_feat, tr_lab)
        pred, scores = svm.predict(val_feat, True)
        low_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
        norm_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
        high_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    pass

def means():
    print("0.9:", high_dcf[:, 0].mean())
    print("0.5:", norm_dcf[:, 0].mean())
    print("0.1:", low_dcf[:,0].mean())
    

if __name__ == "__main__":
    means()
