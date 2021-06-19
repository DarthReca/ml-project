import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import dimensionality_reduction as dr
import models
import numpy as np
import preprocess as prep

def analize_risk():
    train_features, train_labels = dl.load_train_data()
    
    s_f, s_l = cv.shuffle_sample(train_features, train_labels, 5)
    
    norm_dcf = np.empty(5)
    low_dcf = np.empty(5)
    high_dcf = np.empty(5)
    
    gaussianizer = prep.Gaussianizer()
    preprocessor = prep.Preprocessor()
    
    train_features = dr.pca(train_features, 6)
    
    model = models.GaussianModel(0.0)
    for i in range(5):
        (tr_feat, tr_lab), (val_feat, val_lab) = cv.train_validation_sets(s_f, s_l, i)
        
        tr_feat = preprocessor.fit_transform(tr_feat)
        val_feat = preprocessor.transform(val_feat)
        
        # tr_feat = gaussianizer.fit_gaussianize(tr_feat)
        # val_feat = gaussianizer.gaussianize(val_feat)

        model.set_prior(0.1)
        model.fit(tr_feat, tr_lab)
        pred, scores = model.predict(val_feat, True)
       
        norm_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.5, 1, 1)
        low_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.1, 1, 1)
        high_dcf[i] = dra.min_norm_dcf(scores, val_lab, 0.9, 1, 1)
    pass

def means():
    print("0.9:", high_dcf.mean())
    print("0.5:", norm_dcf.mean())
    print("0.1:", low_dcf.mean())

if __name__ == "__main__":
    means()
