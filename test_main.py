# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:19:54 2021

@author: DarthReca
"""

import data_loading as dl
import models
import preprocess as pp
import data_result_analysis as dra
import data_plotting as dp

def main():
    features, labels = dl.load_test_data()
    
    features = pp.apply_all_preprocess(features)    
    
    train_f, train_l = dl.load_train_data()
    
    train_f = pp.apply_all_preprocess(train_f)
    
    #train_f = pp.gaussianize(train_f) 
    #features = pp.gaussianize(features, train_f)
    
    #gm = models.LogisticRegression(1e-5, 0.1)
    gm = models.SupportVectorMachine(k=1.0, C=1e-3, prior_true=0.1,
                                      kernel_type="polynomial",
                                      kernel_grade=2.0, pol_kernel_c=1.0)
       
    #gm.set_threshold(-0.25)
    gm.set_threshold(-0.83)
    
    gm.fit(train_f, train_l)
    
    pred = gm.predict(features)
    
    cm = dra.confusion_matrix(labels, pred)
    
    print(dra.matthews_corr_coeff(cm))


if __name__ == "__main__":
    main()
