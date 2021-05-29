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

def main():
    features, labels = dl.load_train_data()
    
    features = prep.apply_all_preprocess(features)
    
    log_regr = models.LogisticRegression(1.0)

if __name__ == '__main__':
    main()    