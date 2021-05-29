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
    
    models.SupportVectorMachine(0.0, 1.0, 'polynomial', 2.0, 0.0)
    
    
if __name__ == '__main__':
    main()