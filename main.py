import numpy as np
import sklearn.gaussian_process as skga

import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import probability_density_function as pdf

"""Used to display results."""


def main() -> None:
    """Only main."""
    train_features, train_labels = dl.load_train_data()
    test_features, test_labels = dl.load_test_data()
    #plot_data = pt.plot_attributes(train_features, train_labels)
    model = pdf.GaussianModel()
    
    model.fit(train_features, train_labels)
    pred = model.predict(test_features, 0.2)
    print(dra.confusion_matrix(test_labels, pred))
    
    model.fit(train_features, train_labels, naive=True)
    pred = model.predict(test_features, 0.2)
    print(dra.confusion_matrix(test_labels, pred))
    
    model.fit(train_features, train_labels, tied_cov=True)
    pred = model.predict(test_features, 0.2)
    print(dra.confusion_matrix(test_labels, pred))


if __name__ == '__main__':
    main()
