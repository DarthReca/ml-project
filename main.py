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
    #plot_data = pt.plot_attributes(train_features, train_labels)
    model = pdf.GaussianModel()
    model.fit(train_features, train_labels)
    conf_ms = []
    for prio in np.linspace(0.1, 0.8, num=10):
        pred = model.predict(train_features, prio)
        conf_ms.append(dra.confusion_matrix(train_labels, pred))

    dra.roc_det_curves(conf_ms)

if __name__ == '__main__':
    main()
