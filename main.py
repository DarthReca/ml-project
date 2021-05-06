import numpy as np

import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import dimensionality_reduction as dr


def main() -> None:
    """Only main."""
    train_data = dl.load_train_data()
    train_labels = train_data["Class"].to_numpy()
    train_data = train_data.drop("Class", axis="columns")

    test_data = dl.load_test_data()
    test_labels = test_data["Class"].to_numpy()

    #plot_data = pt.plot_attributes(train_data)
    cm = dra.confusion_matrix(test_labels[:100], train_labels[:100])

if __name__ == '__main__':
    main()
