import numpy as np

import data_loading as dl
import data_plotting as pt

"""Used to display results."""
def main() -> None:
    """Only main."""
    train_features, train_labels = dl.load_train_data()
    plot_data = pt.plot_attributes(train_features, train_labels)
    # TODO: use main


if __name__ == '__main__':
    main()
