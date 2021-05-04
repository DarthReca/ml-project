import numpy as np
import data_loading as dl
import data_plotting as pt

"""Used to display results."""


def main() -> None:
    """Only main."""
    test_data = dl.load_test_data()
    labels_name = dl.labels_name
    plot_data = pt.plot_test_data()
    labels_name = dl.labels_name
    # TODO: use main


if __name__ == '__main__':
    main()
