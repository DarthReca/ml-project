import numpy as np
import data_loading as dl
<<<<<<< Updated upstream
=======
import data_plotting as pt
>>>>>>> Stashed changes

"""Used to display results."""
def main() -> None:
    """Only main."""
    test_data = dl.load_test_data()
<<<<<<< Updated upstream
=======
    labels_name = dl.labels_name
    plot_data = pt.plot_test_data()
>>>>>>> Stashed changes
    # TODO: use main

if __name__ == '__main__':
    main()