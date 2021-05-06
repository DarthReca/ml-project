import numpy as np

import data_loading as dl
import dimensionality_reduction as dr
import probability_density_function as pdf

def main() -> None:
    """Only main."""
    train_data = dl.load_train_data()
    train_labels = train_data["Class"].to_numpy()
    train_data = train_data.drop("Class", axis="columns")

    test_data = dl.load_test_data()
    test_labels = test_data["Class"].to_numpy()

    model = pdf.GaussianModel(train_data.to_numpy(), train_labels)
    model.compute_parameters()


if __name__ == '__main__':
    main()
