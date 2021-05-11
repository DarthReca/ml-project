import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np


def main() -> None:
    """Only main."""
    train_features, train_labels = dl.load_train_data()
    test_features, test_labels = dl.load_test_data()

    gaussian_model = models.GaussianModel()

    gaussian_model.fit(train_features, train_labels)
    priors = np.linspace(0.1, 0.5, num=20)
    for p in priors:
        pred = gaussian_model.predict(test_features, p)
        np.save("./saved_data/gaussian-{}.npy".format(p), pred)

    gaussian_model.fit(train_features, train_labels, naive=True)
    pred = gaussian_model.predict(test_features, 0.2)
    np.save("./saved_data/naive_gaussian.npy", pred)

    gaussian_model.fit(train_features, train_labels, tied_cov=True)
    pred = gaussian_model.predict(test_features, 0.2)
    np.save("./saved_data/tied_gaussian.npy", pred)


if __name__ == "__main__":
    main()
