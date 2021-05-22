from itertools import combinations

import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np
import preprocess as prep


def main() -> None:
    """Only main."""
    features, labels = dl.load_train_data()
    gaussian_model = models.GaussianModel()

    features = prep.center_features(features)
    features = prep.standardize_variance(features)
    features = prep.whiten_covariance(features)
    features = prep.normalize_lenght(features)

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    for i in range(k):

        (tr_feat, tr_lab), (ts_feat, ts_lab) = cv.train_validation_sets(
            sampled_f, sampled_l, i
        )

        gaussian_model.fit(tr_feat, tr_lab)
        threshs = np.linspace(-5, 5, 10)
        conf_ms = []
        for i in threshs:
            gaussian_model.set_threshold(i)
            pred, scores = gaussian_model.predict(ts_feat, True)

            conf_ms.append(dra.confusion_matrix(ts_lab, pred))

            print((pred == ts_lab).sum() / pred.shape[0])

        dra.thresholds_error_rates(threshs, conf_ms)

        # np.save("./saved_data/gaussian.npy", pred)

    gaussian_model.fit(features, labels, naive=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/naive_gaussian.npy", pred)

    gaussian_model.fit(features, labels, tied_cov=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/tied_gaussian.npy", pred)


if __name__ == "__main__":
    main()
