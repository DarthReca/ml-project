from itertools import combinations

import cross_validation as cv
import data_loading as dl
import data_plotting as pt
import data_result_analysis as dra
import models
import numpy as np


def main() -> None:
    """Only main."""
    features, labels = dl.load_train_data()
    gaussian_model = models.GaussianModel()

    k = 5
    sampled_f, sampled_l = cv.shuffle_sample(features, labels, k)

    for i in range(k):

        tr_feat = sampled_f[i]
        tr_lab = sampled_l[i]

        ts_feat = np.hstack([sampled_f[x] for x in range(k) if not i == x])
        ts_lab = np.hstack([sampled_l[x] for x in range(k) if not i == x])

        gaussian_model.fit(tr_feat, tr_lab)

        gaussian_model.set_threshold(1.2)
        pred, scores = gaussian_model.predict(ts_feat, True)

        print((pred == ts_lab).sum() / pred.shape[0])

        # np.save("./saved_data/gaussian.npy", pred)

    gaussian_model.fit(features, labels, naive=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/naive_gaussian.npy", pred)

    gaussian_model.fit(features, labels, tied_cov=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/tied_gaussian.npy", pred)


if __name__ == "__main__":
    main()
