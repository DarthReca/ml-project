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
    gaussian_model = models.GaussianModel(0.0)

    features = prep.apply_all_preprocess(features)

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
            cm = dra.confusion_matrix(ts_lab, pred)

            conf_ms.append(cm)

            print(cm)

        dra.thresholds_error_rates(threshs, conf_ms)

    gaussian_model.fit(features, labels, naive=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/naive_gaussian.npy", pred)

    gaussian_model.fit(features, labels, tied_cov=True)
    pred = gaussian_model.predict(features)
    np.save("./saved_data/tied_gaussian.npy", pred)


if __name__ == "__main__":
    main()
