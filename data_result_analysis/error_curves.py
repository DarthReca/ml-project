# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:59:17 2021

@author: DarthReca
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .confusion_matrix import confusion_matrix, tnr_fpr, tpr_fnr


def thresholds_error_rates(
    thresholds: np.ndarray, confusion_matrixes: List[np.ndarray]
) -> None:
    """
    Plot for each threshold the associated error rate.

    Parameters
    ----------
    thresholds : np.ndarray

    confusion_matrixes : List[np.ndarray]
        Confusion matrix i refers to threshold i

    """
    fprs = []
    fnrs = []

    for cm in confusion_matrixes:
        _, fpr = tnr_fpr(cm)
        _, fnr = tpr_fnr(cm)
        fprs.append(fpr)
        fnrs.append(fnr)

    plt.plot(thresholds, fprs, label="False Positive Rate")
    plt.plot(thresholds, fnrs, label="False Negative Rate")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()


def roc_det_curves(confusion_matrixes: List[np.ndarray]) -> None:
    """
    Plot ROC and DET curve.

    Parameters
    ----------
    confusion_matrix : List[np.ndarray]
        list of size t of ndarrays of size(2,2).

    """
    fprs = []
    tprs = []
    fnrs = []
    for cm in confusion_matrixes:
        tpr, fnr = tpr_fnr(cm)
        _, fpr = tnr_fpr(cm)

        fprs.append(fpr)
        tprs.append(tpr)
        fnrs.append(fnr)

    _, (roc, det) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    # ROC
    roc.set_title("ROC")
    roc.plot(fprs, tprs)

    roc.grid(True)
    roc.set_xlim(0, 1)
    roc.set_ylim(0, 1)

    roc.set_xlabel("False positive rate")
    roc.set_ylabel("True positive rate")

    # DET
    det.set_title("DET")
    det.plot(fprs, fnrs)

    det.grid(True)
    det.set_xlim(0, 1)
    det.set_ylim(0, 1)

    det.set_xlabel("False positive rate")
    det.set_ylabel("False negative rate")

    plt.show()
