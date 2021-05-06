# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:59:17 2021

@author: gino9
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

def roc_det_curves(confusion_matrixes: List[np.ndarray]) -> None:
    """
    Plot ROC and DET curve.

    Parameters
    ----------
    confusion_matrix : List[np.ndarray]
        list of size t of ndarrays of size(2,2).

    Returns
    -------
    None

    """
    fprs = []
    tprs = []
    fnrs = []
    for cm in confusion_matrixes:
        
        tpr = cm[1, 1]/cm[:, 1].sum()
        fpr = cm[1, 0]/cm[:, 0].sum()
        fnr = 1 - tpr
        
        fprs.append(fpr)
        tprs.append(tpr)
        fnrs.append(fnr)
        
    fig, (roc, det) = plt.subplots(
        nrows=1, ncols=2, constrained_layout=True)
    
    roc.set_title("ROC")
    roc.plot(fprs, tprs)
    roc.set_xlabel("False positive rate")
    roc.set_ylabel("True positive rate")
    
    det.set_title("DET")
    det.plot(fprs, fnrs)
    det.set_xlabel("False positive rate")
    det.set_ylabel("False negative rate")
    
    plt.show()