# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:08:39 2021

@author: DarthReca
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_matrix(matrix: np.ndarray, colormap: str) -> None:
    """
    Plot a matrix as a square plot and show values as colors.

    Parameters
    ----------
    matrix : np.ndarray
        
    colormap : str
        matplotlib colormap name.
    """
    fig, ax = plt.subplots()
    
    im = ax.imshow(matrix, cmap=colormap)
    
    ax.set_xticks([i for i in range(matrix.shape[1])])
    ax.set_yticks([i for i in range(matrix.shape[0])])
    
    fig.colorbar(im, ax=ax)
    
    plt.show()