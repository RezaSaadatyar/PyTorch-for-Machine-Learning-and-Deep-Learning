# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# ================================== E-mail: Reza.Saadatyar@outlook.com ========================================

import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_outlier(data: Union[pd.DataFrame, np.ndarray], title: str = None, figsize: Union[int, float] = (3, 2.5)) -> None:
    plt.figure(figsize=figsize)

    # Create the boxplot with custom outlier properties
    boxplot = plt.boxplot(data, 
                      boxprops=dict(color='green'), 
                      whiskerprops=dict(color='red'), 
                      medianprops=dict(color='blue'), 
                      capprops=dict(color='gray'), 
                      flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, markeredgecolor='black'), 
                      patch_artist=True)

    # Customize the plot
    plt.xlabel('Features', fontsize=10, va='center')
    plt.tick_params(axis='x', length=2, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=1)
    plt.tick_params(axis='y', length=2, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=1)

    # Create custom legend handles
    green_patch = mpatches.Patch(color='green', label='Box')
    red_patch = mpatches.Patch(color='red', label='Whiskers')
    blue_patch = mpatches.Patch(color='blue', label='Median')
    gray_patch = mpatches.Patch(color='gray', label='Caps')
    orange_patch = mpatches.Patch(color='orange', label='Outliers')

    plt.title(title, fontsize=10) # Add a title
    plt.legend(handles=[green_patch, red_patch, blue_patch, gray_patch, orange_patch], loc='best') # Add the legend
    plt.show()