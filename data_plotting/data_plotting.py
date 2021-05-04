# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:58:53 2021

@author: Hossein.JvdZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sys
# sys.path.append("..")
# import data_loading as dl

pulsar_star = pd.read_csv('Test.csv') #ReadTest Data from local file /Test.csv/ 

target_dataframe = pulsar_star['target_pulsar'] #Possibility of existence

pulsar_star = pulsar_star.drop(columns='target_pulsar') #Emit Possibility of existence the Pulsar to use it just in target_dataframe

fig, axes = plt.subplots(2,4, figsize=(12, 4)) # 2 columns each containing 4 figures, total 8 features

wrong = pulsar_star[target_dataframe==0] # Non-accurate data
pulsar = pulsar_star[target_dataframe==1]  # Accurate data

ax = axes.ravel() # flat axes with numpy ravel
# def plot_test_data():
for i in range(pulsar_star.columns.size):
    _,bins = np.histogram(pulsar_star.iloc[:, i], bins=35) #resulation of each axes
    ax[i].hist(wrong.iloc[:,i],bins=bins,color='r',alpha=0.2) #red color to show false prediction 
    ax[i].hist(pulsar.iloc[:,i],bins=bins,color='g',alpha=0.5) #green color to show true prediction
    ax[i].set_title(pulsar_star.columns[i], fontsize=18) #increase fontsize to 16 for better report image
    ax[i].axes.get_xaxis().set_visible(False)  #the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
    ax[i].set_yticks(())
    ax[i].axis(ymax=200)
    
ax[0].legend(['Wrong', 'Pulsar'], loc='best', fontsize=10)
plt.tight_layout() # let's make good plots
plt.show()