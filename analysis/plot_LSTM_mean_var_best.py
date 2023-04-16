# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:53:20 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_pickle('pred_mean_var.pickle')
std = np.sqrt(res['var'])

fig, ax = plt.subplots()

ax.fill_between(res.index,res['mean'] - 1.96*std,res['mean'] + 1.96*std, alpha = 0.5, color = 'gray')
res[['mean','y']].plot(ax = ax)

plt.xlim(0,500)