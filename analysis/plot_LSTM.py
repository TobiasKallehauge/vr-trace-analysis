# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:53:20 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_pickle('pred.pickle')
res[['y','yhat']].plot()
plt.xlim(0,500)