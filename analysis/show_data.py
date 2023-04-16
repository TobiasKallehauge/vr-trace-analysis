# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:44:34 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

video = 'vp' # options are 'mc', 'ge_cities', 'ge_tour', 'vp'
rate = 30
fps = 60
ds = 1/fps

calibrate, test = [0.6,0.5]


alpha = 0.05
q = norm.ppf(1 - alpha/2)

res = pd.read_pickle(f'results/{video}_{rate}_{fps}_pred.pickle')
res['t'] = np.arange(0,len(res)*ds, step = ds)*1000 # time in miliseconds



# =============================================================================
# Plot some predictions
# =============================================================================

idx = np.arange(3,500)

fig, ax = plt.subplots(figsize = (6,3))

ax.set_title('Linear model')
ax.plot(res['t'][idx],res['y'][idx], c= 'k')
ax.set_xticks([])
ax.set_ylabel('Packet size [kB]')
ax.set_xlabel('Time [ms]')
ax.set_title('XR steam packet sizes')
fig.savefig('data.png', dpi = 500)


