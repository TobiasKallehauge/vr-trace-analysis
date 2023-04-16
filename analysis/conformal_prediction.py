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

calibrate, test = [0.6,0.4]


alpha = 0.05
q = norm.ppf(1 - alpha/2)

res = pd.read_pickle(f'results/{video}_{rate}_{fps}_pred.pickle')
res['t'] = np.arange(0,len(res)*ds, step = ds)*1000 # time in miliseconds

N_cal = int(len(res)*calibrate)
N_tst = len(res) - N_cal

# np.random.seed(1)
res_shuffle = res.sample(frac = 1)
# res_shuffle = res

res_cal = res_shuffle[:N_cal]
res_tst = res_shuffle[N_cal:]

# =============================================================================
# Run conformal prediction
# =============================================================================

# Linear model
resid_TSA = res_cal['mean_TSA'] - res_cal['y']

# compute the score functions as the normalizes abosulute residual
scores = np.abs(resid_TSA)/res_cal['std_TSA']

# get corrected quantile values
alpha_corrected = np.ceil((1-alpha)*(N_cal+1))/N_cal
q_TSA = np.quantile(scores, alpha_corrected)

# LSTM
resid_TSA = res_cal['mean_LSTM'] - res_cal['y']

# compute the score functions as the normalizes abosulute residual
scores = np.abs(resid_TSA)/res_cal['std_LSTM']

# get corrected quantile values
alpha_corrected = np.ceil((1-alpha)*(N_cal+1))/N_cal
q_LSTM = np.quantile(scores, alpha_corrected)


# =============================================================================
# Plot some predictions
# =============================================================================

idx = np.arange(3,500)

fig, ax = plt.subplots(nrows = 2, figsize = (8,6))

ax[0].set_title('Linear model')
ax[0].plot(res['t'][idx],res['y'][idx], label = 'size')
ax[0].plot(res['t'][idx],res['mean_TSA'][idx], label = 'prediction')
ax[0].fill_between(res['t'][idx],
                   res['mean_TSA'][idx] - q_TSA*res['std_TSA'][idx],
                   res['mean_TSA'][idx] + q_TSA*res['std_TSA'][idx],
                   label = '95% conf. interval',
                   color = 'gray',
                   alpha = 0.5)
ax[0].set_xticks([])
ax[0].set_ylabel('Packet size [kB]')
ax[0].legend(loc = 'upper right')
ax[0].set_ylim(27,104)

ax[1].set_title('LSTM neural network')
ax[1].plot(res['t'][idx], res['y'][idx], label = 'size')
ax[1].plot(res['t'][idx], res['mean_LSTM'][idx], label = 'prediction')
ax[1].fill_between(res['t'][idx],
                    res['mean_LSTM'][idx] - q_LSTM*res['std_LSTM'][idx],
                    res['mean_LSTM'][idx] + q_LSTM*res['std_LSTM'][idx],
                    label = '95% conf. interval',
                    color = 'gray',
                    alpha = 0.5)
ax[1].set_xlabel('Time [ms]')
ax[1].set_ylabel('Packet size [kB]')
ax[1].set_ylim(30,100)
fig.suptitle('Prediction of XR packet sizes')
fig.savefig('figures/predictions.png', bbox_inches = 'tight', dpi = 500)


# =============================================================================
# Compute prediction intervals
# =============================================================================

# get number of points within prediction intervals
p_TSA = np.mean((res_tst['y'] > res_tst['mean_TSA'] - q*res_tst['std_TSA']) &\
                (res_tst['y'] < res_tst['mean_TSA'] + q*res_tst['std_TSA']))
p_TSA_CP = np.mean((res_tst['y'] > res_tst['mean_TSA'] - q_TSA*res_tst['std_TSA']) &\
                (res_tst['y'] < res_tst['mean_TSA'] + q_TSA*res_tst['std_TSA']))
p_LSTM = np.mean((res_tst['y'] > res_tst['mean_LSTM'] - q*res_tst['std_LSTM']) &\
                (res_tst['y'] < res_tst['mean_LSTM'] + q*res_tst['std_LSTM']))
p_LSTM_CP = np.mean((res_tst['y'] > res_tst['mean_LSTM'] - q_LSTM*res_tst['std_LSTM']) &\
                (res_tst['y'] < res_tst['mean_LSTM'] + q_LSTM*res_tst['std_LSTM']))
    
print(f'Achieved conf TSA    : {p_TSA*100:.1f}%')
print(f'Achieved conf TSA CP : {p_TSA_CP*100:.1f}%')
print(f'Achieved conf LSTM   : {p_LSTM*100:.1f}%')
print(f'Achieved conf LSTM CP: {p_LSTM_CP*100:.1f}%')
    
# get sizes of prediction invervals
    
C_size_TSA_CP = np.mean(2*res_tst['std_TSA']*q_TSA)
C_size_LSTM_CP = np.mean(2*res_tst['std_LSTM']*q_LSTM)

print(f'Mean size CI TSA : {C_size_TSA_CP:.1f} kB')
print(f'Mean size CI LSTM: {C_size_LSTM_CP:.1f} kB')

