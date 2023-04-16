# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:52:57 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from regression_plotter import RegressionPlotter
from statsmodels.tsa.arima.model import ARIMA


# =============================================================================
# Setup scenario
# =============================================================================

video = 'vp' # options are 'mc', 'ge_cities', 'ge_tour', 'vp'
rate = 30
fps = 60
train, calibrate, test = [0.3, 0.2, 0.5] # split of training, calibation and test in percentages (should sum to 1)
assert np.abs(train + calibrate + test - 1) < 1e-10
alpha = 0.05

# Regression options
method = 'Linear' # Quantile is also an option
guard = 200 # samples thrown away in the beginning

future_steps = 1 # 1 or 6 in article
shift = 1 # tau in the paper
past_steps = 10 # N in the paper
future_steps = 1 # T in the paper

def compute_achieved_conf(video, rate, fps):


    # Load the timeseries, the unit is kB
    dat = RegressionPlotter([video], [rate], [fps]).dataframe.loc[guard:]*rate * 125000 / fps / 1000
    
    # split dataset
    N_trn = int(dat.size*train)
    N_cal = int(dat.size*calibrate)
    N_tst = dat.size - N_trn - N_cal
    dat_trn = dat[:N_trn].values.flatten()
    dat_cal = dat[N_trn: N_trn + N_cal].values.flatten()
    dat_tst = dat[-N_tst:].values.flatten()

# =============================================================================
# Fit linear models based on the Yule-Walker quations
# =============================================================================

    mod_trn = ARIMA(dat_trn, 
                order = (past_steps, 0, 0),
                trend = 'c') # constant trend)
    res = mod_trn.fit()

    
# =============================================================================
#%% Use calibration set to correct the confidence interval
# =============================================================================

    res_cal = res.extend(dat_cal)
    pred_cal = res_cal.get_prediction().summary_frame()
    resid = pred_cal['mean'] - dat_cal
    
    # compute the score functions as the normalizes abosulute residual
    s = np.abs(resid)/pred_cal['mean_se'].values
    s = s[::past_steps] # only uncorrelated samples
    N_cal = s.size
    
    # get corrected quantile values
    alpha_corrected = np.ceil((1-alpha)*(N_cal+1))/N_cal
    q_corrected = np.quantile(s, alpha_corrected)

# =============================================================================
#%% evaluate on Test data
# =============================================================================

    res_tst = res.extend(dat_tst)
    pred_tst = res_tst.get_prediction().summary_frame()
    pred_tst = pred_tst[::past_steps]
    dat_tst = dat_tst[::past_steps]
    pred_tst['mean_ci_lower_CP'] = pred_tst['mean'] - pred_tst['mean_se']*q_corrected
    pred_tst['mean_ci_upper_CP'] = pred_tst['mean'] + pred_tst['mean_se']*q_corrected
    
    p_TSA = np.mean((dat_tst > pred_tst['mean_ci_lower']) & (dat_tst < pred_tst['mean_ci_upper']))
    p_CDF = np.mean((dat_tst > pred_tst['mean_ci_lower_CP']) & (dat_tst < pred_tst['mean_ci_upper_CP']))
    return(p_TSA,p_CDF)

videos = ['ge_cities', 'ge_tour','mc', 'vp']
rates = [10,20,30,40,50]
fps = [30,60]
settings = [(v, r, f) for v in videos for r in rates for f in fps]
N_set = len(settings)
p_TSA = np.zeros(N_set)
p_CDF = np.zeros(N_set)
for i, setting in enumerate(settings):
    print(setting)
    res = compute_achieved_conf(*setting)
    p_TSA[i] = res[0]
    p_CDF[i] = res[1]

# =============================================================================
#%% Plot results
# =============================================================================


# regular plot
fig, ax = plt.subplots()
ax.plot(p_TSA, label = 'TSA')
ax.plot(p_CDF, label = 'CP')
ax.axhline(1-alpha, c = 'r')
ax.legend()

