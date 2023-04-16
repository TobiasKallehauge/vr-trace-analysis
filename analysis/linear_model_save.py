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
train, calibrate, test = [0.5, 0.2, 0.3] # split of training, calibation and test in percentages (should sum to 1)
alpha = 0.05

# Regression options
method = 'Linear' # Quantile is also an option
guard = 200 # samples thrown away in the beginning

future_steps = 1 # 1 or 6 in article
shift = 1 # tau in the paper
past_steps = 6 # N in the paper
future_steps = 1 # T in the paper


# Load the timeseries, the unit is kB
dat = RegressionPlotter([video], [rate], [fps]).dataframe.loc[guard:]*rate * 125000 / fps / 1000

# split dataset
N_trn = int(dat.size*train)
N_cal = int(dat.size*calibrate)
N_tst = dat.size - N_trn - N_cal
dat_trn = dat[:N_trn].values.flatten()
dat_cal_tst = dat[N_trn:].values.flatten()

# =============================================================================
# Fit linear models based on the Yule-Walker quations
# =============================================================================

mod_trn = ARIMA(dat_trn, 
            order = (past_steps, 0, 0),
            trend = 'c') # constant trend)
res = mod_trn.fit()
pred_trn = res.get_prediction().summary_frame()
params = {name : val for name, val in zip(res.param_names,res.params)}
print(res.summary())

# plot some predictions
plt.plot(dat_trn)
plt.plot(res.predict())
plt.xlim(0,500)
plt.show()

# =============================================================================
#%% Predict and save dataset
# =============================================================================

res_cal_tst  = res.extend(dat_cal_tst) 
pred_cal_tst = res_cal_tst.get_prediction().summary_frame()
pred_cal_tst = pred_cal_tst[['mean','mean_se']]
pred_cal_tst.columns = ['mean_TSA', 'std_TSA']
pred_cal_tst['y'] = dat_cal_tst

# save
pred_cal_tst.to_pickle(f'results/{video}_{rate}_{fps}_linear_pred.pickle')