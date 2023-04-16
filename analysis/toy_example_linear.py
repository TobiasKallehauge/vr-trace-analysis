# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:08:54 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(72)

d = 5
var = 1
N_trn = 10000
N_cal = 1000
N_tst = 10000
alpha = 0.05
theta = np.random.random(d)


def get_data(N):
    X = np.random.random((N,d))    
    n = np.random.normal(scale = np.sqrt(var), size = N)
    y = X.dot(theta) + n
    return(X,y)

# =============================================================================
# training
# =============================================================================

X_trn, y_trn = get_data(N_trn)


# estiamte theta
theta_est = np.linalg.lstsq(X_trn, y_trn, rcond = None)[0]

# plot theta results
plt.stem(theta, label = r'$\theta$', markerfmt='C0o')
plt.stem(theta_est, label = r'$\hat{\theta}$', markerfmt='C1o')
plt.legend()

# =============================================================================
# Calibration
# =============================================================================

X_cal, y_cal = get_data(N_cal)

y_pred = X_cal.dot(theta_est)    


resid = y_pred - y_cal

# compute the score functions as the abosulute residual
s = np.abs(resid)

# get corrected quantile values
alpha_corrected = np.ceil((1-alpha)*(N_cal+1))/N_cal
q_corrected = np.quantile(s, alpha_corrected)

# =============================================================================
# Evaluate on test data
# =============================================================================

X_tst, y_tst = get_data(N_tst)

y_pred = X_tst.dot(theta_est)

low_bound = y_pred - q_corrected
high_bound = y_pred + q_corrected


p = np.mean((y_tst > low_bound) & (y_tst < high_bound))
print(f'Achieved conf: {p*100:.1f}%')