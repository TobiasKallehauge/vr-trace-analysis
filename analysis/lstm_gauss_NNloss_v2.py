# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:48:01 2023

@author: Tobias Kallehauge
"""

import numpy as np
from regression_plotter import RegressionPlotter
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim


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
past_steps = 3 # N in the paper
future_steps = 1 # T in the paper

# LSTM parameters
hidden_size = 10
num_layers = 3 # number of stacked LSTMS

# Training parameters
lr = 0.1
epochs = 100
batch_size = 1000



# =============================================================================
# Load data
# =============================================================================

# Load the timeseries, the unit is kB
dat = RegressionPlotter([video], [rate], [fps]).dataframe.loc[guard:]*rate * 125000 / fps / 1000
mean = float(dat.mean())
std = float(dat.std())
dat_norm = (dat - mean)/std

# split dataset
N_trn = int(dat.size*train)
N_cal = int(dat.size*calibrate)
N_tst = dat.size - N_trn - N_cal
dat_trn = dat_norm[:N_trn].values.flatten()
dat_cal_tst = dat_norm[N_trn:].values.flatten()

# export to torch
dat_trn_t = torch.Tensor(dat_trn).reshape(N_trn,1)
dat_trn_batch = DataLoader(dat_trn_t, batch_size = batch_size)
dat_cal_tst_t = torch.Tensor(dat_cal_tst).reshape(N_cal + N_tst,1)

# =============================================================================
# LSTM network class
# =============================================================================

class LSTM_network(nn.Module):
    def __init__(self,hidden_size, num_layers, square = False):
        super().__init__()
        self.LSTM = nn.LSTM(input_size = 1, 
                        hidden_size = hidden_size, 
                        num_layers=num_layers,
                        dropout = 0.1)
        self.lin_layer = nn.Linear(hidden_size,hidden_size, bias = True)
        self.activation = torch.nn.ReLU()
        self.out_layer = nn.Linear(hidden_size,1, bias = True)
        self.square = square

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = self.lin_layer(x)
        x = self.activation(x)
        out = self.out_layer(x)
        if self.square:
            out = torch.square(out)
        return out

# =============================================================================
# Train model for mean
# =============================================================================
    
mod_mean = LSTM_network(hidden_size, num_layers) # model to predict mean and variance
loss_function = nn.MSELoss()
optimizer = optim.SGD(mod_mean.parameters(), lr=lr)


print('Fitting mean')

for i in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    # iterate over minibatches
    for d in dat_trn_batch:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        mod_mean.zero_grad()
    
        # Step 2. Run our forward pass.
        out = mod_mean(d)
    
        # Step 3. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        
        loss = loss_function(out, d)
        loss.backward()
        optimizer.step()
    
    if i % 5 == 0:
        print(f'{i:3d}/{epochs}, loss : {loss.item():7.1e}')

# =============================================================================
#%% Now train model for variance here just given the time series
# =============================================================================



mod_var = LSTM_network(5, num_layers, square = True) # model to predict mean and variance
loss_function = nn.GaussianNLLLoss()
optimizer = optim.SGD(mod_var.parameters(), lr=lr)

print('Fitting variance')

for i in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    # iterate over minibatches
    for d in dat_trn_batch:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        mod_var.zero_grad()
    
        # Step 2. Run our forward pass.
        mean = mod_mean(d)
        var = mod_var(d)
        
    
        # Step 3. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        
        loss = loss_function(mean, d, var)
        loss.backward()
        optimizer.step()
    
    if i % 5 == 0:
        print(f'{i:3d}/{epochs}, loss : {loss.item():7.1e}')



# =============================================================================
#%% Get calibration and test data results
# =============================================================================
with torch.no_grad():
    mean = mod_mean(dat_cal_tst_t).detach().numpy().flatten()
    var = mod_var(dat_cal_tst_t).detach().numpy().flatten()
res = pd.DataFrame({'mean': mean,
                    'var':  var,
                    'y': dat_cal_tst})
res.to_pickle('pred_mean_var.pickle')