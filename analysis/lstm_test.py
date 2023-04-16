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
import torch.utils.data as data


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
lookback = 3 # how many samples in past seen

# Training parameters
lr = 0.1
epochs = 100
batch_size = 500



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
 
def create_dataset(dataset, lookback, future_steps = 1):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + lookback:i + lookback + future_steps]
        X.append(feature)
        y.append(target)
    return torch.Tensor(X), torch.Tensor(y)

# export to torch
dat_trn_t = create_dataset(dat_trn, lookback, future_steps = future_steps)
dat_trn_batch = DataLoader(data.TensorDataset(*dat_trn_t), 
                           batch_size = batch_size,
                           shuffle = True)
dat_cal_tst_t = create_dataset(dat_cal_tst, lookback, future_steps)

# =============================================================================
# Build LSTM network
# =============================================================================

class LSTM_network(nn.Module):
    def __init__(self,hidden_size, num_layers):
        super().__init__()
        self.LSTM = nn.LSTM(input_size = lookback, 
                            hidden_size = hidden_size, 
                            num_layers=num_layers)
        # self.lin_layer = nn.Linear(hidden_size,hidden_size, bias = True)
        # self.activation = torch.nn.ReLU()
        self.out_layer = nn.Linear(hidden_size,1, bias = True)

    def forward(self, x):
        x, _ = self.LSTM(x)
        # x = self.lin_layer(x)
        # x = self.activation(x)
        out = self.out_layer(x)
        return out
    
mod = LSTM_network(hidden_size, num_layers)
loss_function = nn.MSELoss()
optimizer = optim.SGD(mod.parameters(), lr=lr)


# =============================================================================
#%% Train network
# =============================================================================

for i in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    # iterate over minibatches
    for X,y in dat_trn_batch:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        mod.zero_grad()
    
        # Step 2. Run our forward pass.
        pred = mod(X)
    
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()
    
    if i % 5 == 0:
        print(f'{i:3d}/{epochs}, loss : {loss.item():7.1e}')
        
# =============================================================================
#%% Get calibration and test data results
# =============================================================================
with torch.no_grad():
    pred_cal_tst = mod(dat_cal_tst_t[0]).detach().numpy().flatten()
res = pd.DataFrame({'yhat': pred_cal_tst*std + mean,
                    'y': dat_cal_tst_t[1].detach().numpy().flatten()*std + mean})
res.to_pickle('pred.pickle')