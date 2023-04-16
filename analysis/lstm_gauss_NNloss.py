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
lookback = 3

# LSTM parameters
hidden_size = 7
num_layers = 1 # number of stacked LSTMS

# Training parameters
lr = 0.1
epochs = 500
batch_size = 1000
train_nn = True # otherwise load model from file

# lookback = 1 (or 2), hidden_size = 7, and batch_size = 1000, num_layers = 1 seems to work the best


# =============================================================================
#%% Load data
# =============================================================================

# Load the timeseries, the unit is kB
dat = RegressionPlotter([video], [rate], [fps]).dataframe.loc[guard:]*rate * 125000 / fps / 1000
mean_norm = float(dat.mean())
std_norm = float(dat.std())
dat_norm = (dat - mean_norm)/std_norm

# split dataset
N_trn = int(dat.size*train)
N_cal = int(dat.size*calibrate)
N_tst = dat.size - N_trn - N_cal
dat_trn = dat_norm[:N_trn].values.flatten()
dat_cal = dat_norm[N_trn:N_trn + N_cal].values.flatten()
dat_cal_tst = dat_norm[N_trn:].values.flatten()
dat_tst = dat[-N_trn:].values.flatten()

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
dat_cal_t = create_dataset(dat_cal, lookback, future_steps)
dat_cal_tst_t = create_dataset(dat_cal_tst, lookback, future_steps)
dat_tst_t = create_dataset(dat_tst, lookback, future_steps)


# =============================================================================
#%% Build LSTM network
# =============================================================================

class LSTM_network(nn.Module):
    def __init__(self,hidden_size, num_layers):
        super().__init__()
        self.LSTM = nn.LSTM(input_size = lookback, 
                        hidden_size = hidden_size, 
                        num_layers=num_layers)
        self.lin_layer = nn.Linear(hidden_size,hidden_size, bias = True)
        self.activation = torch.nn.ReLU()
        self.out_layer = nn.Linear(hidden_size,2, bias = True)

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = self.lin_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        mean = x[:,0]
        std = x[:,1]
        var = torch.square(std) # std could be neagive
        out = torch.stack((mean,var), dim = 1)
        return out
    
mod = LSTM_network(hidden_size, num_layers) # model to predict mean and variance
loss_function = nn.GaussianNLLLoss()
optimizer = optim.SGD(mod.parameters(), lr=lr)

# =============================================================================
#%% Train network
# =============================================================================

def train_epoch(model,dataset):
    for X,y in dataset:

        model.zero_grad()

        out = model(X)
        mean = out[:,0:1] # mean of prediction
        var = out[:,1:2] # variance of prediction
    
        loss = loss_function(mean, y, var)
        loss.backward()
        optimizer.step()
    
model_path = 'models/mod_LSTM'
if train_nn:
    best_vloss = np.inf
    
    for i in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    
        # train on minibatches
        mod.train(True)
        train_epoch(mod,dat_trn_batch)
        
        mod.train(False)
        
        # evaluate on training dataset
        out = mod(dat_trn_t[0])
        mean = out[:,0:1] # mean of prediction
        var = out[:,1:2] # variance of prediction
        tloss = loss_function(mean, dat_trn_t[1], var)
        
        # evaluate on calibration dataset
        out = mod(dat_cal_t[0])
        mean = out[:,0:1] # mean of prediction
        var = out[:,1:2] # variance of prediction
        vloss = loss_function(mean, dat_cal_t[1], var)
    
        if vloss < best_vloss:
            best_vloss = vloss
    
            torch.save(mod.state_dict(), model_path)
            
        if i % 5 == 0:
            print(f'{i:3d}/{epochs}. Loss:  training : {tloss.item():7.1e}, validation : {vloss.item():7.1e}, best validation: {best_vloss:.1e}')
        
# load best model
mod.load_state_dict(torch.load(model_path))
        
# =============================================================================
#%% Get calibration and test data results used for conformal prediction
# =============================================================================
with torch.no_grad():
    pred_cal_tst = mod(dat_cal_tst_t[0]).detach().numpy()
    
# load dataset from linear model
res = pd.read_pickle(f'results/{video}_{rate}_{fps}_linear_pred.pickle')[3:]




res['mean_LSTM'] = pred_cal_tst[:,0]*std_norm + mean_norm
res['std_LSTM'] =  np.sqrt(pred_cal_tst[:,1])*std_norm
res.to_pickle(f'results/{video}_{rate}_{fps}_pred.pickle')


res = pd.DataFrame({'mean': pred_cal_tst[:,0]*std_norm + mean_norm,
                    'var':  pred_cal_tst[:,1]*std_norm**2,
                    'y': dat_cal_tst_t[1].flatten()*std_norm + mean_norm})
res.to_pickle('pred_mean_var.pickle')

