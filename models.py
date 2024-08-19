# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : Mechanics-AE for damage detection and localization
# //   File        : models.py
# //   Description : Mechanics-AE models
# //
# //   Created On: 3/7/2024
# /////////////////////////////////////////////////////////////////

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as spio
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cpu'


class MIAE(pl.LightningModule): #Mechanics-AE
    def __init__(self, sensor_index, lr, net_size):
        super(MIAE, self).__init__()
        self.sensor_index = sensor_index
        self.lr = lr
        n = 1 if np.isscalar(sensor_index) else len(sensor_index)
        self.n = n
        self.linear1 = nn.Linear(2*n, net_size[0]*n)
        self.linear2 = nn.Linear(net_size[0]*n, net_size[1]*n)
        self.linear22 = nn.Linear(net_size[1]*n, net_size[2]*n)
        self.linear33 = nn.Linear(net_size[2]*n, net_size[3]*n)
        self.linear3 = nn.Linear(net_size[3]*n, net_size[4]*n)
        self.linear4 = nn.Linear(net_size[4]*n, 2*n)
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()
        
    def get_weight(self, args):
        if args.experiment == 'crack':
            w = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
            for i in range(len(w)):
                for j in range(len(w)):
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
            w[12,:] = w[12,:]/1.5
            w[:,12] = w[:,12]/1.5
            w[0,:] = w[0,:]/2
            w[:,0] = w[:,0]/2
        elif args.experiment == 'bc':
            w = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
            for i in range(len(w)):
                for j in range(len(w)):
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
            w[16,:] = w[16,:]/3.5
            w[:,16] = w[:,16]/3.5
        elif args.experiment in ['simu', 'simu_temp', 'simu_noise']:
            a = spio.loadmat('data/W-simu', squeeze_me=True)['w1d']
            ns = len(a)
            w = np.zeros((ns,ns))
            for i in range(ns):
                for j in range(ns):
                    w[i,j]= a[i]/a[j]
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
        elif args.experiment == 'beam_column':
            a = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
            ns = len(a)
            w = np.zeros((ns,ns))
            for i in range(len(w)):
                for j in range(len(w)):
                    w[i,j]= a[i]/a[j]
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
        self.weight = torch.tensor(w[self.sensor_index, self.sensor_index], dtype=torch.float32).to(device)
        
    def mechanics_loss(self, x): 
        n = len(x.shape)
        x1 = torch.unsqueeze(x, n)
        m1 = torch.ones((1,x.shape[-1])).to(device)
        b1 = torch.matmul(x1,m1)
        b2 = b1.permute((0,2,1))
        dist = self.relu((b1-b2)**2-0.001*b2)
        return torch.mul(dist, self.weight)
    
    def load_data(self, train_data, test_data):
        self.train_data = train_data.astype(np.float32)
        self.test_data = test_data.astype(np.float32)
        
    def load_new_data(self, new_data):
        self.new_data = new_data.astype(np.float32)
    
    def encoder(self, inp):
        z = self.relu(self.linear1(inp))
        z = self.relu(self.linear2(z))
        z = self.linear22(z)
        return z 
    
    def decoder(self, inp_z):
        out = self.linear33(inp_z)
        out = self.linear3(out)
        out = self.linear4(out)
        return out
        
    def loss_function(self, inp, out): 
        if len(self.sensor_index) == 2:
            weights = 90
        if len(self.sensor_index) == 4:
            weights = 165 #90
        elif len(self.sensor_index) == 6:
            weights = 90 #60 #60
        else:
            weights = 30
        dd = torch.norm(out,dim=1) - torch.norm(inp,dim=1)
        W_loss1 = torch.mean( self.mechanics_loss(dd[:,:self.n]) )/weights # 30
        W_loss2 = torch.mean( self.mechanics_loss(dd[:,self.n:]) )/weights # 30
        
        reconstruction = self.mse(inp, out)
        # print(f'{reconstruction:0.2e}---{W_loss1:0.2e}---{W_loss2:0.2e}')
        return reconstruction + W_loss1 + W_loss2 
    
    def forward(self, inp):
        z = self.encoder(inp)
        out = self.decoder(z) 
        return z, None, out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        mseloss = torch.mean((out-inp)**2,1)
        normloss = (torch.norm(out,dim=1) - torch.norm(inp,dim=1))**2
        return torch.cat( (mseloss, normloss) )
    
    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size=30, shuffle=True) 
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=30, shuffle=False) 


class AE(pl.LightningModule):
    def __init__(self, sensor_index, lr, net_size):
        super(AE, self).__init__()
        self.sensor_index = sensor_index
        self.lr = lr
        n = 1 if np.isscalar(sensor_index) else len(sensor_index)
        self.n = n
        self.linear1 = nn.Linear(2*n, net_size[0]*n)
        self.linear2 = nn.Linear(net_size[0]*n, net_size[1]*n)
        self.linear22 = nn.Linear(net_size[1]*n, net_size[2]*n)
        self.linear33 = nn.Linear(net_size[2]*n, net_size[3]*n)
        self.linear3 = nn.Linear(net_size[3]*n, net_size[4]*n)
        self.linear4 = nn.Linear(net_size[4]*n, 2*n)
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()
    
    def load_data(self, train_data, test_data):
        self.train_data = train_data.astype(np.float32)
        self.test_data = test_data.astype(np.float32)
        
    def load_new_data(self, new_data):
        self.new_data = new_data.astype(np.float32)
    
    def encoder(self, inp):
        z = self.relu(self.linear1(inp))
        z = self.relu(self.linear2(z))
        z = self.linear22(z)
        return z 
    
    def decoder(self, inp_z):
        out = self.linear33(inp_z)
        out = self.linear3(out)
        out = self.linear4(out)
        return out
        
    def loss_function(self, inp, out): 
        reconstruction = self.mse(inp, out)
        # self.log('mse_loss', reconstruction, on_step=False, on_epoch=True, logger=True)
        # print(f'{reconstruction:0.2e}')
        return reconstruction 
    
    def forward(self, inp):
        z = self.encoder(inp)
        out = self.decoder(z) 
        return z, None, out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        mseloss = torch.mean((out-inp)**2,1)
        normloss = (torch.norm(out,dim=1) - torch.norm(inp,dim=1))**2
        return torch.cat( (mseloss, normloss) )
    
    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size=30, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=30, shuffle=False) 
    

##############################################
########## model for all sensors #############
##############################################
class MIAE_AS(pl.LightningModule): #MIAE_AS all sensors
    def __init__(self, h_dim):
        super(MIAE_AS, self).__init__()
        if h_dim == 8:    # exp
            net_size = [52,32,16,8,32,52]
        elif h_dim == 20: # simu
            net_size = [90,64,32,20,64,90]
        self.linear1 = nn.Linear(net_size[0], net_size[1])
        self.linear2 = nn.Linear(net_size[1], net_size[2])
        self.linear22 = nn.Linear(net_size[2], net_size[3])
        self.linear3 = nn.Linear(net_size[3], net_size[4])
        self.linear4 = nn.Linear(net_size[4], net_size[5])
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()

    def get_weight(self, args):
        if args.experiment == 'crack':
            w = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
            for i in range(len(w)):
                for j in range(len(w)):
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
            w[12,:] = w[12,:]/1.5
            w[:,12] = w[:,12]/1.5
            w[0,:] = w[0,:]/2
            w[:,0] = w[:,0]/2
        elif args.experiment == 'bc':
            w = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
            for i in range(len(w)):
                for j in range(len(w)):
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
            w[16,:] = w[16,:]/3.5
            w[:,16] = w[:,16]/3.5
        elif args.experiment in ['simu']:
            a = spio.loadmat('data/W-simu', squeeze_me=True)['w1d']
            ns = len(a)
            w = np.zeros((ns,ns))
            for i in range(ns):
                for j in range(ns):
                    w[i,j]= a[i]/a[j]
                    if w[i,j] > 1:
                        w[i,j] = 1/w[i,j] 
        self.weight = torch.tensor(w**1, dtype=torch.float32).to(device)
        
    def mechanics_loss(self, x): 
        n = len(x.shape)
        x1 = torch.unsqueeze(x, n)
        m1 = torch.ones((1,x.shape[-1])).to(device)
        b1 = torch.matmul(x1,m1)
        b2 = b1.permute((0,2,1))
        dist = self.relu((b1-b2)**2-0.01*b2)
        return torch.mul(dist, self.weight)

    def load_data(self, train_data, test_data):
        self.train_data = train_data.astype(np.float32)
        self.test_data = test_data.astype(np.float32)
    
    def encoder(self, inp):
        z = self.linear1(inp)
        z = self.linear2(z)
        z = self.linear22(z)
        return z 
    
    def decoder(self, inp_z):
        out = self.linear3(inp_z)
        out = self.linear4(out)
        return out
        
    def loss_function(self, inp, out): 
        dd = torch.norm(out,dim=1) - torch.norm(inp,dim=1)
        W_loss1 = torch.mean( self.mechanics_loss(dd[:,:26]) )/20
        W_loss2 = torch.mean( self.mechanics_loss(dd[:,26:]) )/20
        
        reconstruction = self.mse(inp, out)
        self.log('mse_loss', reconstruction, on_step=False, on_epoch=True, logger=True)
        print(f'{reconstruction:0.2e}---{W_loss1:0.2e}---{W_loss2:0.2e}')
        return reconstruction + W_loss1 + W_loss2 
    
    def forward(self, inp):
        z = self.encoder(inp)
        out = self.decoder(z) 
        return z, None, out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-3)
    
    def training_step(self, inp, batch_nb):
        z, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, inp, batch_nb):
        z, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        mseloss = torch.mean((out-inp)**2,1)
        normloss = (torch.norm(out,dim=1) - torch.norm(inp,dim=1))**2
        return torch.cat( (mseloss, normloss) )
    
    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size = 30, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size = 30, shuffle=False) 
 
class AE_AS(pl.LightningModule): # AE_AS
    def __init__(self, h_dim, experiment):
        super(AE_AS, self).__init__()
        if h_dim == 8 and experiment == 'simu':    # exp
            net_size = [52,32,16,8,32,52] 
        elif h_dim == 8 and experiment != 'simu':
            net_size = [52,64,32,8,64,52]
        elif h_dim == 20: # simu
            net_size = [90,64,32,20,64,90]
        self.linear1 = nn.Linear(net_size[0], net_size[1])
        self.linear2 = nn.Linear(net_size[1], net_size[2])
        self.linear22 = nn.Linear(net_size[2], net_size[3])
        self.linear3 = nn.Linear(net_size[3], net_size[4])
        self.linear4 = nn.Linear(net_size[4], net_size[5])
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()
    
    def load_data(self, train_data, test_data):
        self.train_data = train_data.astype(np.float32)
        self.test_data = test_data.astype(np.float32)
        
    def load_new_data(self, new_data):
        self.new_data = new_data.astype(np.float32)
    
    def encoder(self, inp):
        z = self.relu(self.linear1(inp))
        z = self.relu(self.linear2(z))
        z = self.linear22(z)
        return z 
    
    def decoder(self, inp_z):
        out = self.linear3(inp_z)
        out = self.linear4(out)
        return out
        
    def loss_function(self, inp, out): 
        reconstruction = self.mse(inp, out)
        self.log('mse_loss', reconstruction, on_step=False, on_epoch=True, logger=True)
        print(f'{reconstruction:0.2e}')
        return reconstruction 
    
    def forward(self, inp):
        z = self.encoder(inp)
        out = self.decoder(z) 
        return z, None, out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-3)
    
    def training_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        loss = self.loss_function(inp, out) 
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, inp, batch_nb):
        _, _, out = self.forward(inp)
        mseloss = torch.mean((out-inp)**2,1)
        normloss = (torch.norm(out,dim=1) - torch.norm(inp,dim=1))**2
        return torch.cat( (mseloss, normloss) )
    
    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size = 30, shuffle=True) 
  
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size = 30, shuffle=False) 
    