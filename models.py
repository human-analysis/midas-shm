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

class MechanicsAutoEncoder(pl.LightningModule): #Mechanics-AE
    def __init__(self, h_dim):
        super(MechanicsAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(52, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear22 = nn.Linear(16, h_dim)
        self.linear3 = nn.Linear(h_dim, 32)
        self.linear4 = nn.Linear(32, 52)
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()
        
    def get_weight(self, args):
        w = spio.loadmat(f'data/W-{args.experiment}', squeeze_me=True)['W']
        for i in range(len(w)):
            for j in range(len(w)):
                if w[i,j] > 1:
                    w[i,j] = 1/w[i,j] 
        if args.experiment == 'crack':
            w[12,:] = w[12,:]/1.5
            w[:,12] = w[:,12]/1.5
            w[0,:] = w[0,:]/2
            w[:,0] = w[:,0]/2
        elif args.experiment == 'bc':
            w[16,:] = w[16,:]/3.5
            w[:,16] = w[:,16]/3.5
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



class MechanicsAutoEncoderSimu(pl.LightningModule): # a larger net for the simulation task
    def __init__(self, h_dim):
        super(MechanicsAutoEncoderSimu, self).__init__()
        self.linear1 = nn.Linear(90, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear22 = nn.Linear(32, h_dim)
        self.linear3 = nn.Linear(h_dim, 64)
        self.linear4 = nn.Linear(64, 90)
        self.mse = nn.MSELoss(reduction = 'mean')
        self.relu = nn.ReLU()
        
        a = spio.loadmat('data/W-simu', squeeze_me=True)['w1d']
        ns = len(a)
        w = np.zeros((ns,ns))
        for i in range(ns):
            for j in range(ns):
                w[i,j]= a[i]/a[j]
                if w[i,j] > 1:
                    w[i,j] = 1/w[i,j] 
        w[5,:] = w[5,:]/1.5
        w[:,5] = w[:,5]/1.5
        self.weight = torch.tensor(w**1,dtype=torch.float32).to(device)
        
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
        W_loss1 = torch.mean( self.mechanics_loss(dd[:,:45]) )/20
        W_loss2 = torch.mean( self.mechanics_loss(dd[:,45:]) )/20
        
        reconstruction = self.mse(inp, out)
        self.log('mse_loss', reconstruction, on_step=False, on_epoch=True, logger=True)
        print(f'{reconstruction:0.2e}---{W_loss1:0.2e}---{W_loss2:0.2e}')
        return reconstruction + W_loss1 + W_loss2 
    
    def forward(self, inp):
        z = self.encoder(inp)
        out = self.decoder(z) 
        return z, None, out
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=4e-3)
    
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



class AutoEncoder(pl.LightningModule):
    def __init__(self, h_dim):
        super(AutoEncoder, self).__init__()
        self.linear1 = nn.Linear(52, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear22 = nn.Linear(32, h_dim)
        self.linear3 = nn.Linear(h_dim, 64)
        self.linear4 = nn.Linear(64, 52)
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



class AutoEncoderSimu(pl.LightningModule): # a larger net for the simulation task
    def __init__(self, h_dim):
        super(AutoEncoderSimu, self).__init__()
        self.linear1 = nn.Linear(90, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear22 = nn.Linear(32, h_dim)
        self.linear3 = nn.Linear(h_dim, 64)
        self.linear4 = nn.Linear(64, 90)
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
    
    